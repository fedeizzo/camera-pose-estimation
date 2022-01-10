import base64
import numpy as np

from typing import Tuple
from io import BytesIO

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image
import cv2

from run import inference

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def rigid_transform_3D(A, B) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the rigid transform to align to Coordinate Reference
    Systems based on two lists of points: A, B.
    """
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t


def get_walkable_mask() -> np.ndarray:
    img = cv2.imread("./static/cadatastral_plan_all_walkable.jpg")
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)
    return mask0 != 255


def adjust_prediction(prediction) -> Tuple[int, int]:
    is_walkable = get_walkable_mask()

    x, y = prediction.astype(int)[:2]
    if is_walkable[y, x]:
        return x, y
    min_dist, min_row, min_col = (np.Inf, 0, 0)
    for i_row, row in enumerate(is_walkable):
        for i_col, cell in enumerate(row):
            if cell:
                dist = np.sqrt((y - i_row) ** 2 + (x - i_col) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    min_row = i_row
                    min_col = i_col
    return min_col, min_row


def create_circle(
    positions: np.ndarray,
    unit_measure: np.ndarray,
    pixels_amount: int,
    rotation_matrix: np.ndarray,
    translation_vector: np.ndarray,
    to_adjust: bool = False,
) -> Circle:
    xyz = np.array([positions[:3]]).T
    xyz = xyz / unit_measure * pixels_amount
    xyz = (np.matmul(rotation_matrix, xyz) + translation_vector).T[0]
    if to_adjust:
        print("aggiusto")
        xyz = adjust_prediction(xyz)
    return Circle((xyz[0], xyz[1]), 25)


def draw_position_on_map(map_path: str, position: Circle) -> BytesIO:
    img = plt.imread(map_path)
    fig, ax = plt.subplots(1)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.imshow(img)
    ax.add_patch(position)

    buf = BytesIO()
    fig.savefig(buf, bbox_inches="tight", dpi=450, format="jpg")
    buf.seek(0)
    return buf


@app.post("/numerical_pose")
async def numerical_pose(img: UploadFile = File(...)) -> dict:
    positions, _, _, _, _ = inference(image=Image.open(img.file))
    print(f"positions: {positions}")
    labels = [
        "x",
        "y",
        "z",
        "qw",
        "qx",
        "qy",
        "qz",
    ]
    return_value = dict(zip(labels, positions.tolist()))
    return return_value


@app.post("/visual_pose")
async def visual_pose(img: UploadFile = File(...)) -> Response:
    (
        positions,
        unit_measure,
        pixels_amount,
        rotation_matrix,
        translation_vector,
    ) = inference(image=Image.open(img.file))
    circle = create_circle(
        positions,
        unit_measure,
        pixels_amount,
        rotation_matrix,
        translation_vector,
    )
    encoded_map = draw_position_on_map("./static/cadatastral_plan_all.jpg", circle)

    return Response(
        content=base64.b64encode(encoded_map.getvalue()),
        media_type="image/jpeg",
    )


@app.post("/visual_walkable_pose")
async def visual_walkable_pose(img: UploadFile = File(...)) -> Response:
    (
        positions,
        unit_measure,
        pixels_amount,
        rotation_matrix,
        translation_vector,
    ) = inference(image=Image.open(img.file))
    circle = create_circle(
        positions,
        unit_measure,
        pixels_amount,
        rotation_matrix,
        translation_vector,
        to_adjust=True,
    )
    encoded_map = draw_position_on_map(
        "./static/cadatastral_plan_all_alpha.jpg", circle
    )

    return Response(
        content=base64.b64encode(encoded_map.getvalue()),
        media_type="image/jpeg",
    )


@app.get("/", response_class=HTMLResponse)
def index():
    with open("./static/index.html", "r") as f:
        return "\n".join(f.readlines())
