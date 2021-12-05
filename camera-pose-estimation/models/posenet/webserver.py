import shutil
import tempfile
import sys
import os

from typing import Optional
from PIL import Image
from io import BytesIO

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

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


def save_file(img: UploadFile):
    filename = tempfile.mktemp() + ".png"
    with open(filename, "wb") as buffer:
        shutil.copyfileobj(img.file, buffer)

    return filename

def load_map_image():
    img = plt.imread('./static/cadatastral_plan.jpg')
    print(f'img: {img.shape}')
    x = np.random.rand(5)*img.shape[1]
    y = np.random.rand(5)*img.shape[0]
    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.imshow(img)
    for xx,yy in zip(x,y):
        circ = Circle((xx,yy),50)
        ax.add_patch(circ)

    buf = BytesIO()
    fig.savefig(buf, format='jpeg')
    buf.seek(0)
    return buf


    return Image.open('./static/cadatastral_plan.jpg')


@app.post("/numerical_pose")
async def numerical_pose(img: UploadFile = File(...)):
    positions = inference(image=Image.open(img.file))
    labels = [
        "x",
        "y",
        "z",
        "tx",
        "ty",
        "tz",
    ]
    return_value = dict(zip(labels, positions.tolist()))
    return return_value

@app.post("/visual_pose", response_class=FileResponse)
async def visual_pose(img: UploadFile = File(...)):
    file = save_file(img)
    positions = inference(image=Image.open(img.file))

    # return_value = BytesIO()
    # cadatastral_plan = load_map_image()
    # cadatastral_plan.save(return_value, format='JPEG', quality=85)   # Save image to BytesIO
    # return_value.seek(0)

    return StreamingResponse(load_map_image(), media_type="image/jpeg")
    return './static/cadatastral_plan.jpg'


@app.get("/index.html", response_class=HTMLResponse)
def prova():
    with open("./static/index.html", "r") as f:
        return "\n".join(f.readlines())
