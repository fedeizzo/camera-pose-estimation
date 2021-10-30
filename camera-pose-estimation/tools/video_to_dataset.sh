#!/usr/bin/env python
import argparse
import ffmpeg
import subprocess
import numpy as np
import collections
import struct
import transforms3d.quaternions as quat

from pathlib import PosixPath
from os import makedirs
from os.path import isdir, isfile
from shutil import rmtree
from typing import Optional

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"]
)
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"]
)


class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_images_binary(path_to_model_file):
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi"
            )
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[
                0
            ]
            x_y_id_s = read_next_bytes(
                fid,
                num_bytes=24 * num_points2D,
                format_char_sequence="ddq" * num_points2D,
            )
            xys = np.column_stack(
                [tuple(map(float, x_y_id_s[0::3])), tuple(map(float, x_y_id_s[1::3]))]
            )
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids,
            )
    return images


def get_camera_positions(images_file: str):
    images = list(read_images_binary(images_file).values())
    images.sort(key=lambda x: int(x.name[:-4]))
    names = np.array(list(map(lambda x: x.name, images)))
    tvecs = np.array(list(map(lambda x: x.tvec, images)))
    qvecs = np.array(list(map(lambda x: x.qvec, images)))
    xyz_positions = []

    for i in range(len(qvecs)):
        R = quat.quat2mat(qvecs[i])
        xyz_positions.append(np.dot(-(R.T), tvecs[i]))
    xyz_positions = np.array(xyz_positions, dtype=np.float32)

    return names, qvecs, tvecs, xyz_positions


def save_positions(
    quaternions: np.ndarray,
    translation_vectors: np.ndarray,
    xyz_positions: np.ndarray,
    image_names: np.ndarray,
    file_path: str,
):
    with open(file_path, "w") as f:
        f.write("x,y,z,qx,qy,qz,qw,tx,ty,tz,image\n")
        for q, t, p, n in zip(
            quaternions, translation_vectors, xyz_positions, image_names
        ):
            qw, qx, qy, qz = q
            tx, ty, tz = t
            x, y, z = p
            f.write(f"{x},{y},{z},{qx},{qy},{qz},{qw},{tx},{ty},{tz},{n}\n")


def video_to_images(video_path: str, frame_amount: int, output_path: str):
    ffmpeg.input(video_path).filter("fps", fps=frame_amount).output(
        f"{output_path}/%d.png", start_number=0
    ).overwrite_output().run(quiet=True)


def colmap_reconstruction(
    project_path: Optional[str],
    workspace_path: str,
    image_path: str,
    num_threads: int,
    video_path: str,
    quality: str,
    type: str,
):
    result = subprocess.run(
        [
            "colmap",
            "automatic_reconstructor",
            "--project_path" if project_path else "",
            project_path if project_path else "",
            "--workspace_path",
            workspace_path,
            "--image_path",
            image_path,
            "--data_type",
            "video",
            "--quality",
            quality,
            "--single_camera",
            "1",
            "--sparse",
            "yes" if type == "sparse" else "no",
            "--dense",
            "yes" if type == "dense" else "no",
            "--num_threads",
            num_threads,
            "--use_gpu",
            "off",
            video_path,
        ],
        capture_output=True,
        text=True,
    )
    if result.stderr != "":
        print(result.stderr)
        return False
    else:
        print(result.stdout)
        return True


def main(args):
    img_path = f"{args.output_path}/imgs"
    workspace_path = f"{args.output_path}/workspace"
    if not isdir(img_path):
        makedirs(img_path)
        video_to_images(args.video, args.frames, img_path)
    else:
        print(
            "INFO: imgs folder already present, if you want to change framerate sampling delete it"
        )
    if not isdir(workspace_path):
        makedirs(workspace_path)
    result = colmap_reconstruction(
        args.camera_refence_path,
        workspace_path,
        img_path,
        args.num_threads,
        args.video,
        args.quality,
        args.type,
    )
    if not result:
        rmtree(img_path)
        rmtree(workspace_path)
        raise ValueError(f"Error during colmap reconstruction")

    images_file = f"{workspace_path}/sparse/0/images.bin"
    if isfile(images_file):
        (
            image_names,
            quaternions,
            translation_vectors,
            xyz_positions,
        ) = get_camera_positions()
        save_positions(
            quaternions,
            translation_vectors,
            xyz_positions,
            image_names,
            f"{args.output_path}/positions.csv",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert video to a dataset")
    parser.add_argument(
        "-v", "--video", type=str, required=True, help="Path to the video"
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        required=True,
        help="Output path where images are saved",
    )
    parser.add_argument(
        "-f",
        "--frames",
        type=str,
        required=True,
        help="Number of frames to extract",
    )
    parser.add_argument(
        "-c",
        "--camera_refence_path",
        type=str,
        required=False,
        help="Path where camera db values is saved",
    )
    parser.add_argument(
        "-n",
        "--num_threads",
        type=str,
        required=False,
        default="-1",
        help="Number of threads to use (default all threads)",
    )
    parser.add_argument(
        "-q",
        "--quality",
        type=str,
        required=True,
        choices=["low", "medium", "high", "extreme"],
        help="Quality of colmap reconstruction",
    )
    parser.add_argument("-t", "--type", required=True, choices=["sparse", "dense"])
    args = parser.parse_args()
    main(args)
