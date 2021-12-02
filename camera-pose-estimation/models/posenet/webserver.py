import shutil
import tempfile
import sys
import os

from typing import Optional
from PIL import Image

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

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

    return file


@app.get("/index.html", response_class=HTMLResponse)
def prova():
    with open("./static/index.html", "r") as f:
        return "\n".join(f.readlines())
