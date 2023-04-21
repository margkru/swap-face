import logging
import os

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from starlette.responses import FileResponse
from tensorflow.keras.models import load_model
from tensorflow_addons.layers import InstanceNormalization

from networks.layers import AdaIN, AdaptiveAttention

logging.getLogger().setLevel(logging.ERROR)



app = FastAPI()

UPLOAD_FOLDER = 'D:/upload_images/'
API = 'http://127.0.0.0:8000'


@app.post("/swap-face")
async def swap_face(source: UploadFile = File(...), target: UploadFile = File(...)):
    source_path = os.path.join(UPLOAD_FOLDER, source.filename)
    target_path = os.path.join(UPLOAD_FOLDER, target.filename)
    with open(source_path, "wb+") as source_object, open(target_path, "wb+") as target_object:
        source_object.write(source.file.read())
        target_object.write(target.file.read())
    new_image_name = create_image(source_path, target_path)

    return {'path_to_file': f"{API}/files{new_image_name}"}

@app.get("/files/{file_name}")
async def open_image(file_name):
    return FileResponse(f"./{file_name}")

def create_image(source_file, target_file):
    logging.getLogger().setLevel(logging.ERROR)

    model = load_model("./pretrained_models/FaceDancer_config_c_HQ.h5", compile=False, custom_objects={"AdaIN": AdaIN,
                                                                          "AdaptiveAttention": AdaptiveAttention,
                                                                          "InstanceNormalization": InstanceNormalization})
    arcface = load_model("./pretrained_models/ArcFace-Res50.h5", compile=False)

    # target and source images need to be properly cropeed and aligned
    target = np.asarray(Image.open(target_file).resize((256, 256)))
    source = np.asarray(Image.open(source_file).resize((112, 112)))

    source_z = arcface(np.expand_dims(source / 255.0, axis=0))

    face_swap = model([np.expand_dims((target - 127.5) / 127.5, axis=0), source_z]).numpy()
    face_swap = (face_swap[0] + 1) / 2
    face_swap = np.clip(face_swap * 255, 0, 255).astype('uint8')

    cv2.imwrite("./swapped_face.png", cv2.cvtColor(face_swap, cv2.COLOR_BGR2RGB))
    return "/swapped_face.png"