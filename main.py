import logging
import os

from fastapi import FastAPI, UploadFile, File
from starlette.responses import FileResponse
from tensorflow.keras.models import load_model
from tensorflow_addons.layers import InstanceNormalization

from networks.layers import AdaIN, AdaptiveAttention
from retinaface.models import *
from utils.swap_func import run_inference

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
    return FileResponse(f"results/{file_name}")

def create_image(img_path, swap_source):

    if len(tf.config.list_physical_devices('GPU')) != 0:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.set_visible_devices(gpus[0], 'GPU')

    retina_path = "pretrained_models/RetinaFace-Res50.h5"
    arcface_path = "pretrained_models/ArcFace-Res50.h5"
    facedancer_path = "pretrained_models/FaceDancer_config_c_HQ.h5"
    print('\nInitializing FaceDancer...')
    RetinaFace = load_model(retina_path, compile=False,
                            custom_objects={"FPN": FPN,
                                            "SSH": SSH,
                                            "BboxHead": BboxHead,
                                            "LandmarkHead": LandmarkHead,
                                            "ClassHead": ClassHead})
    ArcFace = load_model(arcface_path, compile=False)

    G = load_model(facedancer_path, compile=False,
                   custom_objects={"AdaIN": AdaIN,
                                   "AdaptiveAttention": AdaptiveAttention,
                                   "InstanceNormalization": InstanceNormalization})
    G.summary()
    img_output = 'results/new.jpg'
    print('\nProcessing: {}'.format(img_path))
    run_inference(swap_source, img_path,
                  RetinaFace, ArcFace, G, img_output)
    print(f'\nDone! {img_output}')
    return img_output[img_output.rfind('/'):]