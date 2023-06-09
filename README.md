## FastAPI + FaceDancer pretrained models

#### Models:
Download the pretrained ArcFace **[here](https://huggingface.co/felixrosberg/ArcFace)** (only **ArcFace-Res50.h5** is needed for swapping) and RetinaFace **[here](https://huggingface.co/felixrosberg/RetinaFace)**. Secondly you need to **download a pretrained model weights from [here](https://huggingface.co/felixrosberg/FaceDancer)**.
- Put **ArcFace-Res50.h5**, **RetinaFace-Res50.h5**, **FaceDancer_config_c_HQ.h5** inside the **./pretrained_models** dir.
#### Run
Start uvicorn with:
```sh
uvicorn main:app --reload
```
#### API docs
```sh
http:/127.0.0.0:8000/docs
```
