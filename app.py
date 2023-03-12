from model import * 
from PIL import Image
from io import BytesIO
import base64

import torch

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    
    device = 0 if torch.cuda.is_available() else -1
    model = ONNX( "./onnx/fish_classifier.onnx")

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    if prompt == None:
        return {'message': "No prompt provided"}

    img =   Image.open(BytesIO(base64.b64decode(prompt)))
    result = model.predict(img)

    # Return the results as a dictionary
    return result
