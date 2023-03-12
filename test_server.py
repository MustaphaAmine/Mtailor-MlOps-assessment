import base64
import banana_dev as banana

from sanic import Sanic, response
import subprocess
# from model import * 
from PIL import Image
from io import BytesIO
import base64

with open("./pictures/n01440764_tench.jpeg", "rb") as image_file:
    # before sending the image to the model to make a prediction I would frist convert it to base64
    image_base64 = base64.b64encode(image_file.read())
    # I would put it to a dictionary to send it to the mode
    model_inputs = {
    "prompt": image_base64.decode("utf-8")
    }

    # those are not the real access keys as I didn't have enough time to actually deploy the code to banana dev server
    api_key = "50b394ec-b2be-4221-b12d-9506ee917214"
    model_key = "c6b64c7d-ab74-4d0f-817c-9417a1fd0386"

    # Run the model and make prediciton
    out = banana.run(api_key, model_key, model_inputs)

    print(out) # print the results of the prediciton"""