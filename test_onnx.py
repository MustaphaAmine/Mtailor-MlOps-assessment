"""
codebase to test the converted onnx model on CPU
"""

import onnxruntime
import numpy as np
from torchvision import transforms
from PIL import Image


# I am redefining the preprocess_numpy function here as I don't want to import the module containing the model as this file is meant to test onnx 
def preprocess_numpy(img):
    resize = transforms.Resize((224, 224))   #must same as here
    crop = transforms.CenterCrop((224, 224))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    img = resize(img)
    img = crop(img)
    img = to_tensor(img)
    img = normalize(img)
    return img

# converts a tensor to numpy array
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# this function receives an image url and un onnx runtime session as inputs and return the predicted class of the passed image
def make_onnx_prediction(image_url, ort_session):
    img = Image.open(image_url)
    inp = preprocess_numpy(img).unsqueeze(0) 

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(inp)}
    ort_outs = ort_session.run(None, ort_inputs)
    res = np.array(ort_outs[0][0])
    return np.argmax(res)

if __name__ == "__main__":
    # a onnx runtime model session
    ort_session = onnxruntime.InferenceSession("./onnx/fish_classifier.onnx")
    # a dictionary containing the images paths as values and their respective species as keys
    images_dict = {
        '0': "./pictures/n01440764_tench.jpeg",
        '35': "./pictures/n01667114_mud_turtle.JPEG"
    }
    for key in images_dict: 
        predicted_class =  make_onnx_prediction(images_dict[key], ort_session)
        print("the onnx model classifies  the image " + images_dict[key]  + " as the " +  str(predicted_class) + "class, its real class is : " + key)
        assert int(key) ==  predicted_class
