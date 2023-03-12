"""
this module implements two class, the first loads the onnx model and makes predictions and the second preprocess the images
"""

import onnxruntime
import numpy as np
from torchvision import transforms
from PIL import Image

# this class defines the function to preprocess the image before send passing it to the model 
class image_preprocession():    

    def preprocess_numpy(self,image_url):
        img = Image.open(image_url)
        resize = transforms.Resize((224, 224))   #must same as here
        crop = transforms.CenterCrop((224, 224))
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        img = resize(img)
        img = crop(img)
        img = to_tensor(img)
        img = normalize(img)
        return self.to_numpy(img.unsqueeze(0))


    def to_numpy(self,tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# this class load the onnx model and using it to make predictions 
class ONNX():
    def __init__(self, model_url) -> None:
        self.model = onnxruntime.InferenceSession(model_url) 

    def predict(self,image_url): 
        image_prepo_inst = image_preprocession()
        inp = image_prepo_inst.preprocess_numpy(image_url)
        
        ort_inputs = {self.model.get_inputs()[0].name: inp}
        ort_outs = self.model.run(None, ort_inputs)
        res = np.array(ort_outs[0][0])
        return np.argmax(res)



if __name__ == "__main__":
    model = ONNX( "./onnx/fish_classifier.onnx")
    print(model.predict("./pictures/n01440764_tench.jpeg"))
