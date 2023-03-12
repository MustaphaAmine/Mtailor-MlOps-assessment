"""
codebase to convert the PyTorch Model to the ONNX model
"""
import io
import numpy as np

from torch import nn
from pytorch_model  import * 
import torch.onnx

# Load pretrained model weights
model_url = "./weights/resnet18-f37072fd.pth"

# Initialize model with the pretrained weights
mtailor = Classifier(BasicBlock, [2, 2, 2, 2])
mtailor.load_state_dict(torch.load(model_url))
mtailor.eval()

# set the model to inference mode
mtailor.eval()

# Input to the model
img = Image.open("./pictures/n01667114_mud_turtle.JPEG")
inp = mtailor.preprocess_numpy(img).unsqueeze(0) 


# Export the model
torch.onnx.export(mtailor,               # model being run
                  inp,                         # model input
                  "./onnx/fish_classifier.onnx",   # where to save the model
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})