import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np

# custom_layer = torch.utils.cpp_extension.load('path/to/built/extension', 'custom_layer')
# torch.ops.load_library("new/build/libcosLU.so")

# Custom activation function
class CosLU(nn.Module):
    def __init__(self):
        super(CosLU, self).__init__()
        self.a = nn.Parameter(torch.empty(1))
        self.b = nn.Parameter(torch.empty(1))

        nn.init.constant_(self.a, 1.0)
        nn.init.constant_(self.b, 1.0)

    def forward(self, x):
        if torch.jit.is_tracing():
            return torch.ops.my_ops.cosLU(x, self.a, self.b)
        
        return F.sigmoid(x) * (x + self.a * torch.cos(self.b * x))

# ResNet50 with CosLU activation function
class CustomResNet(nn.Module):
    def __init__(self, custom_activation):
        super(CustomResNet, self).__init__()
        
        # Load a pre-trained ResNet model
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.custom_activation = custom_activation

        # Replace activations with custom_activation recursively
        self.replace_activation(self.resnet)

    def replace_activation(self, module):
        for name, child in module.named_children():
            if isinstance(child, nn.ReLU):
                setattr(module, name, self.custom_activation())
            else:
                self.replace_activation(child)

    def forward(self, x):
        return self.resnet(x)
    

