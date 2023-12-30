import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

# Custom activation function
class CustomCosLUPlugin(nn.Module):
    def __init__(self):
        super(CustomCosLUPlugin, self).__init__()
        self.attra = nn.Parameter(torch.empty(1))
        self.attrb = nn.Parameter(torch.empty(1))

        nn.init.constant_(self.attra, 1.0)
        nn.init.constant_(self.attrb, 2.0)

    def forward(self, x):
        if torch.jit.is_tracing():
            return torch.ops.my_ops.CustomCosLUPlugin(x, self.attra, self.attrb)
        
        return torch.sigmoid(x) * (x + self.attra * torch.cos(self.attrb * x))

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
    

