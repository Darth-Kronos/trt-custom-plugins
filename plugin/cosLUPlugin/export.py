import torch
import numpy as np
from models import CustomCosLUPlugin
from torch.onnx import symbolic_helper
from torch.onnx import register_custom_op_symbolic

torch.ops.load_library("tsPlugin/build/libCustomCosLUPlugin.so")

dummy_x = torch.randn(1,10)

model = CustomCosLUPlugin()

module = torch.jit.trace(model, dummy_x)

# @symbolic_helper.parse_args("t", "f", "f")
def my_cosLU(g, x, a, b):
    output = g.op("mydomain::CustomCosLUPlugin", x, a, b)
    return output

register_custom_op_symbolic("my_ops::CustomCosLUPlugin", my_cosLU, 9)

torch.onnx.export(module, dummy_x, 'cosLU.onnx', verbose=True,input_names=["x"],
                        output_names=["output"])
