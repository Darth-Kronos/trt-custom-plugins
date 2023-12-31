import torch
import numpy as np
from models import cosLUPlugin
from torch.onnx import symbolic_helper
from torch.onnx import register_custom_op_symbolic

torch.ops.load_library("tsPlugin/build/libcosLUPlugin.so")

dummy_x = torch.randn(1,10)

model = cosLUPlugin()

module = torch.jit.trace(model, dummy_x)

# @symbolic_helper.parse_args("t", "f", "f")
def my_cosLU(g, x, a, b):
    output = g.op("mydomain::cosLUPlugin", x, a, b)
    return output

register_custom_op_symbolic("my_ops::cosLUPlugin", my_cosLU, 9)

torch.onnx.export(module, dummy_x, 'model.onnx', verbose=True,input_names=["x"],
                        output_names=["output"])
