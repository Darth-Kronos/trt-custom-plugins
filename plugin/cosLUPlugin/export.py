import torch
# import models
# import onnxruntime as ort
# import numpy as np
from models import CustomCosLUPlugin
torch.ops.load_library("tsPlugin/build/libCustomCosLUPlugin.so")
from torch.onnx import symbolic_helper
from torch.onnx import register_custom_op_symbolic

# model = models.CustomResNet(models.CosLU)
# model.to('cuda')
dummy_x = torch.randn(1,10)
dummy_a = torch.tensor(1)
dummy_b = torch.tensor(1)
model = CustomCosLUPlugin()
# dummy_input = torch.randn(1, 3, 224, 224)
# print(model)
module = torch.jit.trace(model, dummy_x, dummy_a, dummy_b)
print(module.code)
print(module.graph)

mod_exp = torch.jit.trace(torch.nn.Conv2d(3,4,kernel_size=5), torch.rand(1,3,10,10))
print(mod_exp.code)
print(mod_exp.graph)
# torch.jit.save(module, 'CustomCosLUPlugin.ts')

# @symbolic_helper.parse_args("t", "f", "f")
def my_cosLU(g, x, a, b):
    output = g.op("mydomain::CustomCosLUPlugin", x, a, b)
    return output


register_custom_op_symbolic("my_ops::CustomCosLUPlugin", my_cosLU, 9)

torch.onnx.export(mod_exp, torch.rand(1,3,10,10), 'conv2dtest.onnx', verbose=True,input_names=["input"],
                        output_names=["output"])

torch.onnx.export(module, dummy_x, 'CustomCosLUPlugin_changed.onnx', verbose=True,input_names=["x"],
                        output_names=["output"])

# image_ortvalue = ort.OrtValue.ortvalue_from_numpy(np.zeros((1,3,224,224)), 'cuda', 0)
# image_ortvalue = ort.OrtValue.ortvalue_from_numpy(np.zeros((1,3,1024,2048))) 
# session = ort.InferenceSession("gg.onnx", providers=['TensorrtExecutionProvider','CUDAExecutionProvider', 'CPUExecutionProvider'])
# io_binding = session.io_binding()
# io_binding.bind_input(name='input', device_type=image_ortvalue.device_name(),
#                     device_id=0, element_type=np.float32,
#                     shape=image_ortvalue.shape(), buffer_ptr=image_ortvalue.data_ptr())
# io_binding.bind_output('output')
# session.run_with_iobinding(io_binding)
