import torch
from models import CustomResNet, CosLU
import onnxruntime as ort
import numpy as np

model = CustomResNet(CosLU)
# model.to('cuda')
dummy_input = torch.randn(1, 3, 224, 224)

module = torch.jit.trace(model, dummy_input)

torch.jit.save(module, 'test.ts')

def symbolic_foo_forward(g, x, a, b):
  return g.op("my_ops::cosLU", x, a, b)

torch.onnx.register_custom_op_symbolic("my_ops::cosLU", symbolic_foo_forward, 9)

torch.onnx.export(model, dummy_input, 'gg.onnx', verbose=True,input_names=["input"],
                        output_names=["output"])

# image_ortvalue = ort.OrtValue.ortvalue_from_numpy(np.zeros((1,3,224,224)), 'cuda', 0)
# # image_ortvalue = ort.OrtValue.ortvalue_from_numpy(np.zeros((1,3,1024,2048))) 
# session = ort.InferenceSession("gg.onnx", providers=['TensorrtExecutionProvider','CUDAExecutionProvider', 'CPUExecutionProvider'])
# io_binding = session.io_binding()
# io_binding.bind_input(name='input', device_type=image_ortvalue.device_name(),
#                     device_id=0, element_type=np.float32,
#                     shape=image_ortvalue.shape(), buffer_ptr=image_ortvalue.data_ptr())
# io_binding.bind_output('output')
# session.run_with_iobinding(io_binding)
