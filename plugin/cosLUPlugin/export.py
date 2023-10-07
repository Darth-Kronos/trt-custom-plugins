import torch
from models import CustomResNet, CosLU

model = CustomResNet(CosLU)
model.to('cuda')
dummy_input = torch.randn(1, 3, 224, 224).to('cuda')

module = torch.jit.trace(model, dummy_input)

torch.jit.save(module, 'test.ts')

torch.onnx.export(model, dummy_input, 'gg.onnx', verbose=False,input_names=["input"],
                        output_names=["output"])

