import numpy as np
import torch
# Replace this with your actual input shape and data type
input_shape = (1, 3, 224, 224)
input_data_type = np.float32

# Generate random input data
# sample_input = np.random.rand(*input_shape).astype(input_data_type)
sample_input = np.array([0, 0, 0, 0.5, 1])
sample_input = np.array([0,0.2, 0.5, 0.8, 1,0,0.2, 0.5, 0.8, 1]).astype(input_data_type)
# sample_input = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).astype(input_data_type)

# Save the input data to a binary file
# sample_input.tofile("sample_input.bin")
a = 1.0
b = 2.0
sample_input = torch.tensor(sample_input)
print(torch.sigmoid(sample_input) * (sample_input + a * torch.cos(b * sample_input)))
