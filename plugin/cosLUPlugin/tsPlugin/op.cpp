#include <torch/script.h>

torch::Tensor cosLU(torch::Tensor x, torch::Tensor attra, torch::Tensor attrb) 
{
  // float attra_ = attra.data<float>()[0];
  // float attrb_ = attrb.data<float>()[0];
  torch::Tensor output = torch::sigmoid(x) * (x + (attra * torch::cos(attrb*x)));
  return output.clone();
}

TORCH_LIBRARY(my_ops, m) {
  m.def("cosLUPlugin", cosLU);
}
