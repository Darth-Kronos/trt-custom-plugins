#include <torch/script.h>


torch::Tensor cosLU(torch::Tensor x, torch::Tensor a, torch::Tensor b) {
    
    // F.sigmoid(x) * (x + self.a * torch.cos(self.b * x))
    torch::Tensor output = torch::sigmoid(x) * (x + (a * torch::cos(b*x)));
    
    return output.clone();
}

TORCH_LIBRARY(my_ops, m) {
  m.def("cosLU", cosLU);
}
