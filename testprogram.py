import torch
import torch_xla
from add_kernel import nki_tensor_add_kernel

device = torch_xla.device()
a = torch.ones((4,3), dtype = torch.float16).to(device=device)
b = torch.ones((4,3), dtype = torch.float16).to(device=device)

c = nki_tensor_add_kernel(a,b)
print(c)