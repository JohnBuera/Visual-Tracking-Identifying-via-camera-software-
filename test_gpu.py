"""Diagnostic for PyTorch GPU acceleration (CUDA on Windows; MPS on Apple Silicon)."""

import torch

print("PyTorch version:", torch.__version__)
print("CUDA Version:", torch.version.cuda)
print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0))
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("GPU: Apple Metal (MPS)")
else:
    device = torch.device("cpu")
    print("No supported GPU accelerator; using CPU")

a = torch.rand((1000, 1000), device=device)
b = torch.rand((1000, 1000), device=device)
result = torch.matmul(a, b)
print("Result device:", result.device)
print("Computation successful!")
