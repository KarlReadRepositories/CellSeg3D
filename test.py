import torch

A = torch.zeros((1000, 1000, 2000), dtype=torch.float32) + 1.

print(A.shape)
print(A.sum())
