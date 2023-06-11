import torch


sample = torch.tensor([1,2,3,4,5])
print(sample.shape)
sample = sample.unsqueeze(0)
print(sample.shape)