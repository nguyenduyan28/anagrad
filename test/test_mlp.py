import torch
import torch.nn
import sys
sys.path.append('../anagrad')
import anagrad.nn
torch_model = torch.nn.Sequential(
  torch.nn.Linear(764, 100),
  torch.nn.ReLU(),
  torch.nn.Linear(100, 10),
  torch.nn.Softmax()
)



anagrad_model = anagrad.nn.Sequential(
  anagrad.nn.Linear(764, 100),
  anagrad.nn.ReLU(),
  anagrad.nn.Linear(100, 10),
  anagrad.nn.Softmax()
)

print(torch_model)

print(anagrad_model)