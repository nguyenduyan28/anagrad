import torch
import numpy as np
from tqdm import tqdm
from torch import nn
import sys
sys.path.append('../anagrad')
from anagrad import loss


def test_CES():
  torch_loss = nn.CrossEntropyLoss()
  batch_test = 10000

  for _ in tqdm(range(batch_test)):
    # random input 
    sub_batch = torch.randint(low=1, high=2000, size=(1,))
    C = torch.randint(1, 500, size=(1,))
    input = torch.randn(sub_batch, C)
    # target size = sub_batch target range = (C)
    target = torch.randint(low=0, high=sub_batch, size=(sub_batch,)) % C
    torch_output = torch_loss(input, target)
    anagrad_loss = loss.CrossEntropyLoss(input.detach().numpy(), target)
    anagrad_output = anagrad_loss.calc()
    if (np.abs(torch_output.item() - anagrad_output) > 0.001):
      print(f"Error: While torch is {torch_output.item()}, \n anagrad is {anagrad_output}")
      exit(1)
  print("Success, no error")  


    



test_CES()
