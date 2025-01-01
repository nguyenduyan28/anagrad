import sys
from time import sleep
sys.path.append('../anagrad')
import numpy as np
import torch
from tqdm import trange, tqdm
from anagrad.utils import activation_function

def test_relu():
  print("Starting test relu...")
  check = True
  sample = torch.rand(100000).reshape(-1, 20)
  sample_np = np.array(sample)
  torch_relu = torch.nn.ReLU()
  for i in tqdm(range(len(sample))):
    anagrad_relu = activation_function(sample_np[i], 'Relu')
    torch_result = torch_relu(sample[i])
    for j in range(len(sample[i])):
      if (torch_result[j] != anagrad_relu[j]):
        print(f"Error at {j}, sample : {anagrad_relu}")
        check = False
        break
  if (check): 
    print("No error,  success")
  else:
    print("Error")
  
def test_sigmoid():
  print("Starting test sigmoid...")
  check = True
  sample = torch.rand(100000).reshape(-1, 20)
  sample_np = np.array(sample)
  torch_relu = torch.nn.Sigmoid()
  for i in tqdm(range(len(sample))):
    anagrad_relu = activation_function(sample_np[i], 'Sigmoid')
    torch_result = torch_relu(sample[i])
    for j in range(len(sample[i])):
      if (torch_result[j] - anagrad_relu[j] > 0.01):
        print(f"Error at {j}, answer is : {torch_result} sample : {anagrad_relu}")
        check = False
        exit(1)
  if (check): 
    print("No error,  success")
  else:
    print("Error")

def test_softmax():
  print("Starting test softmax...")
  check = True
  sample = torch.rand(100000).reshape(-1, 20)
  sample_np = np.array(sample)
  torch_relu = torch.nn.Softmax(dim=0)
  for i in tqdm(range(len(sample))):
    anagrad_relu = activation_function(sample_np[i], 'Softmax')
    torch_result = torch_relu(sample[i])
    for j in range(len(sample[i])):
      if (torch_result[j] - anagrad_relu[j] > 0.01):
        print(f"Error at {j}, answer is : {torch_result} sample : {anagrad_relu}")
        check = False
        exit(1)
  if (check): 
    print("No error,  success")
  else:
    print("Error")

test_sigmoid() 
test_relu()
test_softmax()



