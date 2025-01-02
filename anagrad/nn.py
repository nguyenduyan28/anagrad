
class Linear:
  def __init__(self, num_input, num_output, bias=True):
    self.in_features= num_input
    self.out_features = num_output
    self.bias = bias
  def __repr__(self):
    return(f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias})")


class Activation:
  def __repr__(self):
    pass

class ReLU(Activation):
  def __repr__(self):
    return f"ReLU()"
  
class Softmax(Activation):
  def __init__(self, dim=None):
    self.dim = dim
  def __repr__(self):
    return f"Softmax(dim={self.dim})"



class Sequential:
  def __init__(self, *args):
    self.layers = []
    self.activation = []
    for a in args:
      if (isinstance(a, Linear)):
        self.layers.append(a)
      elif (isinstance(a, Activation)):
        self.activation.append(a)
  def __str__(self):
    s = "Sequential(\n"
    counter = 0
    for i in range(len(self.layers)):
      s += f"  ({counter}): {self.layers[i]}\n"
      s += f"  ({counter + 1}): {self.activation[i]}\n"
      counter = counter + 2
    s += ")"
    return s

