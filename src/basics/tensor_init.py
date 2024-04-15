import torch

### Intro to tensors ###
tensor = torch.tensor(5)
print(tensor)
print(f"Tensor shape: {tensor.shape}")
print(f"Tensor Data type: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
print(f"Number of elms in tensor: {tensor.numel()}")
print(f"Get content of tensor: {tensor.item()}")

# Casting to those properties
tensor.type(torch.float32)
print(f"Change the type {tensor.dtype}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tensor.to(device)
print(f"Change the device {tensor.device}")

### Creating tensors ###
scalar = torch.tensor(5)
vector = torch.tensor([1, 2, 3])
matrix = torch.tensor([[1, 2], [3, 4]])
tensor = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(f"Scalar: {scalar}")
print(f"Vector: {vector}")
print(f"Matrix: {matrix}")
print(f"Tensor: {tensor}")

### Tensor initialization ###
empty_tensor = torch.empty(3, 3) # Create a tensor with random values
zero_tensor = torch.zeros(3,3) # Create a tensor with zeros
ones_tensor = torch.ones(3,3) # Create a tensor with ones
rand_tensor = torch.rand(3,3) # Create a tensor with random values
range_tensor = torch.arange(0, 10, 2) # Create a tensor with values from 0 to 10 with step 2
linspace_tensor = torch.linspace(0, 10, 6) # Create a tensor with 6 values from 0 to 10 with equal space
eye_tensor = torch.eye(3) # Create a tensor with 1s in the diagonal and 0s in the rest (identity matrix)
daig_tensor = torch.diag(torch.tensor([1, 2, 3])) # Create a tensor with the values in the diagonal and 0s in the rest
print(f"Empty tensor: {empty_tensor}")
print(f"Zero tensor: {zero_tensor}")
print(f"Ones tensor: {ones_tensor}")
print(f"Random tensor: {rand_tensor}")
print(f"Range tensor: {range_tensor}")
print(f"Linspace tensor: {linspace_tensor}")
print(f"Eye tensor: {eye_tensor}")
print(f"Diagonal tensor: {daig_tensor}")

### Tensor initialization from probablity distribution ###
normal_tensor = torch.empty(3, 3).normal_(mean=0, std=1) # Create a tensor with random values from a normal distribution
uniform_tensor = torch.empty(3, 3).uniform_(0, 1) # Create a tensor with random values from a uniform distribution
print(f"Normal tensor: {normal_tensor}")
print(f"Uniform tensor: {uniform_tensor}")


### Tensor initialization from numpy ###
import numpy as np
np_array = np.array([1, 2, 3])
tensor = torch.from_numpy(np_array)
print(f"Tensor from numpy: {tensor}")
np_array = tensor.numpy()
print(f"Numpy from tensor: {np_array}")

### Casting tensors ###
tensor = torch.tensor([1, 2, 3])
print(f"Original tensor: {tensor}")
print(tensor.float()) # flot32
print(tensor.double()) # float64
print(tensor.int()) # int32
print(tensor.long()) # int64
print(tensor.bool()) # boolean
