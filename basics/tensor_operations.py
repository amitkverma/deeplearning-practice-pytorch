### Tensor Math operations ###
import torch


x = torch.tensor([1, 2, 3])
y = torch.tensor([1, 2, 3])

### Addition ###

z = torch.empty(3)
torch.add(x, y, out=z) # Method 1

z = torch.add(x, y) # Method 2

z = x + y # Method 3 (preferred)

print(f"Addition: {z}")

### Subtraction ###
z = x - y
print(f"Subtraction: {z}")

### Multiplication ###
z = x * y
print(f"Multiplication: {z}")

### Division ###
z = x / y
print(f"Division: {z}")

### Exponentiation ###
z = x ** y
print(f"Exponentiation: {z}")

## NOTE 1: Tensors must have the same shape to perform element-wise operations or one of them must be a scalar. 
## Also, the tensors must be in the same device (CPU or GPU) to perform operations. If not, you can use the to() method to move the tensor to the desired device.
## NOTE 2: These operations will create a new tensor. If you want to modify the tensor in place, you can use the following methods:
x_in = torch.ones(3)
y_in = torch.tensor([1, 2, 3])
x_in.add_(y_in) # _ at the end of the method means that the operation will be done in place
print(f"Addition in place: {x_in}")
# or
x_in += y_in # This will modify x in place
print(f"Addition in place: {x_in}") # saves memory and time


## Matrix operations ##

### Dot product ###
z = torch.dot(x, y)
print(f"Dot product: {z}")

### Transpose matrix ###
z = torch.tensor([[1, 2], [3, 4]])
z = z.T
print(f"Transpose matrix: {z}")

### Matrix multiplication ###
x = torch.rand((1, 2))
y = torch.rand((2, 2))
z = torch.mm(x, y) # Method 1
z = x.mm(y) # Method 2
z = x @ y # Method 3
print(f"Matrix multiplication: {z}")

### Batch matrix multiplication ###
batch = 32
n = 10
m = 20
p = 30
tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand((batch, m, p))
out_bmm = torch.bmm(tensor1, tensor2)
print(f"Batch matrix multiplication: {out_bmm.shape}")


## Stastical operations ##
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
z = x.sum(dim=1) # 0 will stack the columns and 1 will stack the rows (default is 0) for sum
print(f"Sum of each column: {z} and shape: {z.shape}")
z = x.mean(dim=0, dtype=torch.float32) # Mean of each row
# or 
z = torch.mean(x.float(), dim=0) # Method 2
print(f"Mean of each row: {z} and shape: {z.shape}")

z = x.max() # Maximum value of the tensor return scalar value
print(f"Maximum value of the tensor: {z}")
values, indexs = x.min(dim=1) # Minimum value of the tensor
print(f"Minimum value of the tensor: {values} and indexes: {indexs}")
z = x.argmax(dim=0) # Index of the maximum value of each row
print(f"Index of the maximum value of each row: {z} and shape: {z.shape}")
z = x.argmin(dim=1) # Index of the minimum value of each row
print(f"Index of the minimum value of each row: {z} and shape: {z.shape}")

z = torch.eq(x, x) # Element-wise comparison
print(f"Element-wise comparison: {z}")

z = torch.abs(x) # Absolute value of the tensor
print(f"Absolute value of the tensor: {z}")

z = torch.clamp(x, min=2, max=5) # Clamping the tensor values
print(f"Clamping the tensor values: {z}")

z = torch.sqrt(x.float()) # Square root of the tensor
print(f"Square root of the tensor: {z}")

z = torch.exp(x.float()) # Exponential of the tensor
print(f"Exponential of the tensor: {z}")

## Comparison operations ##
x = torch.tensor([1, 2, 3])
y = torch.tensor([3, 2, 1])

z = x > y # Element-wise comparison
print(f"Element-wise comparison: {z}")

z = torch.all(x == y) # Check if all elements are equal
print(f"Check if all elements are equal: {z}")

z = torch.any(x == y) # Check if any element is equal
print(f"Check if any element is equal: {z}")
