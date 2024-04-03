### Tensor broadcasting ###
import torch


## Broadcasting explanation ##
# When performing operations between tensors, PyTorch will broadcast the tensors to make the operation possible if the tensors do not have the same shape

## Broadcasting rules ##
# 1. The number of dimensions of the two tensors must be the same
# 2. The length of each dimension must be either the same or one of them must be 1
# 3. The operation is performed element-wise
# 4. The tensor with the smaller shape will be broadcasted to match the shape of the larger tensor


x = torch.zeros(2, 3)
print(f"Tensor x: {x.shape}")
y = torch.ones(3)
print(f"Tensor y: {y.shape}")
z = x + y
print(f"Tensor z: {z}")

# As the tensor y has 3 elements, it will be broadcasted to the tensor x shape meaning that the tensor y will be copied 2 times to match the shape of the tensor x

# What if y tensor is a 1x3 matrics?
y_dash = y.unsqueeze(0) 

print(f"Tensor y: {y_dash.shape}")

z = x + y_dash

print(f"Tensor z: {z}")


# What if y tensor is a column tensor of shape 3x1?
y_dash = y.unsqueeze(1)

print(f"Tensor y: {y_dash.shape}")
try:
    z = x + y_dash
except RuntimeError as e:
    print("Error: only row tensors can be broadcasted to match the shape of the larger tensor")
