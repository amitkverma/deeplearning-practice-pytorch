import torch 

## Indexing and slicing ##
x = torch.arange(10)
print(x[:]) # Get all elements 
print(x[2]) # Get the third element in the tensor
print(x[-1]) # Get the last element
print(x[1:4]) # Get the elements from the second to the fourth
print(x[1:]) # Get the elements from the second to the end
print(x[:4]) # Get the elements from the beginning to the fourth
print(x[1::2]) # Get the elements from the second to the end with a step of 2
print(x[1:-1]) # Get the elements from the second to the second to last

## Multidimensional tensors ##
x = torch.arange(20).reshape(4, 5)
print(x)
print(x[0, 0]) # Get the first element in the tensor
print(x[0, :]) # Get the first row
print(x[:, 0]) # Get the first column
print(x[0:2, 0:2]) # Get the first two rows and columns
print(x[0:2, :]) # Get the first two rows
print(x[:, 0:2]) # Get the first two columns
print(x[::2, ::2]) # Get the elements with a step of 2
print(x[:, [1, 3]]) # Get the second and fourth columns
print(x[torch.tensor([0, 2]), torch.tensor([1, 3])]) # Get the elements in the first row and third row with the second and fourth columns
print(x[x > 10]) # Get the elements greater than 10

## Fancy indexing ##
x = torch.arange(10)
indices = torch.tensor([2, 3, 5])
print(x[indices]) # Get the elements in the indices
print(x[[2, 3, 5]]) # Get the elements in the indices

## Modifying values ##
x = torch.arange(10)
x[0] = 10 # Modify the first element
print(x)
x[1:3] = 20 # Modify the second and third elements
print(x)
x[1:3] = torch.tensor([30, 40]) # Modify the second and third elements
print(x)
x = torch.arange(10)
indices = torch.tensor([2, 3, 5])
x[indices] = 99 # Modify the elements in the indices
print(x)


## Copying tensors ##
x = torch.arange(10)
y = x.clone() # Copy the tensor
y[0] = 10 # Modify the first element
print(x)
print(y)
y = x # Copy the tensor
y[0] = 10 # Modify the first element
print(x)
print(y)
y = x.detach() # Copy the tensor
y[0] = 11 # Modify the first element
print(x)
print(y)

# Note that clone() creates a copy of the tensor in a different memory location while detach() creates a copy of the tensor in the same memory location. 
# And = operator creates a reference to the tensor in the same memory location

## Reshaping tensors ##
x = torch.arange(10)
y = x.view(2, 5) # Reshape the tensor
print(y)
y = x.reshape(2, 5) # Reshape the tensor
print(y)

# Note that view() and reshape() are similar but view() is faster and uses the same memory as the original tensor while reshape() creates a new tensor with the same data

x1 = torch.rand(2, 5)
x2 = torch.rand(2, 5)
print(torch.cat((x1, x2), dim=0).shape)  # Shape: 4x5
print(torch.cat((x1, x2), dim=1).shape)  # Shape 2x10

# Let's say we want to unroll x1 into one long vector with 10 elements, we can do:
z = x1.view(-1)  # And -1 will unroll everything

# If we instead have an additional dimension and we wish to keep those as is we can do:
batch = 64
x = torch.rand((batch, 2, 5))
z = x.view(
    batch, -1
)  # And z.shape would be 64x10, this is very useful stuff and is used all the time

# Let's say we want to switch x axis so that instead of 64x2x5 we have 64x5x2
# I.e we want dimension 0 to stay, dimension 1 to become dimension 2, dimension 2 to become dimension 1
# Basically you tell permute where you want the new dimensions to be, torch.transpose is a special case
# of permute (why?)
z = x.permute(0, 2, 1)

# Splits x last dimension into chunks of 2 (since 5 is not integer div by 2) the last dimension
# will be smaller, so it will split it into two tensors: 64x2x3 and 64x2x2
z = torch.chunk(x, chunks=2, dim=1)
print(z[0].shape)
print(z[1].shape)

# Let's say we want to add an additional dimension
x = torch.arange(
    10
)  # Shape is [10], let's say we want to add an additional so we have 1x10
print(x.unsqueeze(0).shape)  # 1x10
print(x.unsqueeze(1).shape)  # 10x1

# Let's say we have x which is 1x1x10 and we want to remove a dim so we have 1x10
x = torch.arange(10).unsqueeze(0).unsqueeze(1)

# Perhaps unsurprisingly
z = x.squeeze(1)  # can also do .squeeze(0) both returns 1x10