import torch
import torch.nn as nn
import torch.nn.functional as F


"""
In this file, we will implement the CNN from scratch using pytorch.
"""
torch.manual_seed(0)

class Conv2d:
    def __init__(self, kernel_size) -> None:
        self.kernel_size = kernel_size
        self.weights = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))
        
    def forward(self, x):
        conv_in = self._convolute(x, self.weights)
        return conv_in + self.bias
    
    def _convolute(self, x, kernel):
        k_w, k_h = kernel.shape
        x_w, x_h = x.shape
        output_w = x_w - k_w + 1 # Assuming stride=1 and no padding
        output_h = x_h - k_h + 1
        output = torch.zeros(output_w, output_h)
        for i in range(output_w):
            for j in range(output_h):
                output[i, j] = torch.sum(x[i:i+k_w, j:j+k_h] * kernel)
        return output

x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
conv = Conv2d((2, 2))
print(conv.forward(x))

# Using Pytorch Conv2d to see if it converges to the output

x = torch.ones(1, 1, 3, 3) # Batch size, channels, height, width
y = torch.zeros(1, 1, 2, 2) 
conv = nn.Conv2d(in_channels=1,out_channels=1, kernel_size=(2, 2), stride=1, padding=0, bias=False)
print("initial output",conv.forward(x))
lr = 0.01
for i in range(50):
    out = conv.forward(x)
    loss = F.mse_loss(out, y)
    loss.backward()
    with torch.no_grad():
        conv.weight -= lr * conv.weight.grad
        conv.zero_grad()
    if i % 5 == 0:
        print(f"Loss at iteration {i} is {loss}")

print(conv.forward(x))

## Console output
# tensor([[-6.0034, -6.3662],
#         [-7.0918, -7.4545]], grad_fn=<AddBackward0>)
# initial output tensor([[[[-0.9964, -0.9964],
#           [-0.9964, -0.9964]]]], grad_fn=<ConvolutionBackward0>)
# Loss at iteration 0 is 0.99282306432724
# Loss at iteration 5 is 0.431270956993103
# Loss at iteration 10 is 0.1873391568660736
# Loss at iteration 15 is 0.08137797564268112
# Loss at iteration 20 is 0.03534965589642525
# Loss at iteration 25 is 0.015355474315583706
# Loss at iteration 30 is 0.006670239847153425
# Loss at iteration 35 is 0.002897476078942418
# Loss at iteration 40 is 0.0012586290249601007
# Loss at iteration 45 is 0.0005467343144118786
# tensor([[[[-0.0154, -0.0154],
#           [-0.0154, -0.0154]]]], grad_fn=<ConvolutionBackward0>)
