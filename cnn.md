## Overview

Architecture of a traditional CNN Convolutional neural networks, also known as CNNs, are a specific type of neural networks that are generally composed of the following layers:

![](/assets/images/architecture-cnn-en.jpeg)

The convolution layer and the pooling layer can be fine-tuned with respect to hyperparameters that are described in the next sections.

  

## Types of layer

**Convolution layer (CONV):**  The convolution layer (CONV) uses filters that perform convolution operations as it is scanning the input $I$ with respect to its dimensions. Its hyperparameters include the filter size $F$ and stride $S$. The resulting output $O$ is called _feature map_ or _activation map_.

![](/assets/images/convolution-layer-a.png)

  

**Pooling (POOL):** The pooling layer (POOL) is a downsampling operation, typically applied after a convolution layer, which does some spatial invariance. In particular, max and average pooling are special kinds of pooling where the maximum and average value is taken, respectively.

| Type         | Max pooling                                                               | Average pooling                                           |
|--------------|---------------------------------------------------------------------------|-----------------------------------------------------------|
| Purpose      | Each pooling operation selects the maximum value of the current view     | Each pooling operation averages the values of the current view |
| Illustration |   ![](/assets/images/max-pooling-a.png)   | ![](/assets/images/average-pooling-a.png)                                                                        |                                                           |
| Comments     | • Preserves detected features<br>• Most commonly used                    | • Downsamples feature map<br>• Used in LeNet              |


**Fully Connected (FC):** The fully connected layer (FC) operates on a flattened input where each input is connected to all neurons. If present, FC layers are usually found towards the end of CNN architectures and can be used to optimize objectives such as class scores.

![](/assets/images/fully-connected-ltr.png)

  

## Filter hyperparameters

The convolution layer contains filters for which it is important to know the meaning behind its hyperparameters.

**Dimensions of a filter:** A filter of size $F\times F$ applied to an input containing $C$ channels is a $F \times F \times C$ volume that performs convolutions on an input of size $I \times I \times C$ and produces an output feature map (also called activation map) of size $O \times O \times 1$.

![](/assets/images/dimensions-filter-en.png)

  
_Remark: the application of $K$ filters of size $F\times F$ results in an output feature map of size $O \times O \times K$._

**Stride:** For a convolutional or a pooling operation, the stride $S$ denotes the number of pixels by which the window moves after each operation.

![](/assets/images/stride.png)

**Zero-padding:** Zero-padding denotes the process of adding $P$ zeroes to each side of the boundaries of the input. This value can either be manually specified or automatically set through one of the three modes detailed below:

| Mode        | Valid                                                                                   | Same                                                                                                                                                               | Full                                                                                                                                                                |
|-------------|------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Value     |  P=0  | $$P_{\text{start}} = \left\lfloor \frac{S \left\lceil \frac{S}{I} \right\rceil - I + F - S}{2} \right\rfloor$$<br>$$P_{\text{end}} = \left\lceil \frac{S \left\lceil \frac{S}{I} \right\rceil - I + F - S}{2} \right\rceil$$                | $$P_{\text{start}} ∈ [[0, F - 1]]$$<br> $$P_{\text{end}} = F - 1$$                                                                                                                        |                                                                                                                                                                     |
| Illustration| Padding valid                                                                            | Padding same                                                                                                                                                       | Padding full                                                                                                                                                        |
| Purpose     | • No padding<br>• Drops last convolution if dimensions do not match                        | • Padding such that feature map size has size $⌈I/S⌉$<br>• Output size is mathematically convenient<br>• Also called 'half' padding                              | • Maximum padding such that end convolutions are applied on the limits of the input<br>• Filter 'sees' the input end-to-end                                         |

  

## Tuning hyperparameters

**Parameter compatibility in convolution layer:** By noting $I$ the length of the input volume size, $F$ the length of the filter, $P$ the amount of zero padding, $S$ the stride, then the output size $O$ of the feature map along that dimension is given by:

$$\boxed{O=\frac{I-F+P_\text{start} + P_\text{end}}{S}+1}$$

![](/assets/images/parameter-compatibility-en.jpeg)

  

**Understanding the complexity of the model:** In order to assess the complexity of a model, it is often useful to determine the number of parameters that its architecture will have. In a given layer of a convolutional neural network, it is done as follows:

|                | CONV                                                 | POOL                               | FC                                         |
|----------------|------------------------------------------------------|------------------------------------|--------------------------------------------|
| Illustration   |    ![](/assets/images/table-conv.png) | ![](/assets/images/table-pool.png) | ![](/assets/images/table-fc.png)                                                  |                                    |                                            |
| Input size     | $I \times I \times C$                                | $I \times I \times C$              | $N_{in}$                                   |
| Output size    | $O \times O \times K$                                | $O \times O \times C$              | $N_{out}$                                  |
| Number of parameters | $(F \times F \times C + 1) \cdot K$             | $0$                                | $(N_{in} + 1) \times N_{out}$              |
| Remarks        | • One bias parameter per filter<br>• In most cases, $S < F$<br>• A common choice for $K$ is $2^C$ | • Pooling operation done channel-wise<br>• In most cases, $S=F$ | • Input is flattened<br>• One bias parameter per neuron<br>• The number of FC neurons is free of structural constraints |


**Receptive field:** The receptive field at layer $k$ is the area denoted $R_k \times R_k$ of the input that each pixel of the $k$\-th activation map can 'see'. By calling $F_j$ the filter size of layer $j$ and $S_i$ the stride value of layer $i$ and with the convention $S_0 = 1$, the receptive field at layer $k$ can be computed with the formula:

$$\boxed{R_k = 1 + \sum_{j=1}^{k} (F_j - 1) \prod_{i=0}^{j-1} S_i}$$

_In the example below, we have $F_1 = F_2 = 3$ and $S_1 = S_2 = 1$, which gives $R_2 = 1 + 2\cdot 1 + 2\cdot 1 = 5$._

![](/assets/images/receptive-field-a.png)

  

## Commonly used activation functions

**Rectified Linear Unit:** The rectified linear unit layer (ReLU) is an activation function $g$ that is used on all elements of the volume. It aims at introducing non-linearities to the network. Its variants are summarized in the table below:

| Activation | Formula | Remarks |
|------------|---------|---------|
| ReLU | $g(z) = \max(0, z)$ | • Non-linearity complexities biologically interpretable |
| Leaky ReLU | $g(z) = \max(\epsilon z, z)$ with $\epsilon \ll 1$ | • Addresses dying ReLU issue for negative values |
| ELU | $g(z) = \max(\alpha(e^z - 1), z)$ with $\alpha \ll 1$ | • Differentiable everywhere |

**Softmax:** The softmax step can be seen as a generalized logistic function that takes as input a vector of scores $x\in\mathbb{R}^n$ and outputs a vector of output probability $p\in\mathbb{R}^n$ through a softmax function at the end of the architecture. It is defined as follows:


$$\boxed{p=\begin{pmatrix}p_1\\\vdots\\p_n\end{pmatrix}}\quad\textrm{where}\quad\boxed{p_i=\frac{e^{x_i}}{\displaystyle\sum_{j=1}^ne^{x_j}}}$$

  

