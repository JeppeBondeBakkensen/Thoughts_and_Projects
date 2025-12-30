
### Convolutional layer 
A convolutional layer learns shared filters that detect useful local patterns such as edges, textures, and shapes, and it transforms the input image into a set of feature maps that indicate where those patterns occur, while preserving the spatial arrangement of the information (sometimes at a lower resolution if pooling or stride is used).

#### Stride 
Stride is how far the convolution filter moves each time it slides across the input. If the stride is 1, the filter shifts one pixel at a time, which gives a larger output feature map. If the stride is 2, it jumps two pixels at a time, which reduces the output’s height and width (downsampling) and makes the layer faster, but it also throws away some spatial detail.

##### Spatial dimension
For one spatial dimension, the output size is

$$ H_{out} = \left \lfloor \frac{H-F+2P}{S}\right\rfloor$$
where H is the input size, K the kernel size, P the padding, and S the stride. 

### Pooling layer
A pooling layer is often placed between convolutional layers to shrink the spatial resolution of the feature maps so the network uses less computation, has fewer activations to process, and is less likely to overfit. Pooling works independently on each channel, meaning it does not mix information across the depth dimension; it only reduces the width and height of each feature map. In max pooling, the layer slides a small window (most commonly $2 \times 2$) across each channel and replaces each window with a single value equal to the maximum inside that window.

![[Pasted image 20251231003307.png]]

### Fully connected layer 
A fully connected (FC) layer connects every input activation to every output and computes y=Wx+by = Wx + by=Wx+b. A convolution (CONV) layer does the same dot-product idea but only on local spatial patches, and it reuses the same weights across all locations.Because both are linear dot products, you can convert between them. An FC layer that takes an input volume $7 \times 7 \times 512$  and outputs $4096$ numbers is equivalent to a CONV layer with $4096$ filters of size $7 \times 7$ (stride 1, no padding), producing $1 \times 1 \times 4096$. This makes the network “slide” over larger images efficiently and output a grid of class scores instead of one vector.

#### How to find `in_features` for the first fully connected (FC) layer

**General rule:**  
Track the tensor shape through the network until just before the first FC layer.  
If the tensor has shape `(N, C, H, W)`, then after flattening it becomes `(N, C·H·W)`, so

`in_features = C * H * W`

The **final FC layer** usually has  
`out_features = number_of_classes` (e.g. 10 for CIFAR-10)

###### Example: CIFAR-10 architecture
`Input → (Conv → ReLU → Pool) × 2 → FC → ReLU → FC`

Assume CIFAR-10 input:
- `(N, 3, 32, 32)`

Assume conv settings:
- kernel `3×3`, stride `1`, padding `1`  → spatial size is preserved inside conv layers

###### Block 1
After `Conv2d(3→16, 3×3, s=1, p=1)`  
- `(N, 16, 32, 32)`

After `MaxPool2d(2,2)`  
- `(N, 16, 16, 16)`

###### Block 2
After `Conv2d(16→32, 3×3, s=1, p=1)`  
- `(N, 32, 16, 16)`

After `MaxPool2d(2,2)`  
- `(N, 32, 8, 8)`

###### Flatten → FC
After `torch.flatten(x, 1)`  
- `(N, 32·8·8) = (N, 2048)`

So the FC layers are:
- `fc1 = nn.Linear(2048, hidden_units)`
- `fc2 = nn.Linear(hidden_units, 10)`  (10 classes in CIFAR-10)
---
### ConvNet architecture (how the layers are usually stacked)

Most convolutional neural networks are built by repeating the same “block idea” several times:

1) **Convolutions** learn features while keeping the image as a grid (height × width)  
2) **ReLU** adds nonlinearity so the network can learn complex patterns  
3) **Pooling (optional)** reduces the spatial size so computation drops and features become more robust  
4) After enough downsampling, we often switch to **fully connected (FC)** layers (or another classifier head) to produce class scores

#### A common overall pattern

**Input → feature extraction blocks → classifier head**

In a compact template:

Input  
→ repeat M times: (repeat N times: Conv → ReLU) → optional Pool  
→ repeat K times: FC → ReLU  
→ final FC (class scores)

So you will often see architectures like:

- `Input → FC`  
  (a simple linear classifier, no conv feature extraction)
- `Input → Conv → ReLU → FC`
- `Input → (Conv → ReLU → Pool) × 2 → FC → ReLU → FC`
- `Input → (Conv → ReLU → Conv → ReLU → Pool) × 3 → (FC → ReLU) × 2 → FC`  
  (two convs before each pooling step)


### References 
Stanford notes: [link](https://cs231n.github.io/convolutional-networks/)
