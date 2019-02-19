Box Convolution Layer for ConvNets
==================================

<p align="center">
<img src="https://user-images.githubusercontent.com/9570420/52168173-d1ea7980-2737-11e9-8924-e4e8fd39d0ee.gif">
<br>
Single-box-conv network (from `test-mnist.py`) learns patterns on MNIST
</p>

# What This Is

This is a PyTorch implementation of the box convolution layer as introduced in the 2018 NeurIPS [paper](https://papers.nips.cc/paper/7859-deep-neural-networks-with-box-convolutions):

Burkov, E., & Lempitsky, V. (2018) **Deep Neural Networks with Box Convolutions**. *Advances in Neural Information Processing Systems 31*, 6214-6224.


## Using

```python3
import torch
from box_convolution import BoxConv2d

box_conv = BoxConv2d(16, 8, 240, 320)
help(BoxConv2d)
```

Also, there is a usage example in `test-mnist.py`.

Tested on Debian 9.7, Python 3.5, without CUDA

