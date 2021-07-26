
# Keras implementation of Normalizer-Free Networks and SGD - Adaptive Gradient Clipping

Paper: https://arxiv.org/abs/2102.06171.pdf

Original code: https://github.com/deepmind/deepmind-research/tree/master/nfnets

Do star this repository if it helps your work!

# Installation

pip3 install git+https://github.com/ankan2709/NFnet


Use any of the `NFNetF` models like any other keras Model!

```python

from nfnets_keras import NFNetF3

model = NFNetF3(include_top = True, num_classes = 10)
model.compile('adam', 'categorical_crossentropy')
model.fit(X, y)

```



## WSConv2D
Use `WSConv2D` like any other `keras.layer`.

```python

from nfnets_keras import WSConv2D

conv = Conv2D(16, 3)(l)
w_conv = WSConv2D(16, 3)(conv)
```


## SGD_AGC - Adaptive Gradient Clipping

Similarly, use `SGD_AGC` like `keras.optimizer.SGD`
```python

from nfnets_keras import SGD_AGC


model.compile( SGD_AGC(lr=1e-3), loss='categorical_crossentropy' )
```


# Credit for the original pytroch implementation 
```
https://github.com/vballoli/nfnets-pytorch
```

# Cite Original Work

To cite the original paper, use:
```
@article{brock2021high,
  author={Andrew Brock and Soham De and Samuel L. Smith and Karen Simonyan},
  title={High-Performance Large-Scale Image Recognition Without Normalization},
  journal={arXiv preprint arXiv:},
  year={2021}
}
```
