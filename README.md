# chainer-ADDA
Implementation of [Adversarial Discriminative Domain Adaptation](https://arxiv.org/abs/1702.05464) in [Chainer](https://github.com/chainer/chainer).

**Note** this code depends on [this](https://github.com/chainer/chainer/tree/96753116c96e26d33e4d747673114b06b86ebb6b) version of Chainer (or newer). Please check out the source from that link rather than installing via pip.

## Results
The following results are for the SVHN to MNIST domain adaptation task.
![loss](loss.png)

| Training | % accuracy (Paper) | % accuracy (This implementation) |
| -------- | ----- | ------------------- |
| source only   | 0.601 | 0.575          |
| ADDA          | 0.760 | 0.800          |

## Usage
Run `python train.py -g 0` to train everything using GPU 0. SVHN and MNIST datasets will download automatically. If a classifier pretrained on the source (SVHN) domain is not found, one will be trained first, then continue on to do ADDA.


# Resources
- https://arxiv.org/pdf/1702.05464.pdf
- https://github.com/erictzeng/adda
- https://github.com/davidtellez/adda_mnist64
