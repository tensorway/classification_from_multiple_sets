#!/bin/bash

# 1
python3 train.py -d mnist -b 4096 -e 100 -fin model_checkpoints/mnist.pt
python3 train.py -d cifar -b 4096 -e 100 -fin model_checkpoints/cifar.pt

# 2
python3 train.py -d svhn  -b 4096 -e 100 -fin model_checkpoints/svhn.pt
python3 test_one_on_another.py -d mnist -ptr model_checkpoints/svhn.pt -img imgs/svhn_model_on_mnist.png
python3 test_one_on_another.py -d svhn -ptr model_checkpoints/mnist.pt -img imgs/mnist_model_on_svhn.png

# 3
python3 train.py -d svhn mnist -b 4096 -e 100 -fin model_checkpoints/svhn_and_mnist.pt
