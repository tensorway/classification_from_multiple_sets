# classification_from_multiple_sets

![img](imgs/acc_train.svg)

![img](imgs/acc_valid.svg)


# Domain shift
After training the model on SVNH data it is evaluated on MNIST. As we can see the model is fairly robust and can classify the images fairy well with () accuracy and () loss. This is probably due to the fact that something like an MNIST digit can be found in street numbers. Some digits work best such as 1, but others such as 6 are not that good.
![SVNG model on MNIST](imgs/svhn_model_on_mnist.png)

After training the model on MNIST data it is evaluated on SVNG. As we can see the model does not handle the domain shift well with () accuracy and () loss. This is probably due to the fact that there is no color in MNIST dataset. Some digits work best such as 1, but others such as 6 are not that good.
![SVNG model on MNIST](imgs/mnist_model_on_svnh.png)

