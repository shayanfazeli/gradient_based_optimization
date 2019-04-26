# An overview: Gradient-based Optimization

This repository includes my codes for running and reviewing the effects of different gradient-based optimization approaches on MNIST and CIFAR datasets.

For the results, you can use tensorboard:

```
tensorboard --logdir='./runs'
```

I have used tensorboardX to bind it with PyTorch.

The accuracies and a more detailed report can also be found in this repository:

```
mnist_sgd_withTransform_experiment: 97.55
mnist_sgd_withoutTransform_experiment: 97.03
mnist_sgd_with_momentum_withTransform_experiment: 99.04
mnist_sgd_with_momentum_withoutTransform_experiment: 98.9
mnist_adadelta_withTransform_experiment: 99.35
mnist_adadelta_withoutTransform_experiment: 99.32
mnist_adagrad_withTransform_experiment: 98.84
mnist_adagrad_withoutTransform_experiment: 98.84
mnist_adam_withTransform_experiment: 99.08
mnist_adam_withoutTransform_experiment: 99.26
mnist_adamax_withTransform_experiment: 99.28
mnist_adamax_withoutTransform_experiment: 99.31
mnist_asgd_withTransform_experiment: 97.38
mnist_asgd_withoutTransform_experiment: 97.05
mnist_rmsprop_withTransform_experiment: 10.32
mnist_rmsprop_withoutTransform_experiment: 96.61
mnist_rprop_withTransform_experiment: 96.79
mnist_rprop_withoutTransform_experiment: 96.75
cifar_sgd_withTransform_experiment: 46.96
cifar_sgd_withoutTransform_experiment: 42.41
cifar_sgd_with_momentum_withTransform_experiment: 69.15
cifar_sgd_with_momentum_withoutTransform_experiment: 65.1
cifar_adadelta_withTransform_experiment: 74.9
cifar_adadelta_withoutTransform_experiment: 72.33
cifar_adagrad_withTransform_experiment: 63.85
cifar_adagrad_withoutTransform_experiment: 66.14
cifar_adam_withTransform_experiment: 72.7
cifar_adam_withoutTransform_experiment: 71.65
cifar_adamax_withTransform_experiment: 74.35
cifar_adamax_withoutTransform_experiment: 73.4
cifar_asgd_withTransform_experiment: 47.95
cifar_asgd_withoutTransform_experiment: 43.43
cifar_rmsprop_withTransform_experiment: 10.0
cifar_rmsprop_withoutTransform_experiment: 10.0
cifar_rprop_withTransform_experiment: 44.27
cifar_rprop_withoutTransform_experiment: 42.13
```
