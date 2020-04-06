## Uncertainty Estimation through HyperNetworks

This project stems from a small idea I got of finding a generative neural network which generates other networks. Turns out they're called HyperNetworks [1] and have found multiple use cases in different areas of Machine Learning. Something I thought they would be particularly good at would be _Uncertainty Estimation_, that is learning to estimate the epistemic uncertainty of a model. A first step in a bayesian approach to uncertainty estimation would be to estimate a distribution over the model parameters and inferring the posterior. Finding this distribution is hard since Neural Networks may contain thousands of parameters. Previpous approaches have used approximations such as Variational Inference or even Dropout [2] to estimate this posterior. 

However, modeling complex distributions is something Neural Networks are very good at. In the same way that GANs don't need anything more than a simple discriminator to be able to generate very realistic images, one could perhaps presume that it could generate "samples" of complex neural networks. The only problem is finding a good discriminator. The discriminator puts a measure on how close a generated sample is from the true distribution. If we define the "true distribution" as a net that solves some regression or classification task, the discriminator simple becomes how well the generated network performs on this task!

We this train a generative network by generating the weights of a main network which performs a forward pass and evaluates the loss function. By training it in this way, we find a generative network which with one forward pass can generate large ensembles of neural networks.

Ensemble methods are as most powerful when the models used are as diverse and performant as possible. To ensure diversity between our models, we employ another trick from GAN literature. We add a measure of mutual information between the generated output and noisy samples used as input. By ensuring that the mutual information is high, we see a larger diversity of the generated networks, and with it higher and more robust performance on our toy dataset. 

[Full write up available here](https://gtegner.github.io/https://gtegner.github.io/uncertainty/estimation/2020/01/06/hyper-gan.html)

### Setup
Dependent on my implementation of [MINE](www.github.com/gtegner/mine-pytorch) for mutual information estimation.

```
pip install -r requirements.txt
```


### References
[1]: <https://arxiv.org/pdf/1609.09106.pdf> HyperNetworks
[2]: <https://arxiv.org/abs/1506.02142> Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning
