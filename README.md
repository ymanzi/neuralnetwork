# Neural Network Lib

**Member**: :last_quarter_moon_with_face: [Ymanzi](https://github.com/ymanzi) :first_quarter_moon_with_face:

## Challenge
Implement a Multilayer Perceptron Library from scratch


## Structure
<img src="https://github.com/ymanzi/neuralnetwork/blob/main/srcs/structure.png" alt="struct" width=700 height=400>

## Perceptron (FeedForward)
<img src="https://github.com/ymanzi/neuralnetwork/blob/main/srcs/perceptron.png" alt="perceptron" width=500 height=200>

## Backpropagation Equations
<img src="https://github.com/ymanzi/neuralnetwork/blob/main/srcs/1.png" alt="1" width=500 height=200>
<img src="https://github.com/ymanzi/neuralnetwork/blob/main/srcs/2.png" alt="2" width=400 height=100>
<img src="https://github.com/ymanzi/neuralnetwork/blob/main/srcs/3.png" alt="3" width=400 height=100>
<img src="https://github.com/ymanzi/neuralnetwork/blob/main/srcs/formules.png" alt="Formules" width=500 height=300>

## Loss Functions Implemented
* Cross Entropy
* Means Squared Error

## Activation Functions Implemented
* Sigmoid
* Tanh
* ReLU
* SoftMax

<img src="https://github.com/ymanzi/neuralnetwork/blob/main/srcs/activationtab.png" alt="atab" width=600 height=800>

## Regularization Implemented
Regularization is a technique which makes slight modifications to the learning algorithm such that the model generalizes better. 
This in turn improves the model's performance on the unseen data as well, avoiding overfitting.

* L1/L2 Regularization : L1/L2 regularization try to reduce the possibility of overfitting by keeping the values of the weights and biases small.
* Dropout : To apply DropOut, we randomly select a subset of the units and clamp their output to zero, regardless of the input; this effectively removes those units from the model.
<img src="https://github.com/ymanzi/neuralnetwork/blob/main/srcs/dropout.png" alt="dropout" width=350 height=200>

* Dropout Connect : We apply dropout with the weights, instead of nodes
<img src="https://github.com/ymanzi/neuralnetwork/blob/main/srcs/dropoutconnect.png" alt="dropout" width=350 height=200>

## Resources
* [Activation Functions](https://medium.com/@omkar.nallagoni/activation-functions-with-derivative-and-python-code-sigmoid-vs-tanh-vs-relu-44d23915c1f4)
* [Neural Network Tutorial](http://neuralnetworksanddeeplearning.com/chap1.html)
* [Sigmoid Activation](https://towardsdatascience.com/derivative-of-the-sigmoid-function-536880cf918e)
* [Weight initialization](https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/)
* [He Initialization](https://www.machinecurve.com/index.php/2019/09/16/he-xavier-initialization-activation-functions-choose-wisely/)
* [Dropout Regularization](https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/)
* [Dropout vs Dropout Connect](https://stats.stackexchange.com/questions/201569/what-is-the-difference-between-dropout-and-drop-connect)
* [Momentum Intuitive Explanation](https://www.quora.com/What-is-an-intuitive-explanation-of-momentum-in-training-neural-networks)
