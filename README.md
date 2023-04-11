# Neural network for data represented by fuzzy sets
## Introduction
Example of simple machine learning of neural networks for data represented by blurred sets in Python using the Keras library together with the model utilization function.
## Requirements
For proper functioning of algorithms you need the ```Python``` interpreter version ```3.11.3``` and the following libraries:
- keras (2.12.0)
- scikit-leranr (1.2.2)
- numpy (1.23.5)
- matplotlib (3.7.1)
Algorithm has been tested for the above versions.

# MAIN (file)
## Description
The model created in the above code is a neural network with a multi-layer feedforward architecture (dense neural network), which consists of two dense layers. The input layer has 2 neurons and the output layer has 1 neuron. The output from the network is processed by the sigmoid activation function. Network optimization is performed by the Adam algorithm, and the loss function is binary crossentropy.

In this example, random inputs were used, which were labeled based on the distance from the center of the circle with a radius of ```0.5```. The ReLU function was used as the activation function for the hidden layer and the sigmoid function was used as the activation function for the last layer. Binary cross entropy was used as a cost function and the Adam optimization algorithm. The model has been saved to the file ```model.h5```.

Variables x and y denote the training input for the model. Variable x is a matrix of size ```(1000, 2)``` containing random floating-point numbers in the range [0, 1]. The variable y is a vector of length 1000 that assigns to each point z x a label of class: if the point is inside a circle with radius ```0.5``` and centered at point (0, 0), it is assigned label ```1```, otherwise label ```0```.

## Example execution
Creation of random data
```
[[0.61662318 0.16897111]
 [0.27433151 0.35002714]
 [0.08101186 0.93376906]
 ...
 [0.9598124  0.14696508]
 [0.70578036 0.43673846]
 [0.67153367 0.34263268]]
```
Assign classes to data
```
[1 1 0 ... 0 0 0]
```
Data display

![1](https://user-images.githubusercontent.com/101213292/231111576-06fa85d1-4b54-4d96-be40-5e45676c3475.png)

Learning the Model
```
Epoch 1/50
25/25 [==============================] - 1s 9ms/step - loss: 0.7551 - accuracy: 0.3537 - val_loss: 0.7301 - val_accuracy: 0.4100
...
Epoch 50/50
25/25 [==============================] - 0s 2ms/step - loss: 0.2794 - accuracy: 0.9175 - val_loss: 0.2931 - val_accuracy: 0.9100
```
Display summary for trained model and test
```
Train loss: 0.2769, Train accuracy: 0.9150
Test loss: 0.2931, Test accuracy: 0.9100
```
Display summary for trained model and test on graphs

![2](https://user-images.githubusercontent.com/101213292/231113097-a601ba17-7633-4fb5-ad6c-690b071b34f6.png)


# TEST (file)
## Description
Example of using the model.
We have loaded the previously saved model from the file ```model.h5```. Then we created new test data consisting of ```10``` random samples. Finally, we predicted the values for this data and displayed them in the console. The result is a number in the range [0, 1] denoting the percentage of the point belonging to the class.

## Example execution
Preparation of new test data
```
[[0.15342268 0.04660311]
 [0.4396789  0.70575613]
 [0.10246322 0.00663288]
 [0.88248692 0.81838172]
 [0.14041443 0.17628658]
 [0.09186618 0.95488143]
 [0.1575     0.01667072]
 [0.96690126 0.18696219]
 [0.79942239 0.89741585]
 [0.32436679 0.82508591]]
```

Prediction of values for new data
```
1/1 [==============================] - 0s 116ms/step
```

Displaying results for the selected data
```
[[0.895558  ]
 [0.17771617]
 [0.905767  ]
 [0.02040673]
 [0.8838476 ]
 [0.23025686]
 [0.89796984]
 [0.20714243]
 [0.019393  ]
 [0.16774262]]
```
