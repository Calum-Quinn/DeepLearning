# Deep Learning Project

## Introduction

The goal of this repository is to store all the various files needed for completing [this course](https://openclassrooms.com/en/courses/6532316-introduction-to-deep-learning-models) on deep learning.

I want to use this project as a way of getting to know a new theme in my computer science journey all while studying at the HEIG-vd for my bachelor's degree.

[Here](https://github.com/Disruptive-Engineering-Limited/introduction-to-deep-learning) is a link to the repository I get to look at in case of difficulty completing a step.

## Steps

### 1. Single Neuron

In this exercise the goal is to create our first neural network consisting of just one neuron.
This does not really have any real world applications but at least it gets us started.

![Single-neuron model](01_SingleNeuron/Model.png)

The reason behind this exercise is differentiating between two variables, in this case olives and corn.
We are given information on shape and colour and from this the neuron has to decide whether it is an olive or corn.


### 2. Multiple Neurons

This time we are asked to train multiple neurons to have an actual network.

We can't use one single neuron because we have more than two possible scenarios.
Therefore we have to create a multi layer model.

![Multi-layer model](02_MultipleNeurons/Model.png)

This time the reason for this network is to decide whether there has been an error with the sauce on a pizza.
There are two available sauces but a pizza should have 1 and only 1 on it.
Therefore if the pizza has either no sauce or both, there is a problem.


### 3. Multiple Output Neurons

In this part the goal is to create a network with several output neurons.
We need to make sure the outputs are mutually exclusive as we want the possible categories to be unique for each measurement.
There is no need for multiple layers in this model, simply having an output layer with n neurons for n categories (in this case n = 3).
Each neuron will have all the inputs for the network (in this case 15).

![Multi-output model](03_MultipleOutputNeurons/Model.png)

The use case for this model is presented as a detection system for deciding what colour box to put each pizza in.
It checks the ingredients on the pizza and the neural network has to decide whether that makes the vegetarian, vegan or meaty.


### 4. Deep Fully Connected Network

This exercise guides us in the creation of a deep fully connected network.
We now have 3 layers to our network, input, hidden and output which allow for more complex computing.
This time we use a function called dropout which basically does not activate some random neurons for an epoch.
This avoids having dependencies on preceding parts of the network.

![Dropout](04_DeepFullyConnected/Dropout.png)

We also employed rectified lineau units (ReLU) so as to avoid gradient vanishing during backpropagation of the error through multiple layers.

Lastly we used batches to decide after how many predictions the weights should be updated.

The reason for this network is to predict whether their will be traffic depending on the day and the time.


### 5. Convolutional Neural Networks

This section of the course is not an exercise but more a class on convolutional neural networks.
They are very successful in computer vision applications.

It explains that convolutional neural networks analyses data part by part instead of all together as this is easier to process.
Filters are used to scan individual parts of the data for something specific the filter has been trained to find.
After going over the data, the filter then produces an output image of it's own to represent the data according to it's training.

![Filter](05_ConvolutionalNetworks/Filter.png)

Once the various filters have been over the data you can use pooling layers to compress the information into fewer pixels.
For this there are two main approaches, max pooling and average pooling.

![Pooling](05_ConvolutionalNetworks/Pooling.png)


### 6. First Convolutional Neural Network

This next exercise is the first one with a real application even for me.
The goal is to create a convolutional neural network to recognise shapes.

For this we have a large amount of examples stored [here](06_ConvolutionalNetwork/shapes) that we can use to train the model.
The final network will look something like this:

![Model](06_ConvolutionalNetwork/Model.png)

For each image in the training set we start by creating a convolutional layer with 16 filters.
Then we compress that layer by creating a max pooling layer.
After that we flatten the compressed layer into a one dimensional vector.
Finally we create a dense/output layer to give us the categorisation of the shape.

I decided to add a last section which shows us which images were misclassified to see if there are any specific anomalies.


### 7. Recurrent Neural Network

For this part of the course, the goal is to create and train a recurrent neural network.
This will use many pizza recipes to detect the way they are written.

What makes the model recurrent is the fact that after a defined amount of epochs, the output is reinserted into network.
This means the output is not simply predicted to do with the current inputs but also to do with the previous predictions and inputs.

This is often used when predicting text as you need to know the previous letters in a word/sentence to be able to accurately guess the next one.

![Model](07_RecurrentNetworks/Model.png)

In this exercise we start by converting the input into a machine interpretable format, in this case this means mapping the various characters to numbers.

We then split the text into chunks of characters that will be used to predict the next character in the recipe.
With these chunks we designate input and output chunks so that we can tell the model what we want to achieve.

Between each epoch of training, we will be saving the model because we want to change the embedding layer's setting to only use one chunk when using the model.

### 8. Extra

The course talks about a few other architectures for neural networks and recommends the following reading.
- [Autoencoders](https://blog.keras.io/building-autoencoders-in-keras.html)
- [Attention](https://www.tensorflow.org/tutorials/text/nmt_with_attention)
- [Deep Reinforcement Learning](https://keras.io/examples/rl/actor_critic_cartpole/)
- [Generative Adversarial Networks](https://www.tensorflow.org/tutorials/generative/dcgan)
- [Transfer Learning](https://keras.io/guides/transfer_learning/)
- [Regression](https://www.tensorflow.org/tutorials/structured_data/time_series)
