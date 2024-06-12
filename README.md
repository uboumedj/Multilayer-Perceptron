# 42 Deep learning project: Multilayer Perceptron

Multilayer perceptron is an advanced project in 42's AI branch. It introduces us to neural networks, more precisely the aptly named **multilayer perceptron**.

We are given a dataset representing breast tumor data. The data has 30 parameters, and are either classified as Benign (B) or Malignant (M). The goal is to code, from scratch,
the whole principle of a multilayer perceptron, writing a code as modular as possible (for reusability puroposes in other projects). We must then train a model
to be able to determine whether a tumor is B or M.

## Assignment

The assignment has multiple parts.

First I created classes representing a neural network layer and a model, writing down the whole logic behind concepts like **feed-forwarding**, **back-propagation**, **weight initialisation** and different **activation functions**.

Then, I defined a structure after a few experiments, and trained a neural network with mini batches for **cross validation**.

Finally, I wrote another program to make **predictions** using the trained model on a previously-unseen **test set**, and to evaluate that prediction.

As a bonus, a third program compares the different saved model's **training metrics**.

## Libraries used

* The whole project was coded using python 3.11
* **matplotlib** (*data visualisation, training comparison*)
* **pandas**, **numpy** (*data structure*)
* **click** (*program argument parser*)
* **pickle** (*model saving*)
