### Theory

Deep feedforward networks &mdash; often called feedforward neural networks or multilayer perceptrons (MLPs) &mdash; are fundamental models in deep learning. The goal of a feedforward network is to approximate some target function $f^*$. For example, for a classifier, $y = f^*(x)$ maps an input $x$ to a category $y$.

A layered feedforward network is one in which any path from an input node to an output node traverses the same number of layers. For example, the $n^{th}$ layer of such a network consists of all nodes that are $n$ edge traversals from an input node. A hidden layer is any layer that is neither the input nor the output layer. A network is fully connected if each node in layer $i$ is connected to all nodes in layer $i+1$. Layered feedforward networks have become popular because they often generalise well: when trained on a relatively sparse set of examples they frequently provide correct outputs on unseen test data.

When we use a feedforward neural network to accept an input $x$ and produce an output $\hat{y}$, information flows forward through the network. The inputs $x$ provide the initial information that then propagates to the hidden units at each layer and finally produces $\hat{y}$. This is called **forward propagation**.

During training, forward propagation continues until it produces a scalar cost $J(\theta)$. The **backpropagation algorithm** (Rumelhart et al., 1986), often simply called **backprop**, allows information from the cost to flow backwards through the network to compute gradients. The backpropagation algorithm that trains the feedforward neural network can often find a good set of weights (and biases) in a reasonable amount of time. Backpropagation relies on the chain rule to compute derivatives and typically uses a least-squares or cross-entropy criterion depending on the task.

**Layer-by-Layer (MLP) Transformation:**

* **Input Layer:** Receives raw input $x$.
* **First Hidden Layer:** $h_1 = g_2(W_1 \cdot x + b_1)$ Applies a linear transformation followed by a nonlinearity.
* **Second Hidden Layer (if any):** $h_2 = g_2(W_2 \cdot h_1 + b_2)$ Further transforms the representation
* **Output Layer:** $y = g_n(W_n \cdot h_{n-1} + b_n)$ Produces the final predictions.

More generally, the network can be expressed as a composition of functions:

$$f_n \circ ... \circ f_1(x)$$

**Where:**

* $W_i = A_i \cdot x + b_i$ represents the linear transformation (weights and biases) for layer $i$.
* $\rho$ is the activation function (often the same across layers).

Up to now, we've been discussing neural networks where the output from one layer is used as input to the next layer. Such networks are called **feedforward neural networks**. This means there are no loops in the network - information is always feed forward, never feedback which can also be shown as in Fig.1.

These models are called feedforward because information flows through the function being evaluated from $x$, through the intermediate computations used to define $f$, and finally to the output $y$. There are no feedback connections in which outputs of the model are fed back into itself.

1. **Gradient Based Learning:** For feedforward neural networks, it is important to initialise all weights to small random values; biases may be initialised to zero or to small positive values. Iterative gradient-based optimisation algorithms (e.g., SGD, RMSprop, Adam) are used to train feedforward networks and deepest models.

2. **Learning XOR:** To illustrate the capabilities of feedforward networks, consider the XOR function. XOR returns 1 when exactly one of $x_1$ or $x_2$ is 1, and 0 otherwise. Learning XOR demonstrates that an MLP with a hidden layer can represent non-linearly separable functions.

To make the idea of a feedforward network more concrete, we begin with an example of a fully functioning feedforward network on a very simple task: learning the XOR function. The XOR function ("exclusive or") is an operation on two binary values, $x_1$ and $x_2$. When exactly one of these binary values is equal to 1, the XOR function returns 1. Otherwise, it returns 0. The XOR function provides the target function $y = f^*(x)$ that we want to learn. Our model provides a function $y = f(x;\theta)$ and our learning algorithm will adapt the parameters $\theta$ to make $f$ as similar as possible to $f^*$.

![Architecture of Feedforward Neural Network](images/image5.png)
*Figure 1: Architecture of Feedforward Neural Network*
*(Source: M. A. Nielsen, Neural Networks, and Deep Learning.)*

The process of forward propagation from input to output and backward propagation of errors is repeated several times until the error gets below a predefined threshold. The whole process is represented in the following diagram:

![MLP Process both Forward and Backpropagation](images/image6.png)
*Figure 2: MLP Process both Forward and Backpropagation*
*(Source: Antonio Gulli, Sujit Pal, Deep Learning with Keras)*

The forward and backward propagation process is repeated until the error falls below a predefined threshold. The model is updated to progressively minimise the loss function. In a neural network, individual neuron outputs matter less than the collective behaviour of weights in each layer as shown in Fig.2; the network adjusts its internal weights, so the prediction accuracy increases. Using appropriate features and high-quality labels is fundamental for reducing bias and improving learning.

**Merits of Feedforward Neural Network (MLP):**

* **Scalability:** The number of hidden layers and neurons can be adjusted to match problem complexity.
* **Performance on tabular data:** For many structured datasets, MLPs can outperform more complex models due to their simplicity and ability to learn direct features.
* **Universal function approximation:** MLPs can approximate virtually any continuous function, enabling them to model complex, non-linear relationships.

**Demerits of Feedforward Neural Network (MLP):**

* **Sensitivity to hyperparameters:** Performance depends heavily on choices such as number of layers, units, learning rate, and activation functions.
* **Overfitting:** MLPs can memorise training data and generalise poorly, especially with small datasets.
* **Gradient issues:** Very deep networks may encounter vanishing or exploding gradients, making training difficult.
* **Data requirements:** Large amounts of labelled data are often necessary to train effectively.