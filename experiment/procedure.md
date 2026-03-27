### Procedure

The objective of this experiment is to understand the architecture and training process of a Multilayer Perceptron (MLP) for tabular data by implementing it on the Iris dataset, with detailed visualisation of forward propagation, backpropagation, and hidden-layer activations for selected samples.

1.  **Importing Important Libraries**
    *   Import `numpy` and `pandas` for numerical computation and data handling, `matplotlib` and `seaborn` for data visualisation, and `sklearn` for preprocessing pipelines and evaluation utilities.
    *   Import `tensorflow` and `tensorflow.keras` for model implementation, and `json`, `os`, and `time` as helper utilities.

2.  **Dataset Creation**
    *   Load the Iris dataset from a CSV file using `pandas`.
    *   The dataset shape is (150, 5): four feature columns (SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm) and one label column (Species).
    *   Class distribution: Iris-setosa: 50, Iris-versicolor: 50, Iris-virginica: 50 (balanced).
    *   Scale features using `StandardScaler` and encode labels using `LabelBinarizer` (one-hot encoding).
    *   Split into train/test sets with 80% training and 20% testing; use `stratify` to preserve class proportions.

3.  **Initializing Parameters & Model Building**
    *   Set hyperparameters: `EPOCHS = 100`, `BATCH_SIZE = 8`, `LEARNING_RATE = 0.01`, `OPTIMIZER = "RMSprop"`.
    *   Build the model with an Input layer of 4 neurons, two hidden Dense layers (10 neurons and 8 neurons, both with ReLU activation), and an output Dense layer of 3 neurons with softmax activation.
    *   The model is trained to minimise the categorical cross-entropy loss $J(\theta) = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$, where $C$ is the number of classes, $y_i$ is the true one-hot encoded label, and $\hat{y}_i$ is the predicted probability for class $i$.
    *   Display the model summary.

4.  **Model Training**
    *   Train for 100 epochs using mini-batches of size 8 with a custom `tf.GradientTape` training loop.
    *   At each epoch, perform forward propagation, compute the loss using categorical cross-entropy, and update model parameters via backpropagation.
    *   Record training loss, training accuracy, validation loss, and validation accuracy at each epoch.
    *   Every 10 epochs, save model weight checkpoints and record the L2 norms of kernel gradients for Hidden\_Layer\_1, Hidden\_Layer\_2, and Output\_Layer for a selected inspection sample.

5.  **Model Evaluation**
    *   Load saved checkpoint weights and evaluate the model at each checkpoint epoch (10, 20, …, 100).
    *   For each checkpoint, report the test accuracy, display the confusion matrix, and print the classification report (precision, recall, F1-score, support) with macro and weighted averages.
    *   Plot training and validation loss and accuracy curves up to each checkpoint epoch, and display a bar chart of gradient kernel L2 norms per layer for the chosen inspection sample.
