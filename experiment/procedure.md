### Procedure

**Objective:** Explore the structure and training of MLPs on tabular data by training an MLP on the Iris dataset (4 features, 3 classes) and visualising forward and backprop flows and hidden-layer activations for selected samples.

The Iris dataset used contains four input features — Sepal Length, Sepal Width, Petal Length and Petal Width — and three classes: Iris-setosa, Iris-versicolor and Iris-virginica.

This experiment uses an MLP with one input layer, two hidden layers and one output layer, and visualises forward and backward propagation. Different optimisers (RMSprop, SGD and Adam) are compared to determine which yields the best accuracy.

**Steps**

1. **Import Libraries:** Import the following libraries: numpy for numerical computation, pandas for data handling, matplotlib for visualisation, and sklearn for data loading, preprocessing, model evaluation, and pipeline construction. The simulation implements the MLP using sklearn's MLPClassifier without relying on external deep learning frameworks.

2. **Dataset Loading, Description, and Splitting:**
   * Load the Iris dataset using `sklearn.datasets.load_iris()`. This returns a Bunch object; extract the feature matrix X with shape (150, 4) and the target vector y with shape (150,).
   * The four input features are: Sepal Length, Sepal Width, Petal Length, and Petal Width.
   * Class distribution: Iris-setosa: 50, Iris-versicolor: 50, Iris-virginica: 50 (perfectly balanced).
   * This dataset has no missing values and no duplicate rows; no data cleaning is required.
   * Scale features using StandardScaler and encode labels using one-hot encoding.
   * Perform a primary train/test split: 80% training, 20% testing. Use `stratify=y` to preserve class proportions across both splits.
   * From the 80% training data, reserve a further 20% as a validation set (i.e., 20% of the training portion) for monitoring training progress and early detection of overfitting during model training.

3. **Initialise Parameters and Build the Model:**
   * Typical hyperparameters: epochs = 100, batch_size = 8, learning_rate = 0.01, optimiser = RMSprop (compare with SGD and Adam).
   * Build the model with the following architecture:
     * Input layer: 4 neurons (one per input feature).
     * Hidden Layer 1: dense layer with ReLU activation.
     * Hidden Layer 2: dense layer with ReLU activation.
     * Output layer: 3 neurons with Softmax activation (one per class).
   * Display the model summary and plot the model architecture to verify the structure before training.

4. **Model training:**
   * The model is trained by minimising a cost function, which measures the difference between the predicted output and the true labels. Since the Iris dataset is a multi-class classification problem, the categorical cross-entropy loss is used:

$$J(\theta) = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$$

   * where:
   * $y_i$ is the true label (one-hot encoded),
   * $\hat{y}_i$ is the predicted probability for class $i$,
   * $C$ is the number of classes (here, $C = 3$),
   * $\theta$ represents all model parameters (weights and biases).
   * The gradients $\frac{\partial J(\theta)}{\partial \theta}$ are computed using backpropagation, and the optimisation algorithms (RMSprop, SGD, and Adam) update the parameters to minimise this cost function.
   * Train for 100 epochs with mini-batches of size 8.
   * Plot training and validation curves for loss and accuracy versus epochs.
   * Optionally, visualise forward and backward flows for selected samples and record gradient norms for analysis.

5. **Model evaluation:**
   * Evaluate the model on the test set using accuracy, precision, recall, and F1 score. Visualise results using a confusion matrix.
   * Show a classification report (precision, recall, F1-score, support) and compute macro and weighted averages.
   * Analyse misclassifications and class-wise performance.

6. **Gradient checkpoints (backprop flow):**
   * Record L2 norms of gradients per layer (both hidden layers and output layer) for selected samples and save checkpoints for analysis.
