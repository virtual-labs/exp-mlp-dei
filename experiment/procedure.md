### Procedure

**Objective:**
Explore the structure and training of MLPs on tabular data by training an MLP on the Iris dataset (4 features, 3 classes) and visualising forward and backprop flows and hidden-layer activations for selected samples.

The Iris dataset used contains four input features &mdash; Sepal Length, Sepal Width, Petal Length and Petal Width &mdash; and three classes: Iris-setosa, Iris-versicolor and Iris-virginica.

This experiment uses an MLP with one input layer, two hidden layers and one output layer, and visualises forward and backward propagation. Different optimisers (RMSprop, SGD and Adam) are compared to determine which yields the best accuracy.

**Steps**

1. **Import libraries:**
   Import `numpy` and `pandas` for numerical computation and data handling, `matplotlib` for data visualization, `sklearn` for pipelines, and model calls and evaluations. Use a deep learning framework (e.g., TensorFlow/Keras or PyTorch) for model implementation.

2. **Dataset loading and description:**
   * Load the Iris dataset (e.g., from Kaggle or sklearn.datasets).
   * The dataset shape is (150, 5): four feature columns and one label column.
   * Class distribution: Iris-setosa: 50, Iris-versicolor: 50, Iris-virginica: 50 (balanced).
   * This dataset has no missing values or duplicate rows; minimal cleaning is required.
   * Scale features (e.g., StandardScaler) and encode labels (one-hot encoding).
   * Split into train/test sets with 80% training and 20% testing; use stratify to preserve class proportions.

3. **Initialise parameters and model building:**
   * Typical hyperparameters: epochs=100, batch_size=8, learning_rate=0.01, optimiser=RMSprop (compare with SGD and Adam).
   * Build the model with one input layer, two hidden dense layers, and one output layer (softmax). Display the model summary and plot the model architecture.

4. **Model training:**
   * Train for 100 epochs with mini-batches of size 8. Reserve 20% of the training data for validation.
   * Plot training and validation curves for loss and accuracy versus epochs.
   * Optionally, visualise forward and backward flows for selected samples and record gradient norms for analysis.

5. **Model evaluation:**
   * Evaluate the model on the test set using accuracy, precision, recall, and F1 score. Visualise results using a confusion matrix.
   * Show a classification report (precision, recall, F1-score, support) and compute macro and weighted averages.
   * Analyse misclassifications and class-wise performance.

6. **Gradient checkpoints (backprop flow):**
   * Record L2 norms of gradients per layer (both hidden layers and output layer) for selected samples and save checkpoints for analysis.