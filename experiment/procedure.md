### Procedure

**Objective**

The objective of this part of experiment is to explore structure and training of MLPs on tabular data by training an MLP on the **Iris dataset** (4 features, 3 classes) and to visualize forward or backprop flow and hidden-layer activations on select samples.

The Iris Dataset has been used which has **4 input features** (Sepal Length, Sepal Width, Petal Length and Petal Width) and this dataset has been distributed among three classes: `['Iris-setosa' 'Iris-versicolor' 'Iris-virginica']`.

This part focuses on building an MLP model which has one input, two hidden and one resultant output layer and then visualizing the forward and backward both propagation on this MLP network and by using different types of optimizers such as **RMSprop**, **SGD** and **adam** to find which will give better accuracy result.

---

**Step 1: Import Required Libraries**

Import `numpy` and `pandas` for numerical computation and data handling, `matplotlib` for data visualization, `sklearn` for pipelines, and model calls and evaluations.

---

**Step 2: Dataset Loading and Description**

**2.1 Load the Iris Dataset**
Load the Iris Dataset from Kaggle by importing its file path as input.

**2.2 Dataset Description**
The dataset consists of:

* **Shape:**
**Shape (150, 5)** i.e., 150 instances and 5 columns:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width
- Species

* **Input Features:**
Four input features:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

* **Target Classes:**
Three target classes with equal distribution:

**Class distribution:**
- Iris-setosa: **50**
- Iris-versicolor: **50**
- Iris-virginica: **50**

**2.3 Data Quality**
This dataset is small and reliable that it has **no missing values or duplicated rows**, so this dataset does not require any data cleaning or preprocessing and can be directly moved to the next step of scaling and label features.

**2.4 Feature Engineering**
- **Scaling the features** and then labelling them using **one-hot encoding** technique.
- **Dividing the dataset** among training and test splits: **80% for training** and **20% for testing**.

---

**Step 3: Initializing Parameters & Model Building**

**3.1 Initialize Parameters**
Firstly, initialize the parameters with those values which provide the best possible maximum accuracy:

| Parameter | Value |
|-----------|-------|
| **epochs** | 100 |
| **batch_size** | 8 |
| **learning_rate** | 0.01 |
| **optimizer** | RMSprop |

**3.2 Model Architecture**
Model architecture is built of:
- **1 Input layer**
- **2 Hidden layers (dense)**
- **1 Output layer**

Plot the `model_summary` to visualize the architecture.

**3.3 Visualize Model**
Visualizing the MLP model architecture by plotting it layer by layer and showing all layers arranged in a proper sequence.

---

**Step 4: Model Training**

**4.1 Train the Model**
Train the Iris dataset. The call runs **forward and backward passes** for 100 epochs using mini-batches of 8, holding out 20% of the training data for validation and updating weights via RMSprop on each step.

**4.2 Visualize Training Progress**
Now plotting the **training and validation curves** for loss and accuracy versus epoch to visualize learning and generalization after each fixed interval of epoch of both forward and backprop pass.

**4.3 Forward and Backward Flow**
The model attempts to show both the **forward flow of the MLP** and **backward flow of MLP**, so training the model one by one for both the cases.

---

**Step 5: Model Evaluation**

**5.1 Evaluate Performance**
Evaluate the trained model on the test dataset using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score** metrics

Visualize using **Confusion Matrix**.

**5.2 Analyze Results**
- Analyze misclassifications and class-wise performance.
- Display **Classification report table** (precision, recall, f1-score, support) of each class category as well as showing macro avg and weighted avg.

---

**Step 6: Gradient Analysis**

**Gradient norms recorded at checkpoint (backprop flow for chosen sample)**

Saved gradient L2 norms (per-layer i.e., both hidden layers and output layer kernel grads) for chosen sample.

---

**Summary**

This procedure guides you through the complete process of:
1. Loading and preprocessing the Iris dataset
2. Building an MLP model with optimized parameters
3. Training with visualization of both forward and backward propagation
4. Evaluating model performance with comprehensive metrics
5. Analyzing gradients during backpropagation