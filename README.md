# Multiclass Classification with TensorFlow

## 📌 Overview

This repository focuses on solving **Multiclass Classification** problems using custom-built multilayer neural networks. Utilizing **TensorFlow** and `tf.keras`, this project explores how to structure, compile, and train deep learning models to categorize data into more than two distinct classes.

The repository features implementations on two classic, foundational datasets: the **Wine dataset** and the **Iris dataset**.

## ✨ Projects Included

### 1. Wine Quality Classification (`Wine_data.ipynb`)

This notebook builds a custom feed-forward neural network to classify different types of wine based on their chemical properties (e.g., alcohol content, malic acid, ash).

* **Custom Architecture:** Implements a tailored multilayer perceptron (MLP) with carefully chosen hidden layers and node counts to capture the nonlinear relationships in the chemical data.
* **Feature Scaling:** Demonstrates the importance of normalizing input features so the neural network converges efficiently during gradient descent.

### 2. Iris Species Identification (`multi_class_classification_iris.ipynb`)

This notebook tackles the famous Iris dataset, classifying iris flowers into three distinct species (Setosa, Versicolor, Virginica) based on their sepal and petal measurements.

* **End-to-End Pipeline:** Covers data loading, preprocessing, model definition, training, and evaluation using TensorFlow.
* **Probability Distribution:** Highlights how the model outputs a probability distribution across the three species to make its final prediction.

## 🧮 The Mathematics (Softmax & Cross-Entropy)

To handle multiple exclusive classes, the final layer of both custom neural networks uses the **Softmax activation function**. It converts the raw output logits $\mathbf{z}$ into a normalized probability distribution where all probabilities sum to 1:

$$\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

Where $K$ is the total number of classes.

To measure how well the model's predicted probabilities $\hat{y}$ match the actual one-hot encoded labels $y$, we use **Categorical Cross-Entropy** as our loss function, which the network minimizes during training:

$$L = - \sum_{i=1}^{K} y_i \log(\hat{y}_i)$$

## 🚀 Getting Started

### Prerequisites

To run these notebooks, you will need a standard Python data science environment. For optimal performance with TensorFlow, managing your environment via Conda within a WSL2 setup is highly recommended.

* Python 3.8+
* TensorFlow 2.x
* NumPy
* Pandas
* Scikit-Learn (for dataset loading and splitting)
* Jupyter Notebook or JupyterLab

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Srihari0804/Multiclass_classification.git
cd Multiclass_classification

```


2. Activate your development environment and install the required dependencies:
```bash
conda activate your-tf-env
pip install tensorflow numpy pandas scikit-learn jupyter

```



## 💻 Usage

To explore the custom models and their training loops, launch Jupyter Notebook:

```bash
jupyter notebook

```

From the Jupyter interface, open either `Wine_data.ipynb` or `multi_class_classification_iris.ipynb` and run the cells sequentially to observe the model building and training processes.

## 📂 Project Structure

* `Wine_data.ipynb` - Multiclass classification model for the Wine dataset.
* `multi_class_classification_iris.ipynb` - Multiclass classification model for the Iris dataset.
