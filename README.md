# Fashion MNIST Image Classification — Zalando Dataset

A neural network model for classifying fashion articles from the [Zalando Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist). This project covers end-to-end image classification: data loading, preprocessing, model building, training, and evaluation.

---

## Dataset

Fashion-MNIST is a dataset of Zalando's article images consisting of **70,000 grayscale images** across **10 classes**.

| Split | Size |
|-------|------|
| Training | 60,000 images |
| Test | 10,000 images |

Each image is **28×28 pixels** with a single grayscale channel.

### Class Labels

| Label | Class |
|-------|-------|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

---

## Project Overview

### What this project covers

- **Data Loading** — Loading and inspecting the Fashion-MNIST dataset
- **Preprocessing** — Normalizing pixel values, reshaping inputs, and encoding labels
- **Model Architecture** — Building a neural network using Keras/TensorFlow
- **Training** — Fitting the model and monitoring loss/accuracy across epochs
- **Evaluation** — Assessing model performance on the test set with accuracy metrics and a confusion matrix

---

## Tech Stack

- Python 3.x
- TensorFlow / Keras
- NumPy
- Matplotlib
- scikit-learn

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/onikeyek/AI-Modeling-Fashion-MNIST-Dataset-Zalando-.git
cd AI-Modeling-Fashion-MNIST-Dataset-Zalando-
```

### 2. Install dependencies

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

### 3. Run the notebook

Open the Jupyter notebook and run all cells:

```bash
jupyter notebook
```

The dataset is loaded directly via Keras — no manual download required:

```python
from tensorflow.keras.datasets import fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
```

---

## Results

| Metric | Score |
|--------|-------|
| Test Accuracy | *(add your result here)* |

---

## Key Learnings

- Image data requires normalisation (scaling pixel values to [0, 1]) before feeding into a neural network
- Fashion-MNIST is a more challenging benchmark than standard MNIST due to greater intra-class variation
- Even a simple dense neural network achieves reasonable accuracy; CNNs push performance significantly higher

---

## Author

**Ariyike** — [github.com/onikeyek](https://github.com/onikeyek)
