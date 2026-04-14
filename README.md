# Fashion MNIST — Neural Network Image Classification

A fully connected neural network (Dense NN) for classifying fashion articles from the [Zalando Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist), built with TensorFlow/Keras. The model classifies 28×28 grayscale images across 10 clothing categories with **88.54% test accuracy**.

> Built by Group 6: Jo-ann Obewe · Omole Peter · Naimot Yekini · Khaled Ahmed

---

## Dataset

Fashion-MNIST consists of **70,000 grayscale images** across 10 clothing categories — designed as a more challenging drop-in replacement for the classic MNIST benchmark.

| Split | Size |
|-------|------|
| Training (used) | 49,500 |
| Validation | 10,500 |
| Test | 10,000 |
| **Total** | **70,000** |

Each image is **28×28 pixels**, single grayscale channel, with 6,000 images per class.

### Class Labels

| Label | Class | Label | Class |
|-------|-------|-------|-------|
| 0 | T-shirt/top | 5 | Sandal |
| 1 | Trouser | 6 | Shirt |
| 2 | Pullover | 7 | Sneaker |
| 3 | Dress | 8 | Bag |
| 4 | Coat | 9 | Ankle boot |

---

## Pipeline

```
Data → Preprocessing → Splitting → Model Training → Hyperparameter Tuning → Evaluation
```

---

## Preprocessing

**Normalisation** — Pixel values scaled from `[0–255]` to `[0–1]`:
```python
X = X.astype('float32') / 255.0
```

**Reshaping** — 2D images flattened to 1D vectors for Dense layers:
```python
X = X.reshape(-1, 28 * 28)  # (n, 28, 28) → (n, 784)
```

**Splitting** — Stratified split to preserve class balance across all sets:
```python
# stratify=y ensures proportional class distribution
# Train: 49,500 | Validation: 10,500 | Test: 10,000
```

---

## Model Architecture

```
Input Layer      →  784 neurons (28×28 flattened)
Dense Layer 1    →  128 neurons | ReLU | Dropout 30%
Dense Layer 2    →  64 neurons  | ReLU | Dropout 30%
Output Layer     →  10 neurons  | Softmax
```

### Training Configuration

| Setting | Value |
|---------|-------|
| Optimiser | Adam (`lr = 0.001`) |
| Loss Function | Sparse Categorical Cross-Entropy |
| Batch Size | 64 |
| Max Epochs | 30 |
| Early Stopping | Patience = 5 (monitors `val_loss`) |
| Regularisation | Dropout = 0.30 per hidden layer |
| Configs Tested | 5 architectures |

---

## Key Design Decisions

1. **Reusable model builder** — one function handles all 5 tested configurations via parameters (hidden layers, neurons, activation, dropout, learning rate)
2. **Neuron halving per layer** — each additional hidden layer gets `neurons // 2`, compressing features progressively like a funnel
3. **Softmax output** — converts raw logits to probabilities summing to 1, ideal for multi-class classification
4. **Stratified split** — `stratify=y` ensures no class imbalance is introduced during data splitting
5. **Restore best weights** — early stopping rolls back to the epoch with the lowest `val_loss`, preventing accidental underfitting

---

## Results

| Metric | Score |
|--------|-------|
| **Test Accuracy** | **88.54%** |
| Test Loss | 0.3334 |
| Best Val Accuracy | 89.85% (Config 2 — 2 layers, 128 neurons) |
| Macro F1 Score | 0.88 |
| Best Class (F1) | Trouser / Bag — 0.98 |
| Hardest Class | Shirt — 0.69 |
| Total Errors | 1,146 / 10,000 |

### Top Misclassifications

| Predicted | True Label | Count |
|-----------|------------|-------|
| T-shirt | Shirt | 137 |
| Pullover | Coat | 112 |
| Coat | Pullover | 104 |
| Pullover | Shirt | 89 |
| Coat | Shirt | 82 |

Upper-body tops (Shirt, Coat, Pullover, T-shirt) are the hardest to distinguish due to visual similarity. Structurally distinct items like Trouser, Bag, and Sandal achieved near-perfect F1 scores of 0.97–0.98.

### Learning Curves

The model converges rapidly in the first 5 epochs and stabilises around epoch 15–20. Training and validation curves remain close throughout, indicating good generalisation with no significant overfitting. Early stopping triggered at epoch 25.

---

## Future Work

- **CNN layers** — Convolutional layers would better capture spatial features in clothing images
- **Batch Normalisation** — stabilise and speed up training across deeper architectures
- **Data Augmentation** — random flips and rotations to improve robustness on visually similar classes
- **Learning Rate Scheduling** — dynamic LR decay to fine-tune convergence

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

### 3. Load the dataset

The dataset loads automatically via Keras — no manual download needed:

```python
from tensorflow.keras.datasets import fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
```

### 4. Run the notebook

```bash
jupyter notebook
```

---

## Author

**Ariyike** — [github.com/onikeyek](https://github.com/onikeyek)
