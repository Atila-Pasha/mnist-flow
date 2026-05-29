# 🔢 MNIST Handwritten Digit Classification with PyTorch

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)

![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=for-the-badge&logo=pytorch)

![Dataset](https://img.shields.io/badge/Dataset-MNIST-green?style=for-the-badge)

![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

</div>

---

## 📌 About The Project

This project is a simple yet complete implementation of a **Multi-Layer Perceptron (MLP)** using PyTorch for handwritten digit classification on the MNIST dataset.

The main purpose of this repository was not only to train a neural network, but also to build a clean and modular deep learning project structure similar to real-world machine learning workflows.

The model receives a handwritten digit image as input and predicts which number (0–9) the image represents.

Although the architecture is relatively simple, the model achieves strong performance and demonstrates the core concepts behind deep learning with PyTorch.

---

# 🧠 Model Architecture

The network used in this project is a fully connected neural network:

```text
784 → 128 → 64 → 10
```

### Explanation

- Input Layer:
  - 784 neurons (`28 × 28` flattened image)

- Hidden Layer 1:
  - 128 neurons
  - ReLU activation

- Hidden Layer 2:
  - 64 neurons
  - ReLU activation

- Output Layer:
  - 10 neurons
  - One for each digit class (`0-9`)

---

# 📂 Project Structure

```text
First_project/
│
├── data/
│
├── notebooks/
│   └── exploration.ipynb
│
├── src/
│   ├── model.py
│   ├── dataset.py
│   ├── train.py
│   ├── inference.py
│   └── utils.py
│
├── configs/
│   └── config.yaml
│
├── checkpoints/
│
├── requirements.txt
│
├── README.md
│
└── .gitignore
```

---

# 📊 Dataset

The project uses the famous **MNIST** dataset.

### Dataset Details

| Split | Number of Samples |
|------|------------------:|
| Training Set | 60,000 |
| Test Set | 10,000 |

Each image:

- is grayscale
- has size `28×28`
- belongs to one of the 10 digit classes

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/username/First_project.git

cd First_project
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

### Linux / macOS

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 🚀 Training

Run the training script:

```bash
python src/train.py
```

During training, the project will:

- Train the model
- Evaluate on the test set
- Track loss and accuracy
- Save the trained model
- Plot training curves
- Generate a confusion matrix

---

# 📈 Training Results

The model achieves approximately:

```text
~97% Test Accuracy
```

using only a simple MLP architecture.

---

# 📉 Visualizations

The training pipeline automatically generates:

- ✅ Loss Curve
- ✅ Accuracy Curve
- ✅ Confusion Matrix

These plots help analyze model performance and prediction behavior.

---

# 🔍 Confusion Matrix Analysis

Most predictions are concentrated along the main diagonal of the confusion matrix, which indicates that the model correctly classifies the majority of handwritten digits.

Some common mistakes include:

- `4` being confused with `9`
- `7` being confused with `2`
- occasional confusion between `3` and `5`

These errors are expected because certain handwritten digits can look visually similar depending on writing style.

---

# 💾 Saved Model

After training, the trained weights are automatically saved in:

```text
checkpoints/mnist_mlp.pth
```

---

# 🧪 Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Scikit-learn
- YAML

---

# 📁 Configuration

All hyperparameters are stored inside:

```text
configs/config.yaml
```

Example:

```yaml
batch_size: 64

learning_rate: 0.001

epochs: 10
```

This makes experimentation easier and keeps the training pipeline cleaner.

---

# 🛠 Future Improvements

Some possible improvements for future versions of this project:

- Add Dropout regularization
- Add Batch Normalization
- Implement CNN architecture
- Add TensorBoard logging
- Add command-line arguments
- Deploy using FastAPI

---

# 📚 What I Learned

Through this project I practiced:

- Building neural networks with PyTorch
- Writing training and evaluation loops
- Working with DataLoaders
- Tracking metrics and visualizing results
- Structuring machine learning projects
- Saving and loading trained models

---

# 🤝 Contributing

Contributions and suggestions are welcome.

Feel free to fork the repository and open a pull request.

---

# 📜 License

This project is licensed under the MIT License.

---

<div align="center">

### ⭐ If you found this project useful, consider giving it a star.

</div>