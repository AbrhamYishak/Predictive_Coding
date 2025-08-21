# Predictive Coding Network on MNIST

This project implements a **Predictive Coding Neural Network (PCN)** from scratch using **PyTorch**.  
The model is trained and tested on the **MNIST dataset** (handwritten digits), and learns using a predictive coding–style weight update mechanism instead of standard backpropagation.

---

## Features
- Implementation of a multi-layer predictive coding network.
- Uses lateral connections for recurrent inference updates.
- Training loop with manual weight updates (no PyTorch optimizers).
- Evaluation on MNIST with accuracy and loss tracking.
- Visualization of training vs validation loss and accuracy.

---

## Requirements
Make sure you have **Python 3.8+** and install the required dependencies:

```bash
pip install torch torchvision torchinfo matplotlib numpy
```

---

## How to Run

### Linux / macOS
```bash
python3 main.py
```

### Windows
```bash
python main.py
```

---

## Code Overview
- **`PredictiveCodingNet`**: Custom neural network implementing predictive coding dynamics.
- **Training loop**: Updates weights based on local error signals.
- **Testing loop**: Evaluates accuracy and mean squared error loss on MNIST test set.
- **Plotting**: Generates accuracy and loss curves after training.

---

## Example Output
During training, the script will print:

```
Epoch 1/5, Training Loss: 0.0832, Training Accuracy: 0.7540, Val Loss: 0.0701, Val Accuracy: 0.7895
Epoch 2/5, Training Loss: 0.0623, Training Accuracy: 0.8412, Val Loss: 0.0587, Val Accuracy: 0.8670
...
```

After training, a plot will appear showing **accuracy and loss curves** over epochs.

---

## File Structure
```
.
├── main.py        # Main script containing model, training, evaluation, plotting
├── data/          # MNIST dataset (downloaded automatically)
└── README.md      # This file
```

---

## Notes
- The network uses iterative **inference steps** to update hidden states before weight updates.
- Weight updates are applied manually without `torch.optim` to mimic predictive coding principles.
- You can tweak hyperparameters such as:
  - `num_epochs`
  - `h1_dim, h2_dim, h3_dim`
  - `lr_weight` (learning rate for weights)
  - `lr_state` (learning rate for state updates)
  - `num_infer_steps`

---
