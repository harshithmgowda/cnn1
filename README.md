# CIFAR-10 AI Image Classifier

A high-performance image classification web application trained on the **CIFAR-10** dataset, featuring a **Swiss International Typographic Style** user interface and lightweight **ONNX Runtime** inference ready for deployment to **Vercel** and other cloud platforms.

---

## Features

- **Custom CNN Architecture**: 3 convolutional blocks (Conv2D + ReLU + MaxPool2D) followed by dense classification layers.
- **ONNX Runtime Inference**: Converted from PyTorch to ONNX format to shrink deployment size from ~4 GB down to ~50 MB, well within Vercel's serverless limits.
- **Swiss International Typographic Style UI**:
  - Modular 1px grid layout with visible boundaries
  - Minimalist palette: Swiss Red (`#E53935`), Ink Black (`#0A0A0A`), Chalk Paper (`#F7F7F5`)
  - Inter Black typography with high-contrast hierarchy
  - Zero border-radius on all elements
  - Real-time drag-and-drop file upload, preview, and top-3 confidence rankings
- **Production Ready**: Configured for instant deployment on Vercel via `@vercel/python` serverless functions.

---

## CIFAR-10 Classes

| Index | Class | Index | Class |
|:-----:|:-----:|:-----:|:-----:|
| 0 | airplane | 5 | dog |
| 1 | automobile | 6 | frog |
| 2 | bird | 7 | horse |
| 3 | cat | 8 | ship |
| 4 | deer | 9 | truck |

---

## Project Structure

```text
├── api/
│   └── index.py            # Vercel serverless function entrypoint
├── models/
│   ├── cifar10_cnn.onnx    # ONNX model graph
│   ├── cifar10_cnn.onnx.data # ONNX model weights (~2.5 MB)
│   └── cifar10_cnn.pth     # Original PyTorch weights (~2.5 MB)
├── static/
│   ├── css/style.css       # Swiss International Style CSS
│   └── js/script.js        # Drag-and-drop & classification UI logic
├── templates/
│   └── index.html          # Modular grid template
├── app.py                  # Flask application & ONNX inference
├── convert_to_onnx.py      # PyTorch to ONNX exporter
├── train_and_save.py       # Model training script
├── sample.ipynb            # Jupyter exploration notebook
├── vercel.json             # Vercel route configuration
├── requirements.txt        # Lightweight dependencies (onnxruntime, Flask, Pillow, numpy)
└── README.md
```

---

## Local Development

1. **Clone the repository:**
   ```bash
   git clone https://github.com/harshithmgowda/cnn1.git
   cd cnn1
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the Flask server:**
   ```bash
   python app.py
   ```

4. Open your browser and navigate to:
   ```text
   http://127.0.0.1:5000
   ```
