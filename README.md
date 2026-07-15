# -Persian-Character-Recognition-using-Deep-Learning
A deep learning-based Persian character recognition system capable of identifying Persian letters, digits, and their positional forms (Beginning, Middle, Ending, Standalone).  The model is implemented in PyTorch and includes an interactive Streamlit interface for real-time inference.


## Features

- Persian alphabet recognition
- Persian digit recognition
- Position-aware character classification
- CNN implemented with PyTorch
- Interactive Streamlit interface
- GPU acceleration (CUDA supported)
- Confidence score prediction
- Easy-to-use image uploader


## Supported Characters

✔ Persian Letters

✔ Persian Numbers

✔ Initial Forms

✔ Middle Forms

✔ Final Forms

✔ Standalone Forms



## Model Architecture

The model consists of:

- 4 Convolutional Blocks
- Batch Normalization
- ReLU Activation
- MaxPooling
- Fully Connected Layers
- Dropout Regularization


## Dataset

The dataset contains labeled Persian characters including:

- Letters
- Numbers
- Initial Forms
- Middle Forms
- Final Forms
- Standalone Forms

Images are resized to:
128 × 128 pixels


## Technologies

- Python
- PyTorch
- Torchvision
- Streamlit
- PIL
- NumPy

## Run
```bash
streamlit run app.py
```

## Results

Test Accuracy: 97.34%


## Position-Aware Recognition
Unlike standard character classification systems, this project distinguishes different contextual forms of Persian letters:

- Beginning
- Middle
- Ending
- Standalone

This makes the model more suitable for Persian OCR and document analysis applications.
