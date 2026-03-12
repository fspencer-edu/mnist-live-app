# MNIST Live App

Load MNIST

Normalize and reshape

Apply augmentation that mimics app drawings

Train classifier

Evaluate on:

normal MNIST test set

app-like transformed test set

Save trained model

Inference pipeline

User draws a digit in frontend

Frontend sends PNG every ~200 ms while drawing

Backend:

converts image to grayscale

thresholds and crops the digit

centers it

resizes to MNIST-like 28×28

normalizes

Model predicts probabilities

Backend returns:

predicted digit

confidence

probabilities for all digits 0–9