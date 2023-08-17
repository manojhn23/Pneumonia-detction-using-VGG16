# Pneumonia Detection using Chest X-ray Images and VGG16

This project focuses on detecting pneumonia in chest X-ray images using a pre-trained VGG16 model. The goal is to predict whether a patient has pneumonia based on their chest X-ray images.

## Getting Started

### Prerequisites

- Python (>=3.6)
- TensorFlow
- Keras

### Dataset
Collect a labeled dataset of chest X-ray images. Categorize each image as either "pneumonia" or "normal". Ensure that you have a good distribution of both classes. Divide the dataset into training, validation, and test sets. The images are downloaded from the Kaggle website

### Model Architecture
The model architecture is built upon the VGG16 model, which is a popular deep learning architecture for image classification tasks. This project uses transfer learning by leveraging the pre-trained VGG16 model and fine-tuning it for pneumonia detection.
