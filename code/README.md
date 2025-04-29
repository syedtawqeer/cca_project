# cca_project
📘 Project Overview

This project focuses on developing AI/ML models to predict SPAD values (a measure of chlorophyll content) from images of Capsicum (bell pepper) leaves. The work involved multiple stages, including data collection, preprocessing, model building, and evaluation. Below is a summary of the key steps and tools used throughout the project:

🗂️ Data Collection

Collected a dataset of Capsicum leaf images from SKUAST University (Sher-e-Kashmir University of Agricultural Sciences and Technology).

Each image was paired with its corresponding SPAD value, measured using a SPAD meter to indicate chlorophyll content.

🧹 Data Preprocessing

Cleaned and organized the image data to remove any noise, irrelevant samples, or corrupted files.

Applied resizing and normalization techniques to prepare the images for model training.

Ensured consistent formatting for input into deep learning models.

🧠 Model Development

Built a Convolutional Neural Network (CNN) from scratch tailored to the characteristics of the dataset.

Utilized pre-trained models with the top classification layers removed (include_top=False) for feature extraction and transfer learning:

VGG19

EfficientNetB0

Fine-tuned these models on the Capsicum dataset to improve prediction accuracy.

Employed techniques such as data augmentation, dropout, and learning rate scheduling to enhance model generalization.

🎯 Objective

The primary goal was to accurately predict SPAD values from leaf images, enabling non-destructive estimation of chlorophyll content using AI.

🔧 Tools and Libraries Used

Python

TensorFlow / Keras

NumPy, Pandas

OpenCV for image processing

Matplotlib / Seaborn for visualization
