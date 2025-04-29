# cca_project
🌿 Capsicum Leaf SPAD Value Prediction Using Deep Learning
📘 Project Overview
This project aims to develop a deep learning-based system capable of predicting SPAD values (Soil Plant Analysis Development) — a key indicator of chlorophyll content in leaves — using image data of Capsicum (bell pepper) leaves. The entire process involved data collection from a real-world agricultural source, detailed image preprocessing, and the implementation of both custom-built CNNs and pre-trained deep learning models.

The project was undertaken as a part of an academic initiative to explore AI/ML applications in precision agriculture.

🗂️ Data Collection
Leaf images were collected directly from SKUAST University (Sher-e-Kashmir University of Agricultural Sciences and Technology), ensuring real-world variability in lighting, texture, and background conditions.

For each collected image, the corresponding SPAD value was manually recorded using a SPAD chlorophyll meter, which provided ground-truth labels for supervised learning.

This data served as the foundation for training and evaluating deep learning models capable of estimating chlorophyll content based on visual features.

⚠️ Important Note: The dataset size was limited in quantity, which posed certain challenges during model training. Careful consideration was given to model selection and evaluation to mitigate the effects of data scarcity.

🧹 Data Preprocessing
To ensure high-quality input data for model training, the following preprocessing steps were carried out:

Data Cleaning: Removed blurry, irrelevant, or corrupted images from the dataset to ensure consistent quality.

Resizing and Normalization: Standardized image dimensions and normalized pixel values for compatibility with neural networks.

Format Conversion: Ensured all images were correctly formatted and stored in a structured directory for easy loading.

Data Augmentation: Applied rotation, flipping, zooming, and brightness adjustments to artificially expand the training set and improve model generalization.

Label Matching: Cross-verified the integrity of SPAD value associations with each image to maintain label accuracy.

🧠 Model Development
To model the relationship between image features and SPAD values, both custom and transfer learning-based architectures were implemented:

✅ Custom CNN Architecture:

A lightweight Convolutional Neural Network was built from scratch.

Designed specifically for small datasets and tuned to balance complexity and generalization.

Included convolutional, pooling, dropout, and dense layers to learn hierarchical image features.

✅ Pre-trained Transfer Learning Models:

VGG19 and EfficientNetB0 were utilized with include_top=False to exclude their classification heads.

Feature extraction was performed using these models, followed by custom dense layers for SPAD regression.

Fine-tuning was selectively applied to adapt the models to the Capsicum dataset.

✅ Model Optimization:

Used Adam optimizer, mean squared error (MSE) as the loss function, and monitored validation loss to avoid overfitting.

Implemented early stopping, learning rate reduction, and dropout regularization to enhance model stability and performance.

🎯 Project Goals & Objectives
Build an AI-driven tool capable of estimating chlorophyll content non-destructively from leaf images.

Explore the applicability of deep learning in agriculture, particularly in plant health monitoring and nutrient assessment.

Compare performance between custom CNNs and state-of-the-art transfer learning models on a limited dataset.

Highlight how even with a constrained data environment, meaningful predictions can be extracted through robust preprocessing and model design.

🛠 Tools, Libraries, and Technologies Used
Programming Language: Python

Deep Learning Frameworks: TensorFlow, Keras

Image Processing: OpenCV, Pillow

Data Handling: Pandas, NumPy

Visualization: Matplotlib, Seaborn


🔍 Limitations
The size of the dataset was relatively small, which can limit the model’s ability to generalize to unseen data.

Model performance may not be optimal for deployment in diverse field conditions without additional training data.

Future improvements could include collecting a larger and more diverse dataset, and experimenting with ensemble or hybrid models.
