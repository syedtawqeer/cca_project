import os
import numpy as np
import imgaug.augmenters as iaa
from PIL import Image

# Define augmentation pipeline
seq = iaa.Sequential([
    iaa.Fliplr(0.5),   # horizontally flip 50% of the images
    iaa.Flipud(0.5),   # vertically flip 50% of the images
    iaa.Affine(rotate=(-10, 10)),  # rotate by -10 to +10 degrees
    iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),  # add gaussian noise
    iaa.GaussianBlur(sigma=(0, 3.0)),  # blur images with a sigma of 0 to 3.0
    iaa.Grayscale(alpha=(0.0, 1.0)),  # convert images to grayscale
    iaa.Sometimes(0.5, iaa.ChannelShuffle(1.0)),  # shuffle channels
    iaa.Sometimes(0.5, iaa.Invert(1.0)),  # invert pixel values
    iaa.Sometimes(0.5, iaa.ContrastNormalization((0.5, 2.0))),  # contrast normalization
    iaa.Sometimes(0.5, iaa.Multiply((0.5, 1.5), per_channel=0.5)),  # multiply pixel values
    iaa.Sometimes(0.5, iaa.LinearContrast((0.5, 2.0))),  # linear contrast adjustment
    iaa.Sometimes(0.5, iaa.HistogramEqualization()),  # histogram equalization
    iaa.Sometimes(0.5, iaa.GammaContrast((0.5, 2.0))),  # gamma contrast adjustment
    iaa.Sometimes(0.5, iaa.CLAHE()),  # contrast limited adaptive histogram equalization
    iaa.Sometimes(0.5, iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV")),  # RGB to HSV
    iaa.Sometimes(0.5, iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="LAB")),  # RGB to LAB
    iaa.Sometimes(0.5, iaa.Grayscale(alpha=(0.0, 1.0))),  # convert images to grayscale
    iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 1.0))),  # gaussian blur
    iaa.Sometimes(0.5, iaa.MotionBlur(k=15, angle=[-45, 45])),  # motion blur
    iaa.Sometimes(0.5, iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 2.0))),  # sharpen images
    iaa.Sometimes(0.5, iaa.AddToBrightness((-50, 50))),  # change brightness of images
    iaa.Sometimes(0.5, iaa.AddToHueAndSaturation((-20, 20))),  # change hue and saturation
    iaa.Sometimes(0.5, iaa.MultiplyHueAndSaturation((0.5, 1.5))),  # multiply hue and saturation
    iaa.Sometimes(0.5, iaa.LinearContrast((0.5, 2.0))),  # linear contrast adjustment
    iaa.Sometimes(0.5, iaa.AllChannelsCLAHE()),  # contrast limited adaptive histogram equalization for all channels
    iaa.Sometimes(0.5, iaa.AllChannelsHistogramEqualization()),  # histogram equalization for all channels
    iaa.Sometimes(0.5, iaa.AddToHue((-50, 50))),  # change hue of images
    iaa.Sometimes(0.5, iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6))),  # sigmoid contrast adjustment
    iaa.Sometimes(0.5, iaa.FastSnowyLandscape(lightness_threshold=(128, 255))),  # snowy landscape effect
    iaa.Sometimes(0.5, iaa.Superpixels(p_replace=(0.1, 1.0), n_segments=(16, 128))),  # superpixels
    iaa.Sometimes(0.5, iaa.Cartoon()),  # cartoon effect
    iaa.Sometimes(0.5, iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5))),  # emboss effect
    iaa.Sometimes(0.5, iaa.AverageBlur(k=(2, 7))),  # average blur
    iaa.Sometimes(0.5, iaa.MedianBlur(k=(3, 11))),  # median blur
    iaa.Sometimes(0.5, iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),  # elastic transformation
    iaa.Sometimes(0.5, iaa.PerspectiveTransform(scale=(0.01, 0.1))),  # perspective transformation
    iaa.Sometimes(0.5, iaa.PiecewiseAffine(scale=(0.01, 0.05))),  # piecewise affine transformation
    iaa.Sometimes(0.5, iaa.CoarseDropout(0.02, size_percent=0.1)),  # coarse dropout
    iaa.Sometimes(0.5, iaa.GridDropout(0.1, threshold=(0.1, 0.3), per_channel=True)),  # grid dropout
    iaa.Sometimes(0.5, iaa.Cutout(nb_iterations=(1, 3), size=0.2, squared=False)),  # cutout
    iaa.Sometimes(0.5, iaa.RandomBrightnessContrast()),  # random brightness and contrast
    iaa.Sometimes(0.5, iaa.Blur(sigma=(0, 3.0))),  # blur
    iaa.Sometimes(0.5, iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 2.0))),  # sharpen
])

# Calculate total number of augmented images
total_images = 100
target_images = 2000
augmentation_factor = target_images / total_images

# Augment images to reach target number
for image_name in os.listdir("labeled_images"):
    image_path = os.path.join("labeled_images", image_name)
    image = Image.open(image_path)
    image = np.array(image)
    
    for i in range(int(augmentation_factor)):
        augmented_images = seq(images=[image])
        for j, augmented_image in enumerate(augmented_images):
            augmented_image = Image.fromarray(augmented_image)
            augmented_image.save(f"augmented_images/{image_name.split('.')[0]}aug{i*5 + j}.jpg")


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data Preparation    KCBKCsdc
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
        'augmented_images',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training')

validation_generator = train_datagen.flow_from_directory(
        'augmented_images',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation')

# Build the CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the Model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the Model
history = model.fit(
      train_generator,
      steps_per_epoch=train_generator.samples/train_generator.batch_size ,
      epochs=10,
      validation_data=validation_generator,
      validation_steps=validation_generator.samples/validation_generator.batch_size)

# Evaluate the Model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)