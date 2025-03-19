import os
import gc
import numpy as np
import tensorflow as tf
import rasterio
from tensorflow.keras.layers import (Input, Conv2D, Conv2DTranspose, SeparableConv2D, 
                                     Dropout, BatchNormalization)
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt

# Disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Define loss functions
def dice_loss(y_true, y_pred, smooth=1e-7):
    y_true_f = tf.keras.backend.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = tf.keras.backend.flatten(tf.cast(y_pred, tf.float32))
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
    return 1 - dice


loss_fn = tf.keras.losses.BinaryFocalCrossentropy()


tf.keras.backend.clear_session()
gc.collect()

# Paths
image_folder = 'D:\\GitHub\\Segmentation\\data\\images'
mask_folder = 'D:\\GitHub\\Segmentation\\data\\labels'

# Get file lists
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.tif')])
mask_files = sorted([f for f in os.listdir(mask_folder) if f.endswith('.png')])

selected_channels = list(range(12))  # Select 12 channels
target_size = (256, 256)

# Image resizing function
def resize_image(image, target_size):
    return np.array(Image.fromarray(image).resize(target_size, Image.BILINEAR))

# Data preprocessing function
def preprocess_data(image_file, mask_file, selected_channels, target_size):
    with rasterio.open(image_file) as src:
        image = src.read()
    
    image_selected = image[selected_channels, :, :]
    image_selected = (image_selected - np.min(image_selected)) / (np.max(image_selected) - np.min(image_selected) + 1e-8)
    
    image_resized = np.array([resize_image(img, target_size) for img in image_selected])
    image_resized = np.moveaxis(image_resized, 0, -1)
    
    mask = np.array(Image.open(mask_file).convert('L'))
    mask_resized = resize_image(mask, target_size)
    mask_resized = (mask_resized > 0).astype(np.uint8)  # Ensure binary labels
    
    return image_resized, mask_resized

# Load and preprocess data
X, y = [], []
for image_file, mask_file in zip(image_files, mask_files):
    image_path, mask_path = os.path.join(image_folder, image_file), os.path.join(mask_folder, mask_file)
    image_resized, mask_resized = preprocess_data(image_path, mask_path, selected_channels, target_size)
    X.append(image_resized)
    y.append(mask_resized)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.uint8)

# Print class distribution
print(f"Flood pixels: {np.sum(y) / y.size:.4f}")

# Split dataset into train and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to TensorFlow dataset
AUTOTUNE = tf.data.AUTOTUNE

def preprocess(image, mask):
    image = tf.cast(image, tf.float32) / 255.0
    mask = tf.cast(mask, tf.float32)
    return image, mask

batch_size = 8  

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.map(preprocess).batch(batch_size).shuffle(100).prefetch(AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_dataset = val_dataset.map(preprocess).batch(batch_size).prefetch(AUTOTUNE)

def deeplabv3_plus_multispectral(input_shape):
    inputs = Input(shape=input_shape)

    multispectral_adapter = Conv2D(3, (1, 1), activation='relu', padding='same')(inputs)

    resnet_backbone = ResNet50(weights="imagenet", include_top=False, input_shape=(None, None, 3))
    
    resnet_features = resnet_backbone(multispectral_adapter)  

    for layer in resnet_backbone.layers[1:]:
        layer.trainable = False  

    up1 = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding="same", activation="relu")(resnet_features)
    up1 = SeparableConv2D(256, (3, 3), padding="same", activation="relu")(up1)
    up1 = Dropout(0.7)(up1)
    
    up2 = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding="same", activation="relu")(up1)
    up2 = SeparableConv2D(128, (3, 3), padding="same", activation="relu")(up2)
    up2 = Dropout(0.7)(up2)
    
    up3 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same", activation="relu")(up2)
    up4 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same", activation="relu")(up3)
    up5 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same", activation="relu")(up4)  
    outputs = Conv2D(1, (1, 1), activation="sigmoid")(up5)  

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss=loss_fn, metrics=['accuracy'])  # Increased learning rate
    
    return model

# Create model
model = deeplabv3_plus_multispectral((256, 256, 12))
# Define learning rate reduction and early stopping callbacks
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    verbose=1,
    min_lr=1e-6
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# Train model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=20,
    callbacks=[reduce_lr, early_stop]
)

# Print final accuracies
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
print(f"Final Train Accuracy: {final_train_acc:.4f}")
print(f"Final Validation Accuracy: {final_val_acc:.4f}")
