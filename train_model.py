import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D

# Path to the dataset
dataset_dir = 'combined_dataset/IND_and_NEP'  # Path to the dataset folder

# Define AQI classes and their corresponding numeric labels
label_map = {
    "a_Good": 0,
    "b_Moderate": 1,
    "c_Unhealthy_for_Sensitive_Groups": 2,
    "d_Unhealthy": 3,
    "e_Very_Unhealthy": 4,
    "f_Severe": 5
}

# Initialize lists to hold image data and labels
images = []
labels = []

# Function to load and preprocess images
def load_and_preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))  # Resize to 224x224
    img_array = image.img_to_array(img) / 255.0  # Normalize image values (0-1)
    return img_array

# Loop over each AQI class folder (e.g., a_Good, b_Moderate)
for folder_name, label in label_map.items():
    folder_path = os.path.join(dataset_dir, folder_name)
    
    # Check if the folder exists
    if os.path.exists(folder_path):
        # Loop over all images in the folder
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            
            # Load and preprocess the image
            img_array = load_and_preprocess_image(img_path)
            images.append(img_array)
            labels.append(label)  # Append the corresponding label for the AQI class

# Convert images and labels to numpy arrays
images = np.array(images)
labels = np.array(labels)

# One-hot encode the labels
labels = to_categorical(labels)

# Split the data into training and validation sets (80-20 split)
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

print(f"Training images shape: {X_train.shape}")
print(f"Validation images shape: {X_val.shape}")

# Use a pre-trained ResNet50 model (without the top classification layers)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze the pre-trained layers

# Add custom layers on top of the base model
model = models.Sequential([
    base_model,
    GlobalAveragePooling2D(),  # Pool the features to reduce dimensionality
    layers.Dense(128, activation='relu'),  # Fully connected layer
    layers.Dense(6, activation='softmax')  # Output layer for 6 AQI classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Train the model
history = model.fit(
    X_train, y_train, 
    epochs=10,  # Number of epochs, adjust based on your needs
    batch_size=32,  # Batch size
    validation_data=(X_val, y_val)  # Validation data for evaluation
)

# Evaluate the model on the validation set
val_loss, val_acc = model.evaluate(X_val, y_val)
print(f"Validation loss: {val_loss}")
print(f"Validation accuracy: {val_acc}")

# Save the trained model
model.save('air_quality_model.h5')

