import os
import shutil
from sklearn.model_selection import train_test_split

# Define the dataset path
original_dataset_path = "/kaggle/input/breast-cancer-dataset/BreaKHis_v1/histology_slides"
output_dataset_path = "/kaggle/working/preprocessed_dataset"

# Create folders for the processed dataset
train_path = os.path.join(output_dataset_path, "train")
validation_path = os.path.join(output_dataset_path, "validation")
# Define categories (benign and malignant)
categories = ["benign", "malignant"]

# Train-validation split ratio
train_ratio = 0.8

# Preprocess and flatten the dataset
for category in categories:
    category_path = os.path.join(original_dataset_path, "breast", category)
    output_train_path = os.path.join(train_path, category)
    output_val_path = os.path.join(validation_path, category)
    
    os.makedirs(output_train_path, exist_ok=True)
    os.makedirs(output_val_path, exist_ok=True)
    
    all_images = []
    for root, dirs, files in os.walk(category_path):
        for file in files:
            if file.endswith((".jpg", ".png")):  # Add other image extensions if needed
                all_images.append(os.path.join(root, file))
    
    train_images, val_images = train_test_split(all_images, train_size=train_ratio, random_state=42)
    
    for img in train_images:
        shutil.copy(img, output_train_path)
    for img in val_images:
        shutil.copy(img, output_val_path)

print("Dataset preprocessing completed!")
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
# Define image size and batch size
img_size = (224, 224)
batch_size = 32

# Data generators for train and validation
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)
# Load the ResNet50 model as the base
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))

# Freeze the base model layers
base_model.trainable = False

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model with frozen base
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,
    verbose=1
)
# Unfreeze the base model for fine-tuning
base_model.trainable = True

# Optionally freeze the first few layers of the base model
for layer in base_model.layers[:100]:
    layer.trainable = False

# Recompile the model with a smaller learning rate
model.compile(optimizer=Adam(learning_rate=0.00001), loss='binary_crossentropy', metrics=['accuracy'])

# Fine-tune the model
history_fine_tune = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,
    verbose=1
)
# Predict on the validation set
val_preds = model.predict(validation_generator)
val_preds_classes = (val_preds > 0.5).astype(int).flatten()

# Check class distribution in validation set
print(f"Class distribution in validation set: {dict(zip(validation_generator.class_indices, [sum(validation_generator.classes == k) for k in validation_generator.class_indices.values()]))}")

# Safely calculate AUC-ROC
if len(set(validation_generator.classes)) > 1:  # Check if both classes are present
    auc = roc_auc_score(validation_generator.classes, val_preds)
    print(f"AUC-ROC: {auc:.4f}")
else:
    print("Cannot calculate AUC-ROC: Only one class present in the validation set.")

# Classification report
print("Classification Report:")
print(classification_report(validation_generator.classes, val_preds_classes))
# Plot training results
plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'] + history_fine_tune.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'] + history_fine_tune.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'] + history_fine_tune.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'] + history_fine_tune.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')

plt.show()
