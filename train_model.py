import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.callbacks import Callback

# --- Configuration ---
IMG_HEIGHT, IMG_WIDTH = 64, 64
BATCH_SIZE = 60
EPOCHS = 40
MODEL_PATH = "isl_model.h5"
LABELS_PATH = "labels.npy"

# --- Load Dataset ---
print("Loading dataset...")
data = []
labels = []
class_labels = []

DATASET_PATH = "dataset"  # Make sure this path exists

for idx, class_name in enumerate(sorted(os.listdir(DATASET_PATH))):
    class_path = os.path.join(DATASET_PATH, class_name)
    if not os.path.isdir(class_path):
        continue

    class_labels.append(class_name)
    print(f"\nLoading images for class '{class_name}'...")

    img_files = [f for f in os.listdir(class_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    for i, img_name in enumerate(img_files, 1):
        img_path = os.path.join(class_path, img_name)
        try:
            img = tf.keras.preprocessing.image.load_img(img_path, color_mode="grayscale", target_size=(IMG_HEIGHT, IMG_WIDTH))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            data.append(img_array)
            labels.append(idx)

            # Print every 50 images
            if i % 50 == 0 or i == len(img_files):
                print(f"  Loaded {i}/{len(img_files)} images for class '{class_name}'")

        except Exception as e:
            print(f"Skipping {img_path} due to error: {e}")

# Convert to numpy arrays
data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)

# Shuffle data
data, labels = shuffle(data, labels, random_state=42)

# Print dataset summary
print("\nDataset loaded successfully.")
print("Class distribution:", Counter(labels))

# One-hot encode labels
labels_cat = to_categorical(labels, num_classes=len(class_labels))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    data, labels_cat, test_size=0.2, random_state=42, stratify=labels
)

# --- Save labels ---
np.save(LABELS_PATH, class_labels)

# --- Build Model ---
print("\nBuilding model...")
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(class_labels), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --- Data Augmentation ---
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
datagen.fit(X_train)

# --- Custom Callback for Epoch Progress ---
class EpochProgress(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"\n--- Epoch {epoch+1}/{self.params['epochs']} ---")
    def on_epoch_end(self, epoch, logs=None):
        acc = logs.get('accuracy', 0)
        val_acc = logs.get('val_accuracy', 0)
        loss = logs.get('loss', 0)
        val_loss = logs.get('val_loss', 0)
        print(f"Epoch {epoch+1} ended: loss={loss:.4f}, accuracy={acc:.4f}, val_loss={val_loss:.4f}, val_accuracy={val_acc:.4f}")

epoch_progress = EpochProgress()

# --- Train Model ---
print("\nTraining model...")
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    callbacks=[early_stopping, epoch_progress],
    verbose=0  # Disable default Keras verbose
)

# Save training history
with open("history.pkl", "wb") as f:
    pickle.dump(history.history, f)

# --- Save Model ---
model.save(MODEL_PATH)
print(f"\nModel saved at {MODEL_PATH}")

# --- Evaluate on Test Set ---
print("\nEvaluating model on test set...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc*100:.2f}%")

# --- Per-Class Accuracy ---
print("\nClassification Report (Per Class Accuracy):")
y_true = np.argmax(y_test, axis=1)
y_pred = np.argmax(model.predict(X_test), axis=1)

report = classification_report(
    y_true,
    y_pred,
    target_names=class_labels,
    digits=4
)
print(report)

# --- Confusion Matrix (Plain Text) ---
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)

# --- Plot Training History ---
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()
print("Training complete.")
