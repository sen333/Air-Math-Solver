"""
Train a CNN model to recognize handwritten math symbols (0-9, +, -, x, .)
Dataset: Handwritten Math Symbols from Kaggle
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import cv2
import matplotlib.pyplot as plt

# Configuration
IMG_SIZE = 45
BATCH_SIZE = 32
EPOCHS = 30


class MathSymbolDataLoader:
    """Load and preprocess the Kaggle Handwritten Math Symbols dataset"""

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.label_encoder = LabelEncoder()

    def load_data(self):
        """Load images and labels from dataset directory"""
        images = []
        labels = []

        # Map Kaggle folder names → display symbols
        class_mapping = {
            '0': '0', '1': '1', '2': '2', '3': '3', '4': '4',
            '5': '5', '6': '6', '7': '7', '8': '8', '9': '9',
            '+': '+', '-': '-', 'times': 'x'
        }

        print("Loading dataset...")
        for class_folder in os.listdir(self.dataset_path):
            class_path = os.path.join(self.dataset_path, class_folder)
            if not os.path.isdir(class_path):
                continue

            # Skip unknown folders instead of silently adding them
            if class_folder not in class_mapping:
                print(f"  Skipping unknown folder: {class_folder}")
                continue

            symbol = class_mapping[class_folder]

            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_path, img_file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                        img = img.astype('float32') / 255.0
                        images.append(img)
                        labels.append(symbol)

        if len(images) == 0:
            raise ValueError("No images loaded. Check the dataset folder structure.")

        print(f"Loaded {len(images)} images across {len(set(labels))} classes")

        X = np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        y = self.label_encoder.fit_transform(labels)

        return X, y, self.label_encoder.classes_


def create_cnn_model(num_classes):
    """Create CNN architecture for symbol classification"""
    model = keras.Sequential([
        # Explicit Input layer avoids the input_shape= deprecation warning
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        # Classifier head
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax'),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def plot_training_history(history, save_path):
    """Plot and save training metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Val')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Val')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)  # Fix: close figure to free memory
    print(f"Training history saved to {save_path}")


def train_model(dataset_path, save_dir='models'):
    """Complete training pipeline"""
    os.makedirs(save_dir, exist_ok=True)

    data_loader = MathSymbolDataLoader(dataset_path)
    X, y, class_names = data_loader.load_data()

    print(f"\nClass names: {class_names}")
    print(f"Dataset shape: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training: {len(X_train)}  Testing: {len(X_test)}")

    model = create_cnn_model(num_classes=len(class_names))
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6
        ),
    ]

    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1,
    )

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test Loss:     {test_loss:.4f}")

    # Save model
    model_path = os.path.join(save_dir, 'math_symbol_classifier.h5')
    model.save(model_path)
    print(f"Model saved  → {model_path}")

    class_names_path = os.path.join(save_dir, 'class_names.npy')
    np.save(class_names_path, class_names)
    print(f"Class names  → {class_names_path}")

    plot_training_history(history, os.path.join(save_dir, 'training_history.png'))
    return model, class_names, history


if __name__ == "__main__":
    print("=" * 60)
    print("DATASET SETUP INSTRUCTIONS")
    print("=" * 60)
    print("\n1. Download from Kaggle:")
    print("   https://www.kaggle.com/datasets/xainano/handwrittenmathsymbols")
    print("\n2. Extract to: ./data/handwritten_math_symbols/")
    print("\n3. Expected folders inside:")
    print("   0  1  2  3  4  5  6  7  8  9  +  -  times")
    print("\n" + "=" * 60)

    dataset_path = './data/handwritten_math_symbols'
    if os.path.exists(dataset_path):
        print("\n✓ Dataset found! Starting training...\n")
        model, class_names, history = train_model(dataset_path)
        print("\n✓ Training complete!")
    else:
        print(f"\n✗ Dataset not found at {dataset_path}")
        print("Please download and extract the dataset first.")
