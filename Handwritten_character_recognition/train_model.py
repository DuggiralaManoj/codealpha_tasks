import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


class Config:
    DATASET_VARIANT = 'byclass'

    BATCH_SIZE = 512
    EPOCHS = 20  #first we have trained 10 so  we start it from 11 then merge
    VALIDATION_SPLIT = 0.2
    IMG_SIZE = (28, 28)

    LEARNING_RATE = 0.001
    DROPOUT_RATE = 0.3


def find_dataset_files():
    print("Auto-detecting dataset files...")

    search_patterns = [
        "datasets/**/*.csv",
        "data/**/*.csv",
        "**/*emnist*.csv",
        "*.csv"
    ]

    found_files = {}

    for pattern in search_patterns:
        files = glob.glob(pattern, recursive=True)
        for file in files:
            filename = os.path.basename(file)
            found_files[filename] = file

    print(f" Found {len(found_files)} CSV files:")
    for filename, filepath in found_files.items():
        print(f"   {filename} -> {filepath}")

    return found_files


def get_dataset_files(variant):
    found_files = find_dataset_files()

    train_file = None
    test_file = None
    mapping_file = None

    for filename, filepath in found_files.items():
        if f'emnist-{variant}-train.csv' in filename:
            train_file = filepath
        elif f'emnist-{variant}-test.csv' in filename:
            test_file = filepath

    mapping_patterns = [
        f"**/emnist-{variant}-mapping.txt",
        f"**/*{variant}*mapping.txt"
    ]

    for pattern in mapping_patterns:
        files = glob.glob(pattern, recursive=True)
        if files:
            mapping_file = files[0]
            break

    print(f"Selected files for variant '{variant}':")
    print(f"   Train: {train_file}")
    print(f"   Test: {test_file}")
    print(f"   Mapping: {mapping_file}")

    return train_file, test_file, mapping_file


def load_mapping(mapping_file):
    if not mapping_file or not os.path.exists(mapping_file):
        print(f"Mapping file not found or not specified: {mapping_file}")
        return None

    mapping = {}
    try:
        with open(mapping_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        class_id = int(parts[0])
                        ascii_code = int(parts[1])
                        character = chr(ascii_code)
                        mapping[class_id] = character
        print(f" Loaded mapping: {len(mapping)} classes")
        return mapping
    except Exception as e:
        print(f" Error loading mapping file: {e}")
        return None


def load_csv_data(csv_file, max_samples=None):
    print(f" Loading data from: {csv_file}")

    if not csv_file or not os.path.exists(csv_file):
        raise FileNotFoundError(f"Dataset file not found: {csv_file}")

    try:
        df = pd.read_csv(csv_file, header=None)
        print(f" CSV shape: {df.shape}")

        if max_samples:
            df = df.sample(n=min(max_samples, len(df)), random_state=42)

        labels = df.iloc[:, 0].values
        images = df.iloc[:, 1:].values

        print(f"Loaded {len(labels)} samples")
        print(f"Label range: {labels.min()} to {labels.max()}")
        print(f"Unique classes: {len(np.unique(labels))}")
        print(f"Image data shape: {images.shape}")

        return images, labels

    except Exception as e:
        print(f" Error loading CSV file: {e}")
        print(" Trying alternative loading methods...")

        for sep in [',', ';', '\t']:
            try:
                df = pd.read_csv(csv_file, header=None, sep=sep)
                if df.shape[1] > 1:
                    print(f"Loaded with separator '{sep}'")
                    labels = df.iloc[:, 0].values
                    images = df.iloc[:, 1:].values
                    return images, labels
            except:
                continue

        raise ValueError(f"Could not load CSV file: {csv_file}")


def preprocess_data(images, labels, mapping=None):
    print(" Preprocessing data...")

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    if images.shape[1] == 784:
        images = images.reshape(-1, 28, 28, 1)
    elif len(images.shape) == 3:
        images = np.expand_dims(images, -1)

    images = images / 255.0

    if mapping:
        valid_indices = np.isin(labels, list(mapping.keys()))
        images = images[valid_indices]
        labels = labels[valid_indices]

        unique_labels = sorted(list(set(labels)))
        label_to_index = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        labels = np.array([label_to_index[label] for label in labels])

        print(f"Filtered to {len(images)} valid samples")
        print(f"Label mapping created: {len(unique_labels)} classes")
    else:
        unique_labels = sorted(list(set(labels)))
        if min(unique_labels) != 0:
            label_to_index = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
            labels = np.array([label_to_index[label] for label in labels])

    print(f" Final image shape: {images.shape}")
    print(f" Final labels shape: {labels.shape}")
    print(f"Number of classes: {len(np.unique(labels))}")
    print(f"Label range: {labels.min()} to {labels.max()}")

    return images, labels


def build_improved_model(num_classes, input_shape=(28, 28, 1)):
    print(f" Building improved CNN model for {num_classes} classes...")

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Dropout(0.2),

        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Dropout(0.2),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.3),

        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(Config.DROPOUT_RATE),
        Dense(256, activation='relu'),
        Dropout(Config.DROPOUT_RATE),
        Dense(num_classes, activation='softmax')
    ])

    optimizer = Adam(learning_rate=Config.LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
    return model


def setup_callbacks(model_filename):
    callbacks = [
        ModelCheckpoint(
            filepath=model_filename,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
            save_weights_only=False
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    return callbacks


def plot_training_history(history, variant):
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plot_filename = f'training_history_{variant}.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.show()
        print(f" Training history saved as: {plot_filename}")
    except Exception as e:
        print(f"⚠ Could not save plot: {e}")


def train_model():
    print(f" Starting EMNIST {Config.DATASET_VARIANT.upper()} training...")

    train_file, test_file, mapping_file = get_dataset_files(Config.DATASET_VARIANT)

    if not train_file:
        print(f" Could not find training file for variant: {Config.DATASET_VARIANT}")
        print("Available variants might be:")
        found_files = find_dataset_files()
        variants = set()
        for filename in found_files.keys():
            if 'emnist-' in filename and '-train.csv' in filename:
                variant = filename.replace('emnist-', '').replace('-train.csv', '')
                variants.add(variant)
        for variant in sorted(variants):
            print(f"  - {variant}")
        return None, None

    if not test_file:
        print(f" Could not find test file for variant: {Config.DATASET_VARIANT}")
        return None, None

    mapping = load_mapping(mapping_file)

    try:
        X_train, y_train = load_csv_data(train_file)
        X_train, y_train = preprocess_data(X_train, y_train, mapping)

        X_test, y_test = load_csv_data(test_file)
        X_test, y_test = preprocess_data(X_test, y_test, mapping)

        num_classes = len(np.unique(np.concatenate([y_train, y_test])))

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=Config.VALIDATION_SPLIT,
            random_state=42,
            stratify=y_train
        )

        print(f" Dataset splits:")
        print(f"   Training: {X_train.shape[0]} samples")
        print(f"   Validation: {X_val.shape[0]} samples")
        print(f"   Test: {X_test.shape[0]} samples")
        print(f"   Classes: {num_classes}")

        model = build_improved_model(num_classes)

        model_filename = f"emnist_{Config.DATASET_VARIANT}_model.h5"
        callbacks = setup_callbacks(model_filename)

        os.makedirs("models", exist_ok=True)

        print(" Training started...")
        history = model.fit(
            X_train, y_train,
            batch_size=Config.BATCH_SIZE,
            epochs=Config.EPOCHS,
            initial_epoch=10,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )

        print(" Evaluating on test set...")
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f" Final Test Accuracy: {test_accuracy:.4f}")

        plot_training_history(history, Config.DATASET_VARIANT)

        print(f"Model saved as: {model_filename}")

        if mapping:
            mapping_output = f"class_mapping_{Config.DATASET_VARIANT}.txt"
            with open(mapping_output, 'w') as f:
                for class_id, character in mapping.items():
                    f.write(f"{class_id}: {character}\n")
            print(f" Character mapping saved as: {mapping_output}")

        return model, history

    except Exception as e:
        print(f" Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == '__main__':
    print("=" * 50)
    print(" DATASET DISCOVERY")
    print("=" * 50)
    find_dataset_files()

    print("\n" + "=" * 50)
    print(" TRAINING")
    print("=" * 50)

    try:
        model, history = train_model()
        if model is not None:
            print(" Training completed successfully!")
        else:
            print(" Training failed. Please check the error messages above.")
    except Exception as e:
        print(f" Fatal error: {str(e)}")
        import traceback

        traceback.print_exc()