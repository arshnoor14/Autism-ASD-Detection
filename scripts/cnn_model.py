import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from pathlib import Path

# ====== Config ======
BASE_DIR = Path("data")
TRAIN_DIR = BASE_DIR / "train"
LABELS_CSV = BASE_DIR / "train_labels.csv"

BATCH_SIZE = 32
IMG_HEIGHT = 180
IMG_WIDTH = 180
EPOCHS = 10
SEED = 42

# ====== Model Architecture ======
def create_cnn_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.4),  # Helps reduce overfitting
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ====== Training Logic ======
def train_cnn_model():
    # Load and preprocess labels
    df = pd.read_csv(LABELS_CSV)
    df['label'] = df['label'].astype(str)

    # Train-validation split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=SEED)

    # Data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_data = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=str(TRAIN_DIR),
        x_col="filename",
        y_col="label",
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=True,
        seed=SEED
    )

    val_data = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=str(TRAIN_DIR),
        x_col="filename",
        y_col="label",
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False
    )

    # Create the model
    model = create_cnn_model()
    model.summary()

    # Callbacks
    checkpoint = ModelCheckpoint(
        "best_model.keras", monitor="val_accuracy", save_best_only=True, mode="max", verbose=1
    )

    early_stop = EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
    )

    # Train the model
    model.fit(
        train_data,
        epochs=EPOCHS,
        steps_per_epoch=int(np.ceil(train_data.samples / BATCH_SIZE)),
        validation_data=val_data,
        validation_steps=int(np.ceil(val_data.samples // BATCH_SIZE)),
        callbacks=[checkpoint, early_stop]
    )

    # Save final model
    model.save("autism_classification_model.keras")
    print(" Model trained and saved successfully!")

# Entry point
if __name__ == "__main__":
    train_cnn_model()
