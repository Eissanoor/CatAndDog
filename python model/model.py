import os
import numpy as np
import tensorflow as tf             
from tensorflow.keras import layers, models, optimizers # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array # type: ignore
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau # type: ignore
import matplotlib.pyplot as plt # type: ignore

# Set paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, 'data', 'train')
VALIDATION_DIR = os.path.join(BASE_DIR, 'data', 'validation')
MODEL_PATH = os.path.join(BASE_DIR, 'cat_dog_model1.1.h5')

# Parameters
IMG_WIDTH = 224  # Reduced from 500
IMG_HEIGHT = 224  # Reduced from 374
BATCH_SIZE = 16  # Reduced from 32
EPOCHS = 20
LEARNING_RATE = 1e-4
CONFIDENCE_THRESHOLD = 0.70  # Threshold for confident predictions

def create_data_generators():
    """Create and return training and validation data generators with augmentation"""
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,  # Reduced from 20
    width_shift_range=0.1,  # Reduced from 0.2
    height_shift_range=0.1,  # Reduced from 0.2
    shear_range=0.1,  # Reduced from 0.2
    zoom_range=0.1,  # Reduced from 0.2
    horizontal_flip=True,
    fill_mode='nearest'
)
    
    # Only rescaling for validation
    validation_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load and augment training data
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )
    
    # Load validation data
    validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )
    
    return train_generator, validation_generator

def build_model():
    """Build and return a transfer learning model based on MobileNetV2"""
    
    # Load pre-trained MobileNetV2 model without top layers
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
    )
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Build the model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')  # Binary classification (cat or dog)
    ])
    
    # Compile the model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, train_generator, validation_generator):
    """Train the model with callbacks for better performance"""
    
    # Define callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6
        )
    ]
    
    # Calculate steps per epoch
    train_steps = train_generator.samples // BATCH_SIZE
    validation_steps = validation_generator.samples // BATCH_SIZE
    
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_steps,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=callbacks
    )
    
    return history

def fine_tune_model(model, train_generator, validation_generator):
    """Fine-tune the model by unfreezing some layers of the base model"""
    
    # Unfreeze the top layers of the base model
    base_model = model.layers[0]
    base_model.trainable = True
    
    # Freeze all the layers except the top 30
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    # Recompile the model with a lower learning rate
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE/10),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Define callbacks for fine-tuning
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-7
        )
    ]
    
    # Calculate steps per epoch
    train_steps = train_generator.samples // BATCH_SIZE
    validation_steps = validation_generator.samples // BATCH_SIZE
    
    # Fine-tune the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_steps,
        epochs=10,  # Fewer epochs for fine-tuning
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=callbacks
    )
    
    return history

def plot_training_history(history):
    """Plot training and validation accuracy/loss"""
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(len(acc))
     
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plt.savefig(os.path.join(BASE_DIR, 'training_history.png'))
    plt.show()

def predict_image(image_path, model, confidence_threshold=CONFIDENCE_THRESHOLD):
    """
    Predict if an image contains a cat or a dog, with confidence threshold
    to identify images that are neither cats nor dogs.
    
    Args:
        image_path: Path to the image file
        model: Loaded model for prediction
        confidence_threshold: Minimum confidence required for a valid prediction
        
    Returns:
        Prediction result as a string: 'cat', 'dog', or 'neither cat nor dog'
    """
    # Load and preprocess the image
    img = load_img(image_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    # Make prediction
    prediction = model.predict(img_array)[0][0]
    
    # Check confidence level
    if prediction < (1 - confidence_threshold) or prediction > confidence_threshold:
        # Confident prediction
        if prediction > 0.5:
            return 'dog', prediction
        else:
            return 'cat', 1 - prediction
    else:
        # Not confident enough - likely neither cat nor dog
        return 'neither cat nor dog', max(prediction, 1 - prediction)

def main():
    # Create data generators
    train_generator, validation_generator = create_data_generators()
    
    # Build the model
    model = build_model()
    print("Model summary:")
    model.summary()
    
    # Train the model
    print("\nTraining the model...")
    history = train_model(model, train_generator, validation_generator)
    
    # Fine-tune the model
    print("\nFine-tuning the model...")
    fine_tune_history = fine_tune_model(model, train_generator, validation_generator)
    
    # Plot training history
    plot_training_history(fine_tune_history)
    
    # Save the final model
    model.save(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
