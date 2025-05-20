# Cat and Dog Image Classification Model

This project implements an efficient deep learning model to classify images of cats and dogs using transfer learning with MobileNetV2.

## Project Structure

```
python model/
│
├── data/
│   ├── train/
│   │   ├── cats/
│   │   └── dogs/
│   └── validation/
│       ├── cats/
│       └── dogs/
│
├── model.py         # Model training script
├── predict.py       # Image prediction script
├── requirements.txt # Dependencies
└── README.md        # This file
```

## Setup

1. Install the required dependencies:

```
pip install -r requirements.txt
```

## Training the Model

To train the model, run:

```
python model.py
```

This will:
- Load and preprocess the training and validation data
- Build a model using MobileNetV2 as the base
- Train the model with data augmentation
- Fine-tune the model by unfreezing some layers
- Save the trained model as `cat_dog_model.h5`
- Generate a training history plot

## Making Predictions

To classify a new image, run:

```
python predict.py path/to/your/image.jpg
```

This will:
- Load the trained model
- Preprocess the input image
- Make a prediction (cat or dog)
- Display the image with the prediction result

To only get the prediction without displaying the image:

```
python predict.py path/to/your/image.jpg --no-display
```

## Model Architecture

The model uses transfer learning with MobileNetV2 (pretrained on ImageNet) for efficient and accurate classification:

1. MobileNetV2 base (without top layers)
2. Global Average Pooling
3. Dropout (0.5)
4. Dense layer (128 neurons, ReLU activation)
5. Dropout (0.3)
6. Output layer (1 neuron, sigmoid activation)

## Performance Optimization

The model includes several optimizations:
- Data augmentation to prevent overfitting
- Early stopping to prevent overfitting
- Learning rate reduction when validation loss plateaus
- Model checkpointing to save the best model
- Two-stage training (transfer learning followed by fine-tuning)
