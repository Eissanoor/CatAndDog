# Cat, Dog, and Other Image Classifier

This project provides a scalable image classification system that can identify cats, dogs, and other objects.

## Setup

1. Install requirements:
```
pip install tensorflow numpy matplotlib pillow
```

2. Create the dataset structure:
```
dataset/
├── train/
│   ├── cat/     (put cat images here)
│   ├── dog/     (put dog images here)
│   └── other/   (put non-cat, non-dog images here)
└── validation/
    ├── cat/     (put validation cat images here)
    ├── dog/     (put validation dog images here)
    └── other/   (put validation non-cat, non-dog images here)
```

## Collecting "Other" Images

We've included a helpful tool to easily collect and organize images for the "other" category:

```
python backend/collect_other_images.py
```

This will launch a GUI application that allows you to:
1. Select multiple images
2. Preview each image
3. Add it to either the training or validation "other" category
4. Skip images you don't want to include

Use this tool to collect logos, screenshots, landscapes, and any other images that aren't cats or dogs.

## Training the Model

1. Collect images for your dataset:
   - For the "cat" and "dog" categories, use clear images of cats and dogs
   - For the "other" category, include a variety of images including logos, objects, landscapes, etc.

2. Run the training script:
```
python backend/train_model.py
```

This will train a 3-class model and save it as `cat_dog_other_model.h5` in the backend directory.

## Making Predictions

To classify an image:
```
python backend/predict.py
```

This will open a file dialog for you to select an image for classification.

## Tips for Better Results

1. Include at least 500 images in each category for training
2. For the "other" category, include a diverse set of images
3. Include images similar to what you'll be classifying (logos, etc.) in the "other" category
4. If you're getting incorrect classifications, add more examples of that type to the training data

## Fine-tuning

If you need to adjust the model:
- Change the confidence threshold in `predict.py`
- Increase the number of training epochs in `train_model.py`
- Add more diverse training data 