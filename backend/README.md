# Cat and Dog Classification API

A simple API that uses a pre-trained TensorFlow model to classify images as either cats or dogs.

## Prerequisites

- Node.js (v14+)
- Python (v3.6+)
- The following Python packages:
  - TensorFlow
  - NumPy
  - Pillow

## Installation

1. Install Node.js dependencies:

```bash
npm install
```

2. Install required Python packages:

```bash
pip install tensorflow numpy pillow
```

## Usage

1. Start the server:

```bash
npm start
```

2. Send a POST request to the `/classify` endpoint with an image file in the `image` field:

```bash
curl -X POST -F "image=@/path/to/your/image.jpg" http://localhost:4000/classify
```

3. The response will be a JSON object with the classification result:

```json
{
  "class": "cat",
  "confidence": 0.9876
}
```

## API Endpoints

- `POST /classify` - Upload an image for classification
  - Request: Multipart form data with an image file in the "image" field
  - Response: JSON with classification result

## Notes

- Default model path is set to `E:\AI\cat and dog\python model\cat_dog_model.h5`
- You can specify a custom model path in three ways:
  1. Through the `MODEL_PATH` environment variable: `MODEL_PATH=/path/to/model.h5 npm start`
  2. In the request body when calling the API: send `modelPath` field along with your image
  3. By changing the default path in the server code
- Supported image formats: JPEG, PNG, etc.
- Maximum file size: 10MB 