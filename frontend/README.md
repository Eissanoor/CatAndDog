# Cat vs Dog Classifier

A React application that classifies images as either a cat or a dog using a backend API.

## Features

- Upload an image through an intuitive UI
- Preview the selected image
- Get classification results with confidence scores
- Modern UI using daisyUI and TailwindCSS

## Prerequisites

- Node.js (v14 or newer)
- npm or yarn
- Backend server running on localhost:4000

## Installation

1. Install dependencies:

```bash
npm install
```

2. Run the development server:

```bash
npm run dev
```

3. Open your browser and navigate to `http://localhost:5173`

## Usage

1. Click on the "Select a cat or dog image" input to choose an image
2. Preview your selected image
3. Click the "Classify Image" button to get results
4. View the classification result and confidence score

## API Endpoint

The application uses the following endpoint:

- `POST http://localhost:4000/classify`
  - Request: Form data with a file field containing the image
  - Response: JSON with class and confidence properties

Example response:
```json
{
  "class": "dog",
  "confidence": 0.9999465346336365
}
```

## Technology Stack

- React (with TypeScript)
- Vite
- TailwindCSS
- daisyUI
