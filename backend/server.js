const express = require('express');
const multer = require('multer');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');
const cors = require('cors');
const app = express();
const port = 4000;

app.use(cors());
// Configure storage for uploaded files
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = path.join(__dirname, 'uploads');
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    cb(null, `${Date.now()}-${file.originalname}`);
  }
});

// Create multer instance with file size limits
const upload = multer({ 
  storage: storage,
  limits: { fileSize: 10 * 1024 * 1024 }, // 10MB limit
  fileFilter: (req, file, cb) => {
    if (file.mimetype.startsWith('image/')) {
      cb(null, true);
    } else {
      cb(new Error('Only image files are allowed!'), false);
    }
  }
});

// Endpoint for image classification
app.post('/classify', upload.single('image'), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No image uploaded' });
  }

  const imagePath = req.file.path;
  console.log(imagePath);
  
  // Use model from same directory as server.js
  const modelPath = path.join(__dirname, 'cat_dog_model.h5');
  
  console.log(`Using model at: ${modelPath}`);

  // Call Python script for prediction
  const pythonProcess = spawn('python', [
    path.join(__dirname, 'predict.py'),
    imagePath,
    modelPath
  ]);

  let result = '';
  let errorMsg = '';

  pythonProcess.stdout.on('data', (data) => {
    result += data.toString();
  });

  pythonProcess.stderr.on('data', (data) => {
    errorMsg += data.toString();
  });

  pythonProcess.on('close', (code) => {
    // Clean up the uploaded file
    fs.unlinkSync(imagePath);
    
    if (code !== 0) {
      console.error('Python process error:', errorMsg);
      return res.status(500).json({ error: 'Classification failed', details: errorMsg });
    }

    try {
      // Strip any non-JSON output and parse only the valid JSON part
      const jsonStr = result.trim();
      console.log('Raw output from Python script:', jsonStr);
      
      const prediction = JSON.parse(jsonStr);
      res.json(prediction);
    } catch (error) {
      console.error('JSON parse error:', error);
      console.error('Raw output was:', result);
      
      res.status(500).json({ 
        error: 'Failed to parse prediction result',
        details: error.message,
        rawOutput: result.trim()
      });
    }
  });
});

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});
