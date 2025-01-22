# **API Documentation**

## Overview
This API provides functionality for classifying images as `hotdog` or `not-hotdog`. The service loads a pre-trained model and performs predictions based on uploaded images.

**Request URL:** `https://api-service-748263339953.europe-west1.run.app/`

**Interactive URL** `https://api-service-748263339953.europe-west1.run.app/docs`

## Endpoints

### 1. Root Endpoint
**Description:** Returns a basic message to confirm the API is running.

- **Method:** `GET`
- **URL:** `/`

**Example Request:**
```bash
curl -X 'GET' \
  'https://api-service-748263339953.europe-west1.run.app/' \
  -H 'accept: application/json'
```
**Response:**
```json
{
    "message": "Image Classification API is running"
}
```

---

### 2. Prediction Endpoint
**Description:** Classifies an uploaded image as either `hotdog` or `not-hotdog`.

- **Method:** `POST`
- **URL:** `/predict`
- **Headers:**
  - `Content-Type: multipart/form-data`
- **Request Body:**
  - `file` (required): The image file to classify. Must be `.jpg`, `.jpeg`, or `.png`.

**Example Request:**
```bash
curl -X 'POST' \
  'https://api-service-748263339953.europe-west1.run.app/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'data=@your_image.jpg;type=image/jpeg'
```

**Response:**
```json
{
    "category": "hotdog",
    "probabilities": 0.95
}
```
- `category`: The classification result (`hotdog` or `not-hotdog`).
- `probabilities`: The confidence score for the prediction.

**Error Responses:**
- Invalid file format:
  ```json
  {
      "detail": "Invalid file format. Only JPG and PNG images are supported."
  }
  ```
- Model not loaded:
  ```json
  {
      "detail": "Model is not loaded. Please try again later."
  }
  ```
- Internal server error:
  ```json
  {
      "detail": "Error processing image: <error_message>"
  }
  ```

---

### 3. Health Check Endpoint
**Description:** Checks the health status of the API and confirms if the model is loaded.

- **Method:** `GET`
- **URL:** `/health`

**Example Request**
```bash
curl -X 'GET' \
  'https://api-service-748263339953.europe-west1.run.app/health' \
  -H 'accept: application/json'
```

**Response:**
```json
{
    "status": "healthy",
    "model_loaded": true
}
```

**Error Response:**
Model not loaded:
```json
{
    "detail": "Model not loaded"
}
```
