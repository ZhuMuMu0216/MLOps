import os
import shutil
import io
import torch
import torchvision.transforms as transforms
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from PIL import Image
from google.cloud import storage
from model import ResNet18
import numpy as np
import datetime
import json

model = None
tempfile_dir = "temp"


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Application starting...")
    print("create a temp directory")
    if not os.path.exists(tempfile_dir):
        os.makedirs(tempfile_dir)
    global model
    try:
        print("download model from gcs")
        model = download_model_from_gcs_and_load()
        yield
    finally:
        print("Application closing...")
        print("clean up temp resources")
        if os.path.exists(tempfile_dir):
            shutil.rmtree(tempfile_dir)
            print(f"{tempfile_dir} and all its contents have been removed.")
        print("release model")
        del model


app = FastAPI(
    title="Image Classification API",
    description="A REST API for hotdog/not-hotdog image classification",
    lifespan=lifespan,
)


@app.get("/")
def root():
    response = {
        "message": "Image Classification API is running",
    }
    return response


@app.post("/predict")
async def predict(data: UploadFile = File(...)):
    # validate the file format
    if not data.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="Invalid file format. Only JPG and PNG images are supported.")

    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Please try again later.")

    # Read the image
    image_bytes = await data.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    try:
        # process the image and make predictions
        category, probs = process_image(image)
        return JSONResponse({"category": category, "probabilities": float(probs)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.get("/health")
async def health_check():
    """健康检查端点"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}


def process_image(image):
    input_tensor = preprocess_image(image)
    input_features = extract_features(image)

    with torch.no_grad():
        outputs = model(input_tensor)
        
        probs = torch.sigmoid(outputs).squeeze().item()
    category = "hotdog" if probs < 0.5 else "not hotdog"
    save_prediction_to_gcp(input_features, 1-probs, category)

    return category, 1-probs

def extract_features(img):
    """
    Extract basic image features from a single image.
    """
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),  # Resize image to 128x128
            transforms.ToTensor(),  # Convert to tensor (CxHxW format)
        ]
    )
    image_tensor = transform(img)
    input_numpy = image_tensor.numpy()  # Convert to NumPy
    avg_brightness = np.mean(input_numpy)  # Compute the average pixel intensity (brightness)
    contrast = np.std(input_numpy)  # Compute the standard deviation of pixel values (contrast)
    sharpness = np.mean(np.abs(np.gradient(input_numpy)))  # Compute the mean gradient magnitude (sharpness)
    return np.array([avg_brightness, contrast, sharpness])  # Return features as a NumPy array

# Save prediction results to GCP
def save_prediction_to_gcp(input_features, outputs, category: str):
    """Save the prediction results to GCP bucket."""
    client = storage.Client()
    bucket = client.bucket("mlops-trained-models")
    time = datetime.datetime.now(tz=datetime.UTC)

    # Prepare prediction data
    data = {
        "avg_brightness": input_features[0],
        "contrast": input_features[1],
        "sharpness": input_features[2],
        "category": category,
        "probability": outputs,
        "timestamp": datetime.datetime.now(tz=datetime.UTC).isoformat(),
    }
    blob = bucket.blob(f"prediction_{time}.json")
    blob.upload_from_string(json.dumps(data))
    print("Prediction saved to GCP bucket.")

# download trained model from GCP
def download_model_from_gcs_and_load():
    try:
        # download trained model from GCS
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        key_file = os.path.join(project_root, "keys/cloud_storage_key.json")
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_file
        storage_client = storage.Client()
        bucket = storage_client.bucket("mlops-trained-models")
        blob = bucket.blob("models/model.pth")
        blob.download_to_filename("model.pth")

        # load the trained model
        model = ResNet18()
        model.load_state_dict(torch.load("model.pth"))
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")


# image preprocessing
def preprocess_image(image: Image.Image):
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),  # Resize image to 128x128
            transforms.ToTensor(),  # Convert to tensor (CxHxW format)
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize to [-1, 1]
        ]
    )

    image_tensor = transform(image)
    return image_tensor.unsqueeze(0)

