import os
import shutil
import io
import torch
import torchvision.transforms as transforms
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from google.cloud import storage
from model import ResNet18

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


app = FastAPI(title="Image Classification API",
              description="A REST API for hotdog/not-hotdog image classification",
              lifespan=lifespan)


@app.get("/")
def root():
    response = {
        "message": "Image Classification API is running",
    }
    return response


@app.post("/predict")
async def predict(data: UploadFile = File(...)):
    # validate the file format
    if not data.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
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
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.sigmoid(outputs).squeeze().item()
    category = "hotdog" if probs > 0.5 else "not hotdog"
    return category, probs


# download trained model from GCP
def download_model_from_gcs_and_load():
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket("mlops-trained-models")
        blob = bucket.blob("models/model.pth")
        blob.download_to_filename("model.pth")

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
