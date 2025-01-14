from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torchvision.transforms as transforms
from google.cloud import storage
import io
from model import ResNet18

# 全局模型变量
model = None

# 定义 lifespan 上下文管理器
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时加载模型
    global model
    try:
        model = download_model_from_gcs()
        yield
    finally:
        # 清理资源
        del model

app = FastAPI(
    title="Image Classification API",
    description="A REST API for hotdog/not-hotdog image classification",
    lifespan=lifespan
)

# 从 GCP 下载模型
def download_model_from_gcs():
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket("mlops-trained-models")
        blob = bucket.blob('models/model.pth')
        
        blob.download_to_filename('model.pth')
        
        model = ResNet18()
        model.load_state_dict(torch.load('model.pth'))
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")

# 图像预处理
def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),            # Resize image to 128x128
        transforms.ToTensor(),                   # Convert to tensor (CxHxW format)
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])
    
    image_tensor = transform(image)
    return image_tensor.unsqueeze(0)

@app.get("/")
async def root():
    """API 根路径"""
    return {"message": "Image Classification API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    预测图片分类
    
    - **file**: 上传的图片文件（.jpg, .jpeg, .png）
    
    返回:
    - probabilities: 预测概率
    - category: 预测类别 (hotdog/not hotdog)
    """
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Only JPG and PNG images are supported."
        )
    
    try:
        # 确保模型已加载
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="Model is not loaded. Please try again later."
            )
        
        '''
        Read the image
        '''
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        

        '''
        process the image and make predictions
        '''
        input_tensor = preprocess_image(image) 
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.sigmoid(outputs).squeeze().item()
        category = "hotdog" if probs > 0.5 else "not hotdog"
        
        return JSONResponse({
            'probabilities': float(probs),
            'category': category
        })
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """健康检查端点"""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    return {"status": "healthy", "model_loaded": True}

