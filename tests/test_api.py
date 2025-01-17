from fastapi.testclient import TestClient
from src.mlops_project.api import app


def test_upload_and_predict():
    with TestClient(app) as client:
        # We hardcoded the image file since it won't really affect the test result
        with open("data\\test\\nothotdog\\food (1).jpg", "rb") as file:
            response = client.post("/predict", files={"data": file})
        assert response.status_code == 200, response.text
        assert "category" in response.json()
        assert "probabilities" in response.json()
