from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
import joblib
import numpy as np
from starlette.responses import JSONResponse
from api.middleware import register_middleware

try:
    model = joblib.load('model/best_model.pkl')
    scaler = joblib.load('model/scaler.pkl')
except Exception as e:
    raise RuntimeError(f"Không thể load model hoặc scaler: {e}")

app = FastAPI()

register_middleware(app)

class AbaloneInput(BaseModel):
    Sex: str
    Length: float
    Diameter: float
    Height: float
    Whole_weight: float
    Shucked_weight: float
    Viscera_weight: float
    Shell_weight: float


def encode_sex(sex: str) -> list:
    sex = sex.upper()
    if sex not in ['M', 'F', 'I']:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Giá trị Sex không hợp lệ. Chỉ nhận 'M', 'F' hoặc 'I'.")
    return [
        1 if sex == 'F' else 0,
        1 if sex == 'I' else 0,
        1 if sex == 'M' else 0
    ]


@app.post("/")
def predict_abalone_age(data: AbaloneInput):
    sex_encoded = encode_sex(data.Sex)
    features = [
                   data.Length,
                   data.Diameter,
                   data.Height,
                   data.Whole_weight,
                   data.Shucked_weight,
                   data.Viscera_weight,
                   data.Shell_weight
               ] + sex_encoded

    try:
        X = np.array(features).reshape(1, -1)
        X_scaled = scaler.transform(X)
        rings_pred = model.predict(X_scaled)[0]
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Lỗi khi dự đoán: {str(e)}")

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "message": "Dự đoán thành công!",
            "content": {
                "predicted_rings": round(rings_pred, 2),
                "estimated_age": round(rings_pred + 1.5, 2)
            }
        }
    )
