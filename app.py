from fastapi import FastAPI, Request, Form, Depends
from fastapi.templating import Jinja2Templates
import pickle
import numpy as np
from pydantic import BaseModel
import uvicorn



# Load the model from the saved file

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

app = FastAPI()
templates = Jinja2Templates(directory="templates/")

# Pydantic model for input validation
class FurnitureInput(BaseModel):
    category: str
    sellable_online: str
    other_colors: str
    depth: float
    height: float
    width: float

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(request: Request, input_data: FurnitureInput = Depends()):

        # Extract features from validated input data
        features = np.array([
            input_data.category,
            input_data.sellable_online,
            input_data.other_colors,
            input_data.depth,
            input_data.height,
            input_data.width
        ]).reshape(1, -1)

        # Make a prediction using the loaded model
        prediction = model.predict(features)[0]

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
