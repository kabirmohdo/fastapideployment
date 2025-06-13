from fastapi import FastAPI , Body
from pydantic import BaseModel , Field
import pandas as pd 
import joblib 

app = FastAPI()

model = joblib.load('iris.pkl')

@app.get('/get')
def welcome():
    return {'message': 'Welcome to the ml model API!'}


class Item(BaseModel):
    sepal_length : float
    sepal_width : float
    petal_length : float
    petal_width : float


@app.post('/predict')
async def create_data(data : Item):
    new_data = {
        'sepal length (cm)': data.sepal_length,
        'sepal width (cm)': data.sepal_width,
        'petal length (cm)': data.petal_length,
        'petal width (cm)': data.petal_width
    }

    df = pd.DataFrame([new_data])
    prediction = model.predict(df)
    label = ['Setosa', 'Versicolor', 'Virginica']

    return {
        'Model prediction' : label[prediction[0]]
    }








