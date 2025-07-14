from fastapi import FastAPI , Body , HTTPException
from pydantic import BaseModel , Field
import pandas as pd 
import joblib 

app = FastAPI()

model = joblib.load('iris.pkl')

@app.get('/get')
def welcome():
    return {'message': 'Welcome to the ml model API!'}


class Item(BaseModel):
    sepal_length : float = Field(lt=3)
    sepal_width : float
    petal_length : float
    petal_width : float


@app.post('/predict' , status_code=201)
async def create_data(data : Item):
    new_data = {
        'sepal length (cm)': data.sepal_length,
        'sepal width (cm)': data.sepal_width,
        'petal length (cm)': data.petal_length,
        'petal width (cm)': data.petal_width
    }

    if new_data['sepal length (cm)']  > 3 :
        raise HTTPException(status_code=400, detail="Sepal length must be less than 3 cm")

    

    df = pd.DataFrame([new_data])
    prediction = model.predict(df)
    label = ['Setosa', 'Versicolor', 'Virginica']

    return {
        'Model prediction' : label[prediction[0]]
    }








