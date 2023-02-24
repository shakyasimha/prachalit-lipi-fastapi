from fastapi import FastAPI, UploadFile
import uvicorn
import tensorflow as tf
from tensorflow import keras


# Some constants to be used in the program
IMG_SIZE = (32,32)
APP_HOST = '0.0.0.0'
APP_PORT = '5000'




# Importing the model
model = keras.saved_model.load('vgg16-prachalit/')





# Function for predicting
def predict(image_tensor):
    # Some prediction code goes here
    output = model(image_tensor)
    predictions = output.numpy().tolist()

    return predictions




# Functions for preprocessing the image
def preproc_img(image):
    input_tensor = tf.image.decode_image(image)
    input_tensor = tf.image.convert_image_dtype(input_tensor, tf.float32)
    input_tensor = tf.image.resize(input_tensor, IMG_SIZE)
    input_tensor = tf.expand_dims(input_tensor, 0)

    return input_tensor




# Function for returning the prediction of image 
def predict_image(image):
    image_tensor = preproc_img(image)
    result = predict(image_tensor)
    return max(result)



# Defining the FastAPI instance here
app = FastAPI()

@app.root('/')
async def root_func():
    return {'message': 'this is the root function'}

@app.post('/predict_image')
async def upload_image(image: UploadFile):
    try:
        result = predict_image(await image.read())
    except Exception as e:
        print(e) 
        result = "null"

    return {'prediction': result}


if __name__ == "__main__":
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)