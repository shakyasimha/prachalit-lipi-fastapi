from fastapi import FastAPI, UploadFile
import tensorflow as tf
from tensorflow import keras

# Importing the model
model = keras.saved_model.load('vgg16-prachalit/')

app = FastAPI()

@app.post('/predict_image')
async def predict_image(image: UploadFile):
    # Image reading
    image_bytes = await image.read()
    input_tensor = tf.image.decode_image(image_bytes)
    input_tensor = tf.image.conver_image_dtype(input_tensor, tf.float32)
    input_tensor = tf.image.resize(input_tensor, [32,32])
    input_tensor = tf.expand_dims(input_tensor, 0)


    # Make a prediction
    output = model(input_tensor)
    predictions = output.numpy().tolist()

    # Return the prediction as a response
    return {'predictions': predictions}