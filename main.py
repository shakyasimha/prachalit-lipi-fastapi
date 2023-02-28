from fastapi import FastAPI, UploadFile
import uvicorn
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2

# Some constants to be used in the program
IMG_SIZE = (32,32)
APP_HOST = '0.0.0.0'
APP_PORT = '5000'

# Character mapping for the character prediction
char_map = {
    0:'𑑐(0)',   1:'𑑑(1)',    2:'𑑒(2)',   3:'𑑓(3)',      4: '𑑔(4)',     5: '𑑕(5)',    6: '𑑖(6)',    7: '𑑗(7)',
    8:'𑑘(8)',   9:'𑑙(9)',    10:'𑑉(OM)', 11:'𑐀(A)',    12: '𑐁(AA)',   13: '𑐀𑑅(AH)',  14: '𑐂(I)',    
    15:'𑐃(II)',16:'𑐄(U)',   17:'𑐅(UU)',  18:'𑐆(R)',    19: '𑐆𑐺(RR)',  20: '𑐊(E)',   21: '𑐋(AI)',    22: '𑐌(O)',    
    23:'𑐍(AU)', 24:'𑐈(L)',  25:'𑐉(LL)',   26:'𑐎(KA)',   27: '𑐎𑑂𑐳(KSA)', 28: '𑐏(KHA)',29: '𑐐(GA)',    30: '𑐑(GHA)',    
    31:'𑐒(NGA)',32:'𑐔(CA)',  33:'𑐕(CHA)', 34:'𑐖(JA)',   35: '𑐖𑑂𑐘(JñA)',  36: '𑐗(JHA)',37: '𑐗(JHA-alt)',38: '𑐘(NYA)',    
    39:'𑐚(TA)', 40:'𑐛(TTHA)', 41:'𑐜(DDA)', 42:'𑐝(DHA)',  43: '𑐞(NNA)', 44: '𑐟(TA)',  45: '𑐟𑑂𑐬(TRA)',    46: '𑐠(THA)',
    47:'𑐡(DA)', 49:'𑐣(NA)',   50:'𑐥(PA)',  51:'𑐦(PHA)',  52: '𑐧(BA)',  53: '𑐨(BHA)',  54: '𑐩(MA)',    55: '𑐫(YA)', 
    56:'𑐬(RA)', 57: '𑐮(LA)', 58:'𑐰(WA)', 59:'𑐱(SHA)',    60: '𑐱(SHA-alt)', 61: '𑐲(SSA)',    62: '𑐳(SA)', 63: '𑐴(HA)'
}



# Importing the model
model = keras.saved_model.load('vgg16/')


# Function for segmenting the image 
# def segment_image(image):


# Function for returning the prediction of image 
def predict_image(image):
    # Preprocessing part goes here
    image = cv2.imread(image)
    image = cv2.resize(image, IMG_SIZE)
    # image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)

    output = model.predict(image)
    predicted_class = np.argmax(output)

    return char_map[predicted_class]

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

# if __name__ == "__main__":
    # uvicorn.run(app, host=APP_HOST, port=APP_PORT)