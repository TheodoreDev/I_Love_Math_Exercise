"""
This module is the main flask application.
"""

from flask import Flask, request, render_template
from blueprints import *
import cv2
import numpy as np
import tensorflow as tf
import re
import mahotas
import base64
import imutils

app = Flask(__name__)
app.secret_key = b'A Super Secret Key'


# YOUR PART: change to your new model h5 and add more label_names like how you train in your notebook
model = tf.keras.models.load_model('static/first_half_mul.h5')
label_names = ['0', '1', '2', '3', '4', '5', '10']

app.register_blueprint(home_page)

def parse_image(imgData):
    imgstr = re.search(b"base64,(.*)", imgData).group(1)
    img_decode = base64.decodebytes(imgstr)
    with open("output.jpg", "wb") as file:
        file.write(img_decode)
    return img_decode

# YOUR PART: deskew function

# YOUR PART: center_extent function

@app.route("/upload/", methods=["POST"])
def upload_file():
    img_raw = parse_image(request.get_data())
    nparr = np.fromstring(img_raw, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # YOUR PART: convert to gray

    gray = 

    # YOUR PART: blur with GaussianBlur (play around with parameters)
    blurred = 

    # YOUR PART: edge with adaptiveThreshold (play around with parameters)
    edged =

    # YOUR PART: find contours, remember to input with edged.copy()
    _, cnts, _ =    

    # sort based on (x,y) position
    cnts = sorted([(c, cv2.boundingRect(c)[0]) for c in  cnts], key=lambda x: x[1])

    math_detect = []

    for (c, _) in cnts:

        # find rectangle box of a coutour
        (x, y, w, h) = 
        if w >=5 and h>5:

            # make sure we cut it a bit longer in height
            roi = edged[y:y+int(1.2*h), x:x+w]
            thresh = roi.copy()

            # YOUR PART: deskew now with size 28
            thresh = 

            # YOUR PART: center_extent now with size (28, 28)
            thresh = 

            # to be sure, reshape to correct dimension 
            thresh = np.reshape(thresh, (28, 28, 1))

            # standardization by divide it by 255
            thresh
            
            # make prediction 
            predictions = model.predict(np.expand_dims(thresh, axis=0))
            digit = np.argmax(predictions[0])

            # draw rectangle on image (for debugging purpose - optional)
            cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)

            # draw text on image (for debugging purpose - optional)
            cv2.putText(image, label_names[digit], (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)


            # LATER: when we want to add other operations like division, we need to add more condition below (some hacks)
            math_detect.append(label_names[digit])

    def convert_math(math_detect):
        for i in range(0, len(math_detect)):

            if math_detect[i] == '10':
                math_detect[i] = '*'
            # YOUR PART: add more condition for 11 and 12 for minus and plus
           
        return math_detect


    def calculate_string(math_detect):
        math_detect = convert_math(math_detect)
        calculator = ''.join(str(item) for item in math_detect)
        result = calculator
        return result

    result = calculate_string(math_detect)

    return result


@app.route("/calcu/", methods=["POST"])
def calcu():
    val = request.get_data()
    val = str(request.get_data())
    val1=val[2:-1]

    # eval allows use to calculate the string of math "4+5", "6*4+2", "4/6+52"
    result = str(eval(val1))
    return result



if __name__ == '__main__':
    app.run(debug=True)