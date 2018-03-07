import sys
sys.setdefaultencoding('utf-8')

import os
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

from flask import Flask, request, jsonify
from flask_cors import CORS
import urllib.request
import os
import numpy as np
from keras.preprocessing import image
from keras.models import load_model

app= Flask(__name__)
CORS(app)

if __name__ == "__main__":
    app.run(host='0.0.0.0')

# 64 x 64 because thats the size of the inputs
test_image = image.load_img('cat_or_dog_1.jpg', target_size = (64, 64))
# From image to array. This array will include 3, the number of channels
test_image = image.img_to_array(test_image)
# A 4th dimension is added to account for the batch size
test_image = np.expand_dims(test_image, axis = 0)
#Overwritting the classifier to use the saved one
classifier = load_model('gee_that_a_dog.h5')
result = classifier.predict(test_image)
#training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print ("You may now begin your jolly computations")
    
# How to know the probability for dog or cat?

@app.route('/categorize', methods=['POST'])
def categorize():
    # Parsing the picture URL from the JSON
    #url = request.get_json()["URL"]
    url = request.form['text']
    # Downloading the picture
    urllib.request.urlretrieve(url, "animal.jpg")
    # Transforming the picture
    test_image = image.load_img('animal.jpg', target_size = (64, 64))
    # From image to array. This array will include 3, the number of channels
    test_image = image.img_to_array(test_image)
    # A 4th dimension is added to account for the batch size
    test_image = np.expand_dims(test_image, axis = 0)
    # Feeding the picture into the CNN
    result = classifier.predict(test_image)
    
    
    if result[0][0] == 1: prediction = 'dog'
    else: prediction = 'cat'
    # Deletes the picture
    os.remove("animal.jpg")
      
    return jsonify({'result': prediction})
    
@app.route('/test', methods=['GET'])
def test():
    return "1"
