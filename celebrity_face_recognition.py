
# coding: utf-8

# # Face Recognition for the Indian Celebrity Using Facenet
# 
# 
# FaceNet learns a neural network that encodes a face image into a vector of 128 numbers. By comparing two such vectors, you can then determine if two pictures are of the same person.
#     
# 
# We will be using a pre-trained mode. The project is inspired from coursera course assignment.
# 
# Let's load the required packages. 
# 

# In[2]:


from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

np.set_printoptions(threshold=np.nan)


# ## Load the pretrained model

# In[ ]:


model = faceRecoModel(input_shape=(3, 96, 96))


# ## Compile the model
# 
# Since we are just predicting the model's output so there is no need of loss function and accuracy which is required while evaluating the model. More about this can be found [here](https://stackoverflow.com/questions/46127625/need-to-compile-keras-model-before-model-evaluate)

# In[ ]:


load_weights_from_FaceNet(model)


# In[ ]:


print("Total Params:", model.count_params())


# ### 3.2 - Face Recognition
# 
# 1. Compute the target encoding of the image from image_path
# 2. Find the encoding from the database that has smallest distance with the target encoding. 
#     - Initialize the `min_dist` variable to a large enough number (100). It will help you keep track of what is the closest encoding to the input's encoding.
#     - Loop over the database dictionary's names and encodings.
#         - Compute L2 distance between the target "encoding" and the current "encoding" from the database.
#         - If this distance is less than the min_dist, then set min_dist to dist, and identity to name.

# In[ ]:


def recognize_face(face_descriptor, database):
    encoding = img_to_encoding(face_descriptor, model)
    min_dist = 100
    identity = None

    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():

        # Compute L2 distance between the target "encoding" and the current "emb" from the database.
        dist = np.linalg.norm(db_enc - encoding)

        print('distance for %s is %s' % (name, dist))

            #   If this distance is less than the min_dist, then set min_dist to dist, and identity to name
            #   less than equals to 80 Jahnavi Kapoor,less than equals to 154 Hritick,
            #   less than equals to 235 Alia Bhatt, less than equals to 317 Salman Khan,
            #   less than equals to 425 Shahrukh Khan, less than equals to 439 Aamir Khan,
        if dist < min_dist:
            min_dist = dist
            identity = name
    
    if int(identity) <= 80 and min_dist <=0.63:
        return ("Jahnavi Kapoor"), min_dist 
    if int(identity) > 80 and int(identity) <= 154 and min_dist <=0.63:
        return ("Hritick Roushan"), min_dist
    if int(identity) > 154 and int(identity) <= 235 and min_dist <=0.63:
        return ("Alia Bhatt"), min_dist
    if int(identity) > 235 and int(identity) <= 317 and min_dist <=0.63:
        return ("Salman Khan"), min_dist
    if int(identity) > 317 and int(identity) <= 425 and min_dist <=0.63:
        return ("Shahrukh Khan"), min_dist
    if int(identity) > 425 and int(identity) <= 439 and min_dist <=0.63:
        return ("Aamir Khan"), min_dist
       


# In[ ]:


import cv2
import numpy as np
import dlib
import glob
from scipy.spatial import distance
from imutils import face_utils
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# In[ ]:


detector = dlib.get_frontal_face_detector()


# In[ ]:


def extract_face_info(img, img_rgb, database):
    faces = detector(img_rgb)
    x, y, w, h = 0, 0, 0, 0
    if len(faces) > 0:
        for face in faces:
            (x, y, w, h) = face_utils.rect_to_bb(face)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
            image = img[y:y + h, x:x + w]
            if(image.size == 0):
                continue
            name, min_dist = recognize_face(image, database)
            
            if min_dist < 1.8:
                cv2.putText(img, "Face : " + name, (x, y - 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
                cv2.putText(img, "Dist : " + str(min_dist), (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, ( 255,0, 0), 2)
            else:
                cv2.putText(img,  name, (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)


# In[ ]:


def img_path_to_encoding(image_path, model):
    img1 = cv2.imread(image_path, 1)
    return img_to_encoding(img1, model)
    

def img_to_encoding(image, model):
    image = cv2.resize(image, (96,96))
    img = image[...,::-1]
    img = np.around(np.transpose(img, (2,0,1))/255.0, decimals=12)
    x_train = np.array([img])
    embedding = model.predict(x_train)
    return embedding


# In[ ]:


def initialize():
    database = {}

    # load all the images of individuals to recognize into the database
    for file in glob.glob("images/*"):
        identity = os.path.splitext(os.path.basename(file))[0]
        database[identity] = img_path_to_encoding(file, model)
    return database


# In[ ]:


def recognize():
    database = initialize()
    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if(img.size == 0 or img_rgb.size == 0):
            continue
                
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        subjects = detector(gray, 0)
        for subject in subjects:
            extract_face_info(img, img_rgb, database)
        
        cv2.imshow('Recognizing faces', img)
        if cv2.waitKey(100) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# In[ ]:


recognize()

