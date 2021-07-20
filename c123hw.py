import pandas as pd
import numpy as np
import seaborn as sb
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from PIL import Image
import PIL.ImageOps 
import os
import time , ssl


X = np.load('image.npz')['arr_0']
y = pd.read_csv('labels.csv')['labels']

print(pd.Series(y).value_counts())

classes = ['A', 'B', 'C', 'D', 'E','F', 'G', 'H', 'I', 'J', "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

nclasses = len(classes)

X_train , X_test , y_train , y_test = train_test_split(X,y,random_state = 9 , train_size = 7500 , test_size = 2500) 

X_train_scale = X_train / 255

X_test_scale = X_test / 255

lr = LogisticRegression(solver = 'saga',multi_class= 'multinomial')

lr.fit(X_train_scale , y_train )

prediction = lr.predict(X_test_scale)

accuracy = accuracy_score(prediction , y_test)

print('Accuracy:',accuracy)


capture = cv2.VideoCapture(0)

while True:
    try:
        ret , frame = capture.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        height , width = gray.shape()
        upper_left = (int(width/2 - 56),int(height/2 - 56))
        bottom_right = (int(width/2 + 56),int(height/2 + 56))
        cv2.rectangle(gray , upper_left , bottom_right,(0,255,0),2)
        roi = gray[upper_left[1]:bottom_right[1],upper_left[0]:bottom_right[0]]
        im_pil = Image.fromarray(roi)

        # L format means each pixel is represented by a single value from 0 to 255
        image_bw = im_pil.convert('L')

        image_bw_resized = image_bw.resize((28,28),Image.ANTIALIAS)
        image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized)
        pixel_filter = 20
        min_pixel = np.percentile(image_bw_resized_inverted,pixel_filter)
        image_bw_resized_inverted_scaled = np.clip(image_bw_resized_inverted-min_pixel,0,255)
        max_pixel = np.max(image_bw_resized_inverted)
        image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel
        
        test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784)

        test_prediction = lr.predict(test_sample)

        print('PREDICTED CLASS IS : ',test_prediction)

        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        pass

capture.release()
cv2.destroyAllWindows()