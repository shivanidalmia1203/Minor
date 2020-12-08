from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import time
import os, random

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
classifier =load_model(r'Emotions.h5')

class_labels = ['Angry','Happy','Neutral','Sad','Surprise']

cap = cv2.VideoCapture(0)

print("Running")

l = []
countEmotion = [0,0,0,0,0]

t_end = time.time() + 15
while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
    # rect,face,image = face_detector(frame)


        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

        # make a prediction on the ROI, then lookup the class

            preds = classifier.predict(roi)[0]
            label=class_labels[preds.argmax()]
            l.append(label)
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        else:
            cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
    
    cv2.imshow('Emotion Detector',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):   
        break
    if time.time()>t_end:
        break

cap.release()
cv2.destroyAllWindows()

for i in range(len(l)):
            if l[i]=='Angry':
                countEmotion[0] += 1
            if l[i]=='Happy':
                countEmotion[1] += 1
            if l[i]=='Neutral':
                countEmotion[2] += 1
            if l[i]=='Sad':
                countEmotion[3] += 1
            if l[i]=='Surprise':
                countEmotion[4] += 1


emotion = max(countEmotion)
# print(emotion)
index1=0
for i in range(len(countEmotion)):
    if emotion == countEmotion[i]:
        index1=i
        break

finalEmotion = class_labels[index1]
print(countEmotion)
print(finalEmotion)

path =os.path.join(r'Songs',finalEmotion)
os.chdir(path)
for root, folders, files in os.walk(path):
    file = random.choice(files)
    os.startfile(file)


