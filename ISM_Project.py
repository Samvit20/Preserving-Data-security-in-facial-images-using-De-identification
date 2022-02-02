import numpy as np
import cv2
import argparse
import dlib
import imutils
import math
from collections import OrderedDict
print("\nModules Imported\n")

facial_features_coordinates = {}

FACIAL_LANDMARK_INDEXES=OrderedDict([
    ("Mouth",(48,68)),
    ("Right_Eyebrow",(17,22)),
    ("Left_Eyebrow",(22,27)),
    ("Right_Eye",(36,42)),
    ("Left_Eye",(42,48)), 
    ("Nose",(27,35)),
    ("Jaw",(0,17)),
])

def shape_to_numpy_array(shape,dtype="int"):
    coordinates=np.zeros((68,2),dtype=dtype)

    for i in range(0,68):
        coordinates[i]=(shape.part(i).x,shape.part(i).y)
    return coordinates

def visualize_facial_landmarks(image,shape,colors=None,alpha=0.75):
    overlay=image.copy()
    output=image.copy()

    if colors is None:
        colors=[(0,0,0),(0,0,0),(0,0,0),
                (0,0,0),(0,0,0),
                (0,0,0),(0,0,0)]
    for (i,name) in enumerate(FACIAL_LANDMARK_INDEXES.keys()):
        print(i,name,"\n")
        (j,k)=FACIAL_LANDMARK_INDEXES[name]
        pts=shape[j:k]
        facial_features_coordinates[name]=pts

        if name=="Jaw":
            for l in range(1,len(pts)):
                ptA=tuple(pts[l-1])
                ptB=tuple(pts[l])
                cv2.line(overlay,ptA,ptB,colors[i],2)
    
    cv2.addWeighted(overlay,alpha,output,1-alpha,0,output)

    print(facial_features_coordinates)
    return output

detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


image=cv2.imread("1.jpg")
image=cv2.resize(image,(300,300),interpolation=cv2.INTER_AREA)
face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
faces=face_cascade.detectMultiScale(gray,1.1,4)

#TO DISPLAY BORDER AROUND THE FACE OF ANY IMAGE
# while True:
#     for (x,y,w,h) in faces:
#         cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
#     cv2.imshow("Face Detection",image)
#     key=cv2.waitKey(1)

#     if key==81 or key==113:
#         break


image2=cv2.imread("2.jpg")
image2=cv2.resize(image2,(300,300),interpolation=cv2.INTER_AREA)
face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
gray2=cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
faces2=face_cascade.detectMultiScale(gray2,1.1,4)

image3=cv2.imread("3.jpg")
image3=cv2.resize(image3,(300,300),interpolation=cv2.INTER_AREA)
face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
gray3=cv2.cvtColor(image3,cv2.COLOR_BGR2GRAY)
faces3=face_cascade.detectMultiScale(gray3,1.1,4)

image4=cv2.imread("4.jpg")
image4=cv2.resize(image4,(300,300),interpolation=cv2.INTER_AREA)
face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
gray4=cv2.cvtColor(image4,cv2.COLOR_BGR2GRAY)
faces4=face_cascade.detectMultiScale(gray4,1.1,4)

image5=cv2.imread("5.jpg")
image5=cv2.resize(image5,(300,300),interpolation=cv2.INTER_AREA)
face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
gray5=cv2.cvtColor(image5,cv2.COLOR_BGR2GRAY)
faces5=face_cascade.detectMultiScale(gray5,1.1,4)

faces_array=[]
faces_array.append(faces)
faces_array.append(faces2)
faces_array.append(faces3)
faces_array.append(faces4)
print(faces_array)

pointsf1=[]
pointsf2=[]
pointsf3=[]
pointsf4=[]
for (x,y,w,h) in faces:
    for i in range (x,x+w):
        for j in range (y,y+h):
            pointsf1.append([i,j])
for (x,y,w,h) in faces2:
    for i in range (x,x+w):
        for j in range (y,y+h):
            pointsf2.append([i,j])
for (x,y,w,h) in faces3:
    for i in range (x,x+w):
        for j in range (y,y+h):
            pointsf3.append([i,j])
for (x,y,w,h) in faces4:
    for i in range (x,x+w):
        for j in range (y,y+h):
            pointsf4.append([i,j])
# print(np.array(pointsf1).shape)
# print(np.array(pointsf2).shape)
# print(np.array(pointsf3).shape)
# print(np.array(pointsf4).shape)

# print(image[0][0])
# print(image2[0][0])
# print(np.add(image[0][0],image2[0][0]))
# print(np.divide(np.add(image[0][0],image2[0][0]),2).astype(int))

for i in range (0,300):
    for j in range (0,300):
        a=math.pow(gray[i][j],2)
        b=math.pow(gray2[i][j],2)

        y=np.add(a,b)
        z=math.sqrt(y)
        gray3[i][j]=np.divide(z,2).astype(int)

for i in range (0,300):
    for j in range (0,300):
        a=math.pow(gray2[i][j],2)
        b=math.pow(gray4[i][j],2)

        y=np.add(a,b)
        z=math.sqrt(y)
        gray5[i][j]=np.divide(z,2).astype(int)

# while True:
#     cv2.imshow("Gray Image 1",gray)
#     cv2.imshow("Gray Image 2",gray2)
#     cv2.imshow("Gray Image 3",gray3)
#     cv2.imshow("Gray Image 4",gray4)
#     cv2.imshow("Gray Image 5",gray5)
#     key=cv2.waitKey(1)

#     if key==81 or key==113:
#         break

rects=detector(gray2,1)

for (i,rects) in enumerate(rects):
    shape=predictor(gray2,rects)
    shape=shape_to_numpy_array(shape)

    output=visualize_facial_landmarks(image4,shape)
    cv2.imshow("Output",output)
    cv2.waitKey(0)
