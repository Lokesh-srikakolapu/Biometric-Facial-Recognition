import cv2
import numpy as np
import face_recognition

img1 = face_recognition.load_image_file('pics/Sundar.jpeg')
img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)

# img2 = face_recognition.load_image_file('mypics/2 year.jpg')
# img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)

# img5 = face_recognition.load_image_file('mypics/5 year.jpg')
# img5 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)

img7 = face_recognition.load_image_file('pics/Sundar1.jpeg')
img7 = cv2.cvtColor(img7,cv2.COLOR_BGR2RGB)

fcLoc1 = face_recognition.face_locations(img1)[0]
encode1 = face_recognition.face_encodings(img1)[0]
cv2.rectangle(img1,(fcLoc1[3],fcLoc1[0]),(fcLoc1[1],fcLoc1[2]),(255,255,255),2)

fcLoc7 = face_recognition.face_locations(img7)[0]
encode7 = face_recognition.face_encodings(img7)[0]
cv2.rectangle(img7,(fcLoc7[3],fcLoc7[0]),(fcLoc7[1],fcLoc7[2]),(255,255,255),2)

results = face_recognition.compare_faces([encode1],encode7)
fcDis = face_recognition.face_distance([encode1], encode7)
print(results,fcDis)
cv2.putText(img7,f'{results} {round(fcDis[0],2)}',(0,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)



cv2.imshow('Now Me',img1)
# cv2.imshow('2 back Me',img2)
# cv2.imshow('5 back Me',img5)
cv2.imshow('7 back Me',img7)
cv2.waitKey(0)
