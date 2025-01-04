import cv2
import numpy as np
import face_recognition

imgElon = face_recognition.load_image_file('image/Elon_Musk.jpg')
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)
image_test = face_recognition.load_image_file('image/BILL.jpg')
image_test = cv2.cvtColor(image_test, cv2.COLOR_BGR2RGB)


faceLoc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(225,0,255),2)


faceLoc_test = face_recognition.face_locations(image_test)[0]
encodeElon_test = face_recognition.face_encodings(image_test)[0]
cv2.rectangle(image_test,(faceLoc_test[3],faceLoc_test[0]),(faceLoc_test[1],faceLoc_test[2]),(225,0,255),2)


results = face_recognition.compare_faces([encodeElon],encodeElon_test)


faceDis = face_recognition.face_distance([encodeElon],encodeElon_test)

cv2.putText(image_test,f'{results}{round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,225))

cv2.imshow('Elon musk test', image_test)
cv2.imshow('Elon musk',imgElon)
cv2.waitKey(0)