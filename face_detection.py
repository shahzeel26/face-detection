import cv2 #importing the library of cv2

from random import randrange # this will help to import random color identificiation boxes
trained_face_data=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#choose an image to detect faces in
# img=cv2.imread('img1.jpg')

#to capture video from webcam
webcam=cv2.VideoCapture(0)
# key=cv2.waitKey(1)

#iterate forever over the frames
while True:
    successful_frame_read, frame=webcam.read()
    #must convert to grayscale
    grayscaled_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

     #detect faces
    face_coordinates=trained_face_data.detectMultiScale(grayscaled_img)

    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(randrange(256),randrange(256),randrange(256)),10)

    cv2.imshow('zeel shah face detector',frame)
    # cv2.waitKey(1)
    key=cv2.waitKey(1)
    #stop if q or Q key is pressed
    if key==81 or key==113:
        break
#Release the VideoCapture object
webcam.release()



# #must convert to grayscale
# grayscaled_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# #detect faces
# face_coordinates=trained_face_data.detectMultiScale(grayscaled_img)

# print(face_coordinates)
# (x,y,w,h)=face_coordinates[0]

# # cv2.rectangle(img,(209,137),(209+373,137+373),(0,255,0),2)
# cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),10)
# (x,y,w,h)=face_coordinates[1]


# cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),10)

# (x,y,w,h)=face_coordinates[2]

# cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),10)

# for (x,y,w,h) in face_coordinates:
#     cv2.rectangle(img,(x,y),(x+w,y+h),(randrange(128,256),randrange(128,256),randrange(128,256)),10)

#display the image with the faces
# cv2.imshow('zeel shah face detector',img)
# cv2.waitKey()



print("Code completed")