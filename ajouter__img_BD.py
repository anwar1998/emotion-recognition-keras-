import numpy as np
import cv2
import matplotlib.pyplot as plt
#test_image = cv2.imread('image_tres_jolie.jpg')
video = cv2.VideoCapture(0)
haar_cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def convertToRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
sampleNum=0
drp = True
k=0
while drp:
    ret, test_image = video.read()
    if ret == True:
        test_image_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        faces_rects = haar_cascade_face.detectMultiScale(test_image_gray, scaleFactor=1.2, minNeighbors=5);
        if not faces_rects == ():
            for (x, y, w, h) in faces_rects:
                cv2.rectangle(test_image, (x, y), (x + w, y + h), (0, 255, 255), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (30, 30)
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2
            image = cv2.putText(test_image, ""+str(sampleNum), org, font,
                                fontScale, color, thickness, cv2.LINE_AA)
            roi = test_image_gray[y:y+h, x:x+w]
            cv2.imwrite("base_donne_img/brahim/"+str(sampleNum) + ".jpg",roi)
            sampleNum = sampleNum + 1
        cv2.imshow("rrr", test_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            drp = False
        elif sampleNum>1000:
            drp = False
    else:
        drp = False
video.release()
cv2.destroyAllWindows()

