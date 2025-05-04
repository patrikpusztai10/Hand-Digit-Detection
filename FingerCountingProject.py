import cv2
import time
import os
import HandTrackingModule as htm

wCam,hCam=640,480
cap=cv2.VideoCapture(0)
cap.set(3,wCam) #width
cap.set(4,hCam) #height

folderPath="FingerImages"
myList= os.listdir(folderPath)

overlayList=[]
for imPath in myList:
    image=cv2.imread(f"{folderPath}/{imPath}")
    overlayList.append(image)#we add the images to cv2

pTime=0
tipIds=[4,8,12,16,20]
detector=htm.handDetector()
while True:
    fingers = []
    success,img=cap.read()
    img=detector.findHands(img)
    lmList=detector.findPosition(img,draw=False)
    if len(lmList)!=0:

        #Special case for the thumb-Right hand
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:  # verify if point 3-THUMB_IP is on the left or right
            fingers.append(1)  # open
        else:
            fingers.append(0)  # close
        for id in range(1,5):
            if lmList[tipIds[id]][2]<lmList[tipIds[id]-2][2]:#verify if it's lower or not
                fingers.append(1) #open
            else:
                fingers.append(0) #close

        totalFingers=fingers.count(1)
        h,w,c=overlayList[0].shape#returns to height and width of the image
        img[0:h,0:w]=overlayList[totalFingers]
        cv2.rectangle(img,(420,225),(570,425),(0,255,0),cv2.FILLED)
        cv2.putText(img,f"{totalFingers}",(445,374),cv2.FONT_HERSHEY_COMPLEX,5,(255,255,0),10)
    #display the frame rate
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,f"FPS:{int(fps)}",(400,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

    cv2.imshow("Image",img)
    cv2.waitKey(1)

