import cv2
import mediapipe as mp
import time

cap=cv2.VideoCapture(0) #uses webcam0

mpHands= mp.solutions.hands
hands=mpHands.Hands()
mpDraw=mp.solutions.drawing_utils

pTime=0
cTime=0
while True:
    success,img=cap.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #convert it to RGB
    results=hands.process(imgRGB)#processes the image
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            #we extract the information for each hand
            for id, lm in enumerate(handLms.landmark):
                #print(id,lm)
                h,w,c=img.shape#we extract the height , weight and channel of the image
                cx,cy=int(lm.x*w),int(lm.y*h)#coordinates of the parts
                print(id,cx,cy)
                if id==0:
                    cv2.circle(img,(cx,cy),25,(255,0,255),cv2.FILLED)

            mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)
    #display FPS
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime

    #put the text on the screen
    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,255),2)

    cv2.imshow("Image",img)
    cv2.waitKey(1)