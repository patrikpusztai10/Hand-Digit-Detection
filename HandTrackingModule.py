import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self):

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self,img,draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert it to RGB
        self.results = self.hands.process(imgRGB)  # processes the image
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img
    def findPosition(self,img,handNo=0,draw=True):
        # we extract the information for each hand
        lmList=[]#list which will contain the landmarks
        if self.results.multi_hand_landmarks:
            myHand=self.results.multi_hand_landmarks[handNo]#we get the information for a specific hand
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape  # we extract the height , weight and channel of the image
                cx, cy = int(lm.x * w), int(lm.y * h)# coordinates of the parts
                lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return lmList
def main():
    pTime = 0
    cTime = 0
    detector=handDetector()
    cap = cv2.VideoCapture(0)  # uses webcam0
    while True:
        success, img = cap.read()
        img=detector.findHands(img)
        lmList=detector.findPosition(img)
        if len(lmList)!=0:
            print(lmList[0])
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        # put the text on the screen
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2)
        cv2.imshow("Image", img)
        cv2.waitKey(1)
if __name__=="__main__":
    main()
