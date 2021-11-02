# importing modeules 
import cv2
import time
import numpy as np
import PoseClass as pc

# initializing webcam and Detector
cap = cv2.VideoCapture(0)
detector = pc.PoseDetector()
count = 0
dir  = 0
pTime = 0
# starting and doing the things after starting webcam
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 2)
    # resizing dimensions of webcam
    frame = cv2.resize(frame, (1280,720))


    frame = detector.findPos(frame, False)

    # getting position of every point 
    lmList = detector.getPos(frame, False)
    if len(lmList) != 0:
        # finding Angle
        angle = detector.findAngle(frame, 12, 14, 16)

        per = np.interp(angle, (70, 160), (0,100))
        bar = np.interp(angle, (70, 150), (650, 100))
        color = (255,0,255)
        if per == 0:
            color = (0,255,0)
            if dir == 0:
                count+=0.5
                dir = 1
        if per == 100:
            color = (0,255,0)
            if dir == 1:
                count+=0.5
                dir = 0
                
        # Drawing bar 
        cv2.rectangle(frame, (1100,100), (1175,650), color, 3)
        cv2.rectangle(frame, (1100, int(bar)), (1175,650), color, cv2.FILLED)
        cv2.putText(frame, f'{int(per)} %', (1100,75), cv2.FONT_HERSHEY_PLAIN, 4, color, 4)

        # drawing current count 
        cv2.rectangle(frame, (0,450), (300,770), (0,255,0), cv2.FILLED)
        cv2.putText(frame, str(int(count)), (50,640), cv2.FONT_HERSHEY_PLAIN, 10, (255,0,0), 20)

        # getting frame rate 
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(frame, str(int(fps)), (50,100), cv2.FONT_HERSHEY_PLAIN, 5, (255,0,0), 5)

    # showing the frame 
    cv2.imshow('pose', frame)

    # condition for getting out
    k = cv2.waitKey(10)
    if k == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()