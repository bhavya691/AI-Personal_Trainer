# import modules
import cv2
import mediapipe as mp
import math
# initializing class 
class PoseDetector:
    def __init__(self, mode=False, upperBody=False, smoothness=True, detectionCon = 0.5, trackingCon = 0.5):
        self.mode = mode
        self.upperBody = upperBody
        self.smoothness = smoothness
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upperBody, self.smoothness, self.detectionCon, self.trackingCon)


    def findPos(self, frame, draw = True):

        # converting image from BGR to RGB 
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # processing frame and getting landmarks
        self.results = self.pose.process(frameRGB)

        # drawing circles with lines if there is some result and drawing is true 
        if(self.results.pose_landmarks):
            if (draw):
                self.mpDraw.draw_landmarks(frame, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        
        # returning frame 
        return frame


    def getPos(self, frame, draw = True):
        # making a list 
        self.lmList = []
        # checking that if there is any result 
        if self.results.pose_landmarks:
            # looping through each landmark one by one enumerate for getting id
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = frame.shape
                # taking cx and cy according to the image and then appending id, cx and cy in lmList
                cx, cy = int(lm.x*w), int(lm.y*h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(frame, (cx, cy), 10, (0,0,255), cv2.FILLED)

        # returning lmList
        return self.lmList
    
    def findAngle(self, frame, p1, p2, p3, draw=True):
        # Getting landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # Calculate the Angle
        angle = math.degrees(math.atan2(y3-y2, x3-x2) - math.atan2(y1-y2, x1-x2))
        if angle < 0:
            angle += 360

        # Drawing circles on landmarks
        if draw:
            cv2.line(frame, (x1,y1), (x2,y2), (255,255,255), 4)
            cv2.line(frame, (x3,y3), (x2,y2), (255,255,255), 4)
            cv2.circle(frame, (x1, y1), 10, (0,255,255), cv2.FILLED)
            cv2.circle(frame, (x1, y1), 20, (0,255,255), 2)
            cv2.circle(frame, (x2, y2), 10, (0,255,255), cv2.FILLED)
            cv2.circle(frame, (x2, y2), 20, (0,255,255), 2)
            cv2.circle(frame, (x3, y3), 10, (0,255,255), cv2.FILLED)
            cv2.circle(frame, (x3, y3), 20, (0,255,255), 2)
            # cv2.putText(frame, str(int(angle)), (x2 - 20, y2+50), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
        
        # returning the vlaue of angle 
        return angle


def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 700)
    detector = PoseDetector()
    while True:
        ret, frame = cap.read()
        frame = detector.findPos(frame)
        lmList = detector.getPos(frame)
        print(lmList)
        cv2.imshow('pose', frame)
        k = cv2.waitKey(10)
        if k == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()