import cv2
import mediapipe as mp
import time
import numpy as np
import gesturerecognition as gr
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
while True:
    tstart = time.time()
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print (results)
    lmList = [] #empty list
    if results.multi_hand_landmarks: #list of all hands detected.
        #By accessing the list, we can get the information of each hand's corresponding flag bit
        for handlandmark in results.multi_hand_landmarks:
            for id,lm in enumerate(handlandmark.landmark): #adding counter and returning it
                # Get finger joint points
                h,w,_ = img.shape
                cx,cy = int(lm.x*w),int(lm.y*h)
                lmList.append([id,cx,cy]) #adding to the empty list 'lmList'
            mpDraw.draw_landmarks(img,handlandmark,mpHands.HAND_CONNECTIONS)
        #print(type(results.multi_hand_world_landmarks[0].landmark[0]))
        #print(type(results.multi_hand_world_landmarks[0]))
        gr.gestrec(results.multi_hand_landmarks[0])
        #print(np.array(results.multi_hand_world_landmarks[0].landmark[0]))
        #print(results.multi_hand_world_landmarks[0][5])
        #print(results.multi_hand_world_landmarks[0][17])
        #input()
    
    #if lmList != []:
    #    #getting the value at a point
    #                    #x      #y
    #    x1,y1 = lmList[4][1],lmList[4][2]  #thumb
    #    x2,y2 = lmList[8][1],lmList[8][2]  #index finger
    #    #creating circle at the tips of thumb and index finger
    #    cv2.circle(img,(x1,y1),13,(255,0,0),cv2.FILLED) #image #fingers #radius #rgb
    #    cv2.circle(img,(x2,y2),13,(255,0,0),cv2.FILLED) #image #fingers #radius #rgb
    #    cv2.line(img,(x1,y1),(x2,y2),(255,0,0),3)  #create a line b/w tips of index finger and thumb
 

    # img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    tend = time.time()
    #print(tend -tstart)
    cv2.imshow('Input', img)
    cv2.waitKey(10)


cap.release()       
cv2.destroyAllWindows()