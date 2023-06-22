import os
import numpy as np
import cv2
import mediapipe as mp
from handsign import extract_keypoints,draw_landmarks,detector
mphol=mp.solutions.holistic
hol=mphol.Holistic(min_tracking_confidence=0.5,min_detection_confidence=0.5)
sequence_length=30
no_frames=15
data_path=os.path.join('mpdata4')
actions=np.array(['hello','rock','fuck','thumsup','Love','you','peace','koreanlove','super','stop'])
def hand_extract_keypoints(results):
    lh_arr=np.array([[ln.x,ln.y,ln.z] for ln in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)    
    rh_arr=np.array([[ln.x,ln.y,ln.z] for ln in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh_arr,rh_arr])

cap=cv2.VideoCapture(0)
for action in actions:
    for i in range(sequence_length):
        for j in range(no_frames):
            ret,frame=cap.read()
            res,frame=detector(frame)
            draw_landmarks(frame,res)
            if j==0:
                cv2.putText(frame, 'STARTING COLLECTION', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, 'Collecting frames for {} Video Number {}'.format(action, i), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                cv2.imshow('OpenCV Feed', frame)
                cv2.waitKey(500)
            else: 
                cv2.putText(frame, 'Collecting frames for {} Video Number {}'.format(action, i), (15,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                cv2.imshow('OpenCV Feed', frame)
            key=hand_extract_keypoints(res)
            res_path=os.path.join(data_path,action,str(i),str(j))
            np.save(res_path,key)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
cap.release()
cv2.destroyAllWindows()     
