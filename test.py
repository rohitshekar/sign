import os
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
from handsign import extract_keypoints,draw_landmarks,detector
def hand_extract_keypoints(results):
    lh_arr=np.array([[ln.x,ln.y,ln.z] for ln in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)    
    rh_arr=np.array([[ln.x,ln.y,ln.z] for ln in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh_arr,rh_arr])

mphol=mp.solutions.holistic
hol=mphol.Holistic(min_tracking_confidence=0.5,min_detection_confidence=0.5)
sequence_length=30
no_frames=30
threshold=0.8
model=tf.keras.saving.load_model('sign9.h5')
actions=np.array(['hello','rock','fuck','thumsup','Love','you','peace','koreanlove','super','stop'])
cap=cv2.VideoCapture(0)



sequence=[]
sentence=[]
predictions=[]
while True:
        ret,frame=cap.read()
        res,frame=detector(frame)
        keypoints=hand_extract_keypoints(res)
        draw_landmarks(frame,res)
        sequence.append(keypoints)
        sequence=sequence[-15:]
        if len(sequence)==15:
         res=model.predict(np.expand_dims(sequence,axis=0))[0]
         if res[np.argmax(res)] > threshold:
          cv2.putText(frame,str(actions[np.argmax(res)]),(0,100),cv2.FONT_HERSHEY_COMPLEX,3,(0,255,0),2,cv2.LINE_AA)
            
            
        cv2.imshow('result',frame)
        # Break gracefully
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()     