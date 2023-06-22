import os
import numpy as np
import mediapipe as mp
import cv2

mphol=mp.solutions.holistic
mpdraw=mp.solutions.drawing_utils
hol=mphol.Holistic()
cap=cv2.VideoCapture(0)

def detector(im):
    im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    im.flags.writeable=False
    results=hol.process(im)
    im.flags.writeable=False
    im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    return results,im

def draw_landmarks(im,results):
    mpdraw.draw_landmarks(im,results.face_landmarks,mphol.FACEMESH_TESSELATION)
    mpdraw.draw_landmarks(im,results.pose_landmarks,mphol.POSE_CONNECTIONS)
    mpdraw.draw_landmarks(im,results.left_hand_landmarks,mphol.HAND_CONNECTIONS)
    mpdraw.draw_landmarks(im,results.right_hand_landmarks,mphol.HAND_CONNECTIONS)

def extract_keypoints(results):
    pos_arr=np.array([[ln.x,ln.y,ln.z,ln.visibility] for ln in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    face_arr=np.array([[ln.x,ln.y,ln.z] for ln in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
    lh_arr=np.array([[ln.x,ln.y,ln.z] for ln in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)    
    rh_arr=np.array([[ln.x,ln.y,ln.z] for ln in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pos_arr,face_arr,lh_arr,rh_arr])

if __name__=='__main__':
 while True:
    ret,frame=cap.read()
    res,frame=detector(frame)
    draw_landmarks(frame,res)
    print(len(extract_keypoints(res)))
    cv2.imshow('frame',frame)
    extract_keypoints(res)
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break