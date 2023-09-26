import cv2
# Numpy for array related functions
import numpy as np
# Dlib for deep learning based Modules and face landmark detection
import dlib
#face_utils for basic operations of conversion
from imutils import face_utils
from pygame import mixer
import time

import numpy as np
import cv2
from tkinter import *
from scipy.spatial import distance as distance

root = Tk()
root.title(" Driver Drowsiness ")
root.configure(background="lightblue")

#Initializing the camera and taking the instance


#Initializing the face detector and landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"C:\Users\LENOVO\Downloads\Driver_Drowsisness\shape_predictor_68_face_landmarks.dat")

#status marking for current state




def compute(ptA,ptB):
	dist = np.linalg.norm(ptA - ptB)
	return dist

def blinked(a,b,c,d,e,f):
	up = compute(b,d) + compute(c,e)
	down = compute(a,f)
	ratio = up/(2.0*down)

	#Checking if it is blinked
	if(ratio>0.25):
		return 2
	elif(ratio>0.21 and ratio<=0.25):
		return 1
	else:
		return 0

def main():
	cap = cv2.VideoCapture(r"C:\Users\LENOVO\Pictures\Camera Roll")
	sleep = 0
	drowsy = 0
	active = 0
	status=""
	color=(0,0,0)
	count=0
	while True:
		_, frame = cap.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		faces = detector(gray)
		face_frame = frame.copy()
		#detected face in faces array
		for face in faces:
			x1 = face.left()
			y1 = face.top()
			x2 = face.right()
			y2 = face.bottom()

			
			cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

			landmarks = predictor(gray, face)
			landmarks = face_utils.shape_to_np(landmarks)

			#The numbers are actually the landmarks which will show eye
			left_blink = blinked(landmarks[36],landmarks[37], 
				landmarks[38], landmarks[41], landmarks[40], landmarks[39])
			right_blink = blinked(landmarks[42],landmarks[43], 
				landmarks[44], landmarks[47], landmarks[46], landmarks[45])
			
			#Now judge what to do for the eye blinks
			if(left_blink==0 or right_blink==0):
				sleep+=1
				drowsy=0
				active=0
				if(sleep>6):
						count+=1
						status="SLEEPING !!!"
						color = (255,0,0)
						mixer.init()
						mixer.music.load(r"C:\Users\LENOVO\Downloads\Driver_Drowsisness\sound\warning.mpeg")
						cv2.imwrite("dataset/frame_yawn%d.jpg"% sleep, frame)
						mixer.music.play()
						while mixer.music.get_busy():  # wait for music to finish playing
							time.sleep(1)
			elif(left_blink==1 or right_blink==1):
				sleep=0
				active=0
				drowsy+=1
				if(drowsy>6):
						count+=1
						status="Drowsy !"
						color = (0,0,255)
						mixer.init()
						mixer.music.load(r"C:\Users\LENOVO\Downloads\Driver_Drowsisness\sound\warning_yawn.mpeg")
						mixer.music.play()
						while mixer.music.get_busy():  # wait for music to finish playing
							time.sleep(1)
			else:
				drowsy=0
				sleep=0
				active+=1
				if(active>6):
					status="Active :)"
					color = (0,255,0)
				
			cv2.putText(frame, status, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color,3)

			for n in range(0, 68):
				(x,y) = landmarks[n]
				cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

		cv2.imshow("Frame", frame)
		
		key = cv2.waitKey(1) & 0xFF 
	
	

		if key == ord('q'):
			if count >= 0:
				t3.delete("1.0", END)
				t3.insert(END, count)
			break

def camera():
	cap = cv2.VideoCapture(0)
	sleep = 0
	drowsy = 0
	active = 0
	status=""
	color=(0,0,0)
	count=0
	while True:
		_, frame = cap.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		faces = detector(gray)
		face_frame = frame.copy()
		#detected face in faces array
		for face in faces:
			x1 = face.left()
			y1 = face.top()
			x2 = face.right()
			y2 = face.bottom()

			
			cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

			landmarks = predictor(gray, face)
			landmarks = face_utils.shape_to_np(landmarks)

			#The numbers are actually the landmarks which will show eye
			left_blink = blinked(landmarks[36],landmarks[37], 
				landmarks[38], landmarks[41], landmarks[40], landmarks[39])
			right_blink = blinked(landmarks[42],landmarks[43], 
				landmarks[44], landmarks[47], landmarks[46], landmarks[45])
			
			#Now judge what to do for the eye blinks
			if(left_blink==0 or right_blink==0):
				sleep+=1
				drowsy=0
				active=0
				if(sleep>6):
						count+=1
						status="SLEEPING !!!"
						color = (255,0,0)
						mixer.init()
						mixer.music.load(r"C:\Users\LENOVO\Downloads\Driver_Drowsisness\sound\warning.mpeg")
						cv2.imwrite("dataset/frame_yawn%d.jpg"% sleep, frame)
						mixer.music.play()
						while mixer.music.get_busy():  # wait for music to finish playing
							time.sleep(1)
			elif(left_blink==1 or right_blink==1):
				sleep=0
				active=0
				drowsy+=1
				if(drowsy>6):
						status="Drowsy !"
						color = (0,0,255)
						mixer.init()
						mixer.music.load(r"C:\Users\LENOVO\Downloads\Driver_Drowsisness\sound\warning_yawn.mpeg")
						mixer.music.play()
						while mixer.music.get_busy():  # wait for music to finish playing
							time.sleep(1)
			else:
				drowsy=0
				sleep=0
				active+=1
				if(active>6):
					status="Active :)"
					color = (0,255,0)
				
			cv2.putText(frame, status, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color,3)

			for n in range(0, 68):
				(x,y) = landmarks[n]
				cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

		cv2.imshow("Frame", frame)
		
		key = cv2.waitKey(1) & 0xFF 
	
	

		if key == ord('q'):
			if count >= 0:
				t4.delete("1.0", END)
				t4.insert(END, count)
			break


w2 = Label(root,justify=LEFT, text=" Driver Drowsiness Detection using Machine learning ")
w2.config(font=("Elephant", 30),background="lightblue")
w2.grid(row=1, column=0, columnspan=2, padx=100,pady=40)

NameLb1 = Label(root, text="Please Select the Options  ")
NameLb1.config(font=("Elephant", 12),background="lightblue")
NameLb1.grid(row=5, column=0, pady=10)

S1Lb = Label(root,  text="Video")
S1Lb.config(font=("Elephant", 14))
S1Lb.grid(row=7, column=0, pady=10 )

S2Lb = Label(root,  text="Use Camera")
S2Lb.config(font=("Elephant", 14))
S2Lb.grid(row=8, column=0,pady=10)

lr = Button(root, text="Video",height=2, width=10, command=main)
lr.config(font=("Elephant", 12),background="green")
lr.grid(row=15, column=0,pady=20)
lr = Button(root, text="Camera",height=2, width=10, command=camera)
lr.config(font=("Elephant", 12),background="green")
lr.grid(row=16, column=0,pady=20)

NameLb = Label(root, text="Predict using:")
NameLb.config(font=("Elephant", 15),background="lightblue")
NameLb.grid(row=13, column=0, pady=20)

t3 = Text(root, height=2, width=15)
t3.config(font=("Elephant", 15))
t3.grid(row=15, column=1 ,padx=60)
t4 = Text(root, height=2, width=15)
t4.config(font=("Elephant", 15))
t4.grid(row=16, column=1 ,padx=60)

root.mainloop()

