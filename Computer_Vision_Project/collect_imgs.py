import os
import cv2

#What is done?
#In this file we collect data for the different hand Gestures!
#Opening the camera to capture in real-time the different letters!
#Captured image are saved in a folder called "data", which is organized in another folder. 

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

#Variables: 
number_of_Alphabets = 26
dataset_size = 100
flag = 0
Letter = "Nothing in here"
Letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
           "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", 
           "Y", "Z"]

cap = cv2.VideoCapture(0)
for j in range(number_of_Alphabets):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for Alphabet {}'.format(j))
    
    done = False
    while True:
        Letter = Letters[flag]
        ret, frame = cap.read()
        cv2.putText(frame, f'Ready? Press {Letter} ! :)', (100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord(f'{Letter}'):
            flag += 1
            counter = 0
            break
            
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(1)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()
