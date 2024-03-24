import pickle
import cv2
import mediapipe as mp
import numpy as np
import train_classifier as train_classifier
import time

# Load the pre-trained machine learning model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Open the webcam
cap = cv2.VideoCapture(0)

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define a dictionary mapping numerical labels to alphabetical characters from 'A' to 'Z'
labels_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',
                10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',
                19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z'}

# Used to know the predicted letter.
current_letter = None
letter_count = 0
timer_time = 10  # Adjust the time as needed

while True:
    data_aux = []
    x_ = []
    y_ = []

    # Read a frame from the webcam
    ret, frame = cap.read()

    # Get the height and width of the frame
    H, W, _ = frame.shape

    # Convert the frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe Hands
    results = hands.process(frame_rgb)
    
    # Check if hand landmarks are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks and hand connections on the frame
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)
                
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # Calculate bounding box coordinates
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        # Make prediction using the model if hand landmarks are detected
        max_sequence_length = max(len(seq) for seq in train_classifier.data_dict['data'])
        data_aux_padded = data_aux + [0] * (max_sequence_length - len(data_aux))

        # Make prediction using the model
        prediction = model.predict([np.asarray(data_aux_padded)])
        predicted_character = labels_dict[int(prediction[0])]

        # Check if the detected letter is the same as the previous one
        if predicted_character == current_letter:
            letter_count += 1
            if letter_count >= timer_time:
                print(predicted_character)  # Print the detected letter
                time.sleep(3)
                letter_count = 0  # Reset the count
        else:
            current_letter = predicted_character
            letter_count = 1  # Reset the count if the detected letter changes

        # Draw bounding box and predicted character on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    # Display the frame
    cv2.imshow('frame', frame)
    
    # Wait for key press and check if 'q' is pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()