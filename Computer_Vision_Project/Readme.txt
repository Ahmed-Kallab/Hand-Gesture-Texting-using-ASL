Project Name: Hand Gesture Texting using ASL.

Description: This project provides real-time feedback for specific hand gestures in American Sign Language (ASL). Its purpose is to assist individuals in communicating with others who are not familiar with ASL gestures and to bridge gaps in communication.

Installation: This project runs on Python 3.11.5. The required modules to enable all functionalities are OpenCV, Mediapipe, Numpy, and Scikit-learn.

Usage: I recommend starting by opening the "collect_images" file and running it. This file activates your laptop camera and prompts you to enter the correct key. Upon pressing the key, it will begin capturing images of the respective hand gesture. Next, open "create_dataset"; this is required to create a file where all the captured images will be saved, along with the letter used to teach the model. After that, open "train_classifier" to train the model. Lastly, open "inference_classification" to activate the laptop camera again and interpret every correct gesture that was performed.

Data Collection: This project collects images directly from users and does not rely on any external images to train the model, except for those captured earlier by the user.

Model Training: The model utilizes 20% of the captured images to self-train for each letter, and no external images are used.

Inference: The model makes decisions based on the trained images. During training, each image detects the hand and assigns different indexes to each finger. Each finger has three points that track its changes in each letter. A total of 100 images are captured for each letter, preferably in different positions but with the same hand gesture.

Results: The training model yields an accuracy score, indicating the percentage of accuracy of the model.

License: The initial code was provided by Ahmed Ibrahim on GitHub.

Contributing: Numerous debugging sessions and feature additions were incorporated into the model to ensure an outstanding user experience.

Credits: All edits and added features were implemented by Ahmed Kallab.

Acknowledgments: Gratitude to everyone who contributed to this project, whether by offering advice, reviewing code, or providing moral support.
