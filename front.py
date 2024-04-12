import cv2 as cv
import face_recognition
import numpy as np
import copy
import time 
import torch
import torch.nn as nn
from PIL import Image
from playsound import playsound
import streamlit as st

# Define global variables
frame_placeholder = st.empty()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Model loading
class ConvNet_Yaxis(torch.nn.Module):
    def __init__(self):
        super(ConvNet_Yaxis, self).__init__()

        f2 = 8
        self.layer2 = nn.Sequential(
            nn.Conv2d(1, f2, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(f2),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(50 * 25 * f2, 200)
        self.fc2 = nn.Linear(200, 20)
        self.fc3 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

class ConvNet_Xaxis(torch.nn.Module):
    def __init__(self):
        super(ConvNet_Xaxis, self).__init__()

        f2 = 8
        self.layer2 = nn.Sequential(
            nn.Conv2d(1, f2, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(f2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(50 * 25 * f2, 200)
        self.fc2 = nn.Linear(200, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

def maxAndMin(featCoords, mult=1):
    adj = 10 / mult
    listX = []
    listY = []
    for tup in featCoords:
        listX.append(tup[0])
        listY.append(tup[1])
    maxminList = np.array([min(listX) - adj, min(listY) - adj, max(listX) + adj, max(listY) + adj])
    return (maxminList * mult).astype(int), (np.array([sum(listX) / len(listX) - maxminList[0],
                                                        sum(listY) / len(listY) - maxminList[1]]) * mult).astype(int)

# Preps a color pic of the eye for input into the CNN
def process(im):
    left_eye = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    left_eye = cv.resize(left_eye, dsize=(100, 50))
    top = max([max(x) for x in left_eye])
    left_eye = (torch.tensor([[left_eye]]).to(dtype=torch.float, device=device)) / top
    return left_eye

# Function to handle cheating detected
def handle_cheating(cheating_counter):
    global last_beep_time

    current_time = time.time()
    if current_time - last_beep_time >= 2:
        # Play the beep sound if 10 seconds have passed since the last beep
        playsound('/home/von/Anti cheat/beep.mp3')
        last_beep_time = current_time  # Update the time of the last beep
    
    cheating_counter += 1  # Increment the cheating counter
    # Implement actions to be taken when cheating is detected
    # Example: Display warning message, disable access, log incident, etc.
    print("Cheating detected! Please maintain focus on the task.", cheating_counter)
    return cheating_counter

def eyetrack(xshift=100, yshift=150, frameShrink=0.15):
    frame_data_list = []  # List to store frame data
    cheating_counter = 0  # Initialize cheating counter
    ConvNetY = ConvNet_Yaxis().to(device)
    ConvNetY.load_state_dict(torch.load("/home/von/Anti cheat/yModels/model_y_epoch_50.plt", map_location=device))
    ConvNetY.eval()

    ConvNetX = ConvNet_Xaxis().to(device)
    ConvNetX.load_state_dict(torch.load("/home/von/Anti cheat/xModels/model_x_epoch_10.plt", map_location=device))
    ConvNetX.eval()

    webcam = cv.VideoCapture(0)

    if not webcam.isOpened():
        print("Error: Could not open webcam.")
        return []

    mvAvgx = []
    mvAvgy = []
    scale = 10
    margin = 200
    margin2 = 50
    mirrored_x = 0
    mirrored_y = 0

    # Initialize variables for gaze stabilization
    gaze_history = []  # List to store past gaze coordinates
    gaze_history_length = 10  # Length of the gaze history

    while True:
        ret, frame = webcam.read()
        if not ret:
            print("Error: Failed to capture frame from the webcam.")
            break
        
        frame = cv.flip(frame, 1)
        smallframe = cv.resize(copy.deepcopy(frame), (0, 0), fy=frameShrink, fx=frameShrink)
        smallframe = cv.cvtColor(smallframe, cv.COLOR_BGR2GRAY)

        feats = face_recognition.face_landmarks(smallframe)
        if len(feats) > 0:
            leBds, leCenter = maxAndMin(feats[0]['left_eye'], mult=1 / frameShrink)
            left_eye = frame[leBds[1]:leBds[3], leBds[0]:leBds[2]]
            left_eye = process(left_eye)
            x = ConvNetX(left_eye).item() * 1440 - xshift
            y = ConvNetY(left_eye).item() * 900 - yshift
            avx = sum(mvAvgx) / scale
            avy = sum(mvAvgy) / scale

            mvAvgx.append(x)
            mvAvgy.append(y)

            if len(mvAvgx) >= scale:
                if abs(avx - x) > margin and abs(avy - x) > margin:
                    mvAvgx = mvAvgx[5:]
                    mvAvgy = mvAvgy[5:]
                else:
                    if abs(avx - x) > margin2:
                        mvAvgx = mvAvgx[1:]
                    else:
                        mvAvgx.pop()
                    if abs(avy - y) > margin2:
                        mvAvgy = mvAvgy[1:]
                    else:
                        mvAvgy.pop()

                face_center = (frame.shape[1] // 2, frame.shape[0] // 2)
                mirrored_x = int(frame.shape[1] - x)
                mirrored_y = int(y)
                
                # Add current gaze coordinates to history
                if len(gaze_history) >= gaze_history_length:
                    gaze_history.pop(0)  # Remove oldest gaze coordinate if history is full
                gaze_history.append((mirrored_x, mirrored_y))  # Append current gaze coordinate
                
                # Smooth gaze coordinates using a moving average
                window_size = 3  # Adjust the window size according to your preference
                if len(gaze_history) >= window_size:
                   smoothed_gaze_x = sum([x for x, _ in gaze_history[-window_size:]]) // window_size  # Calculate moving average of X coordinates
                   smoothed_gaze_y = sum([y for _, y in gaze_history[-window_size:]]) // window_size  # Calculate moving average of Y coordinates
                else:
                   smoothed_gaze_x = mirrored_x  # Use current coordinate if history is not long enough
                   smoothed_gaze_y = mirrored_y  # Use current coordinate if history is not long enough


                # Draw stabilized gaze arrow
                arrow_color = (0, 255, 0)
                cv.arrowedLine(frame, face_center, (smoothed_gaze_x, smoothed_gaze_y), arrow_color, 2, cv.LINE_AA, 0, 0.3)

                # Display coordinates on screen
                cv.putText(frame, f'X: {smoothed_gaze_x}, Y: {smoothed_gaze_y}', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
                
                # Encode the frame to an image data object
                _, buffer = cv.imencode('.jpg', frame)
                frame_data = buffer.tobytes()

                frame_data_list.append(frame_data)

                # Limit the size of the frame data list to avoid memory issues
                max_frame_data_list_size = 100
                if len(frame_data_list) > max_frame_data_list_size:
                   frame_data_list.pop(0)  # Remove the oldest frame data

                # Perform cheating detection based on gaze coordinates
                if detect_cheating(mirrored_x, mirrored_y):
                   cheating_counter = handle_cheating(cheating_counter)  # Handle cheating detected

                # Display the frame in the Streamlit app
                frame_placeholder.image(frame, channels="BGR", use_column_width=True)

    webcam.release()
    cv.destroyAllWindows()

    return frame_data_list

def detect_cheating(gaze_x, gaze_y, threshold_x=0, threshold_y=0, max_attempts=10, disable_duration=10):
    # Implement cheating detection logic based on gaze coordinates
    # Check if gaze coordinates are outside the expected range with a threshold
    
    # Define the expected range of gaze coordinates
    min_x, max_x = -10, 250  # Example range for x-coordinate
    min_y, max_y = 50, 650   # Example range for y-coordinate
    
    # Calculate the threshold range
    min_x_threshold = min_x - threshold_x
    max_x_threshold = max_x + threshold_x
    min_y_threshold = min_y - threshold_y
    max_y_threshold = max_y + threshold_y
    
    # Check if gaze coordinates fall outside the threshold range
    if gaze_x < min_x_threshold or gaze_x > max_x_threshold or gaze_y < min_y_threshold or gaze_y > max_y_threshold:
        detect_cheating.attempts += 1  # Increment the attempts counter
        if detect_cheating.attempts >= max_attempts:
            detect_cheating.last_detection_time = time.time()  # Update the last detection time
            return True  # Cheating detected after maximum attempts
    else:
        detect_cheating.attempts = 0  # Reset the attempts counter if within the threshold
    
    # Check if enough time has passed since the last detection
    if time.time() - detect_cheating.last_detection_time < disable_duration:
        return False  # Cheating detection temporarily disabled
    else:
        return False  # No cheating detected

# Initialize attempts counter and last detection time as static attributes of the function
detect_cheating.attempts = 0
detect_cheating.last_detection_time = 0
last_beep_time = 0  # Variable to store the time of the last beep

def main():
    eyetrack()

if __name__ == "__main__":
    main()

