import face_recognition  # Importing the face_recognition library for facial landmark detection
import numpy as np
import cv2 as cv
import copy
from matplotlib import pyplot as plt
import pyautogui
import time
import os

pyautogui.FAILSAFE = False
def maxAndMin(featCoords,mult = 1):
    """
    Function to find the maximum and minimum coordinates of facial features.

    Args:
        featCoords (list): List of coordinates of facial features.
        mult (float): Multiplier for adjusting the coordinates.

    Returns:
        tuple: A tuple containing the adjusted maximum and minimum coordinates.
    """
    adj = 10/mult  # Adjustment value
    listX = []
    listY = []
    for tup in featCoords:
        listX.append(tup[0])  # Extracting x-coordinates
        listY.append(tup[1])  # Extracting y-coordinates
    # Calculating maximum and minimum coordinates
    maxminList = np.array([min(listX)-adj,min(listY)-adj,max(listX)+adj,max(listY)+adj])
    print(maxminList)
    return (maxminList*mult).astype(int), (np.array([sum(listX)/len(listX)-maxminList[0], sum(listY)/len(listY)-maxminList[1]])*mult).astype(int)

# def findCircs(img):
#     # Implementation of circle detection algorithm using Hough Circles
#     circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 2, 20, param1 = 200, param2 = 50, minRadius=1, maxRadius=40)
#     # circles = np.uint16(np.around(circles))
#     return circles

# def findBlobs(img):
#     # Implementation of blob detection algorithm using SimpleBlobDetector
#     params = cv.SimpleBlobDetector_Params()
#     params.minThreshold = 10
#     params.maxThreshold = 200
#     params.filterByArea = True
#     params.maxArea = 3000
#     detector = cv.SimpleBlobDetector_create(params)
#     keypoints = detector.detect(img)
#     return keypoints

def getWebcam(feed=False):
    """
    Function to capture images from webcam and detect facial landmarks.

    Args:
        feed (bool): Flag to display the webcam feed.

    Returns:
        numpy.ndarray: An image array containing the detected facial landmarks.
    """
    webcam = cv.VideoCapture(0)  # Initializing the webcam
    # Frame coordinates go frame[y][x]
    haventfoundeye = True  # Flag to indicate if eyes are not found
    screenw = 1440  # Width of the screen
    screenh = 900  # Height of the screen

    while True:
        ret, frame = webcam.read()  # Reading a frame from the webcam
        smallframe = cv.resize(copy.deepcopy(frame), (0,0), fy=.15, fx=.15)  # Resizing the frame
        smallframe = cv.cvtColor(smallframe, cv.COLOR_BGR2GRAY)  # Converting to grayscale

        feats = face_recognition.face_landmarks(smallframe)  # Detecting facial landmarks
        if len(feats) > 0:  # If facial landmarks are detected
            # Extracting coordinates of the left eye and adjusting them
            leBds,leCenter = maxAndMin(feats[0]['left_eye'],mult = 1/.15)
            left_eye = frame[leBds[1]:leBds[3], leBds[0]:leBds[2]]  # Extracting the left eye region
            left_eye = cv.cvtColor(left_eye, cv.COLOR_BGR2GRAY)  # Converting left eye to grayscale
            ret, thresh = cv.threshold(left_eye, 50, 255, 0)  # Applying thresholding to the left eye

            # Find weighted average for center of the eye
            TMP = 255 - np.copy(thresh)  # Inverting the image
            y = np.sum(TMP, axis=1) / len(TMP[0])  # Calculating weighted average along y-axis
            x = np.sum(TMP, axis=0) / len(TMP)  # Calculating weighted average along x-axis
            y = y > np.average(y) + np.std(y)  # Thresholding along y-axis
            x = x > np.average(x) + np.std(x)  # Thresholding along x-axis
            y = int(np.dot(np.arange(1, len(y) + 1), y) / sum(y))  # Calculating y-coordinate of the eye center
            x = int(np.dot(np.arange(1, len(x) + 1), x) / sum(x))  # Calculating x-coordinate of the eye center
            haventfoundeye = False  # Setting the flag to indicate eyes are found

            left_eye = cv.cvtColor(left_eye, cv.COLOR_GRAY2BGR)  # Converting left eye to BGR color space
            cv.circle(left_eye, (x, y), 2, (20, 20, 120), 3)  # Drawing a circle at the eye center
            cv.circle(left_eye, (int(leCenter[0]), int(leCenter[1])), 2, (120, 20, 20), 3)  # Drawing a circle at the estimated center

            if feed:  # If feed flag is True
                cv.imshow('frame', left_eye)  # Displaying the left eye

                if cv.waitKey(1) & 0xFF == ord('q'):  # Exiting if 'q' is pressed
                    break
            elif not haventfoundeye:  # If eyes are found
                plt.imshow(left_eye)  # Displaying the left eye using matplotlib
                plt.title('my EYEBALL')  # Setting the title
                plt.show()  # Showing the plot
                return left_eye  # Returning the left eye image array

def getEye(times = 1,frameShrink = 0.15, coords = (0,0), counterStart = 0, folder = "eyes"):
    """
    Function to capture eye images from webcam.

    Args:
        times (int): Number of eye images to capture.
        frameShrink (float): Factor to resize the captured frames.
        coords (tuple): Coordinates of the eye.
        counterStart (int): Starting index for naming the captured images.
        folder (str): Folder to save the captured images.
    """
    os.makedirs(folder, exist_ok=True)  # Creating the folder if it doesn't exist
    webcam = cv.VideoCapture(0)  # Initializing the webcam
    counter = counterStart  # Initializing the counter

    while counter < counterStart+times:  # Loop until desired number of images are captured
        ret, frame = webcam.read()  # Reading a frame from the webcam
        smallframe = cv.resize(copy.deepcopy(frame), (0, 0), fy=frameShrink, fx=frameShrink)  # Resizing the frame
        smallframe = cv.cvtColor(smallframe, cv.COLOR_BGR2GRAY)  # Converting to grayscale

        feats = face_recognition.face_landmarks(smallframe)  # Detecting facial landmarks
        if len(feats) > 0:  # If facial landmarks are detected
            leBds, leCenter = maxAndMin(feats[0]['left_eye'], mult=1/frameShrink)  # Extracting and adjusting left eye coordinates

            left_eye = frame[leBds[1]:leBds[3], leBds[0]:leBds[2]]  # Extracting the left eye region
            left_eye = cv.cvtColor(left_eye, cv.COLOR_BGR2GRAY)  # Converting left eye to grayscale
            left_eye = cv.resize(left_eye, dsize=(100, 50))  # Resizing the left eye
            cv.imshow('frame', left_eye)  # Displaying the left eye

            if cv.waitKey(1) & 0xFF == ord('q'):  # Exiting if 'q' is pressed
                break

            cv.imwrite(
                folder + "/" + str(coords[0]) + "." + str(coords[1]) + "." + str(
                    counter) + ".jpg", left_eye)  # Saving the left eye image
            counter += 1  # Incrementing the counter

for i in [404,951]:  # Looping over x-coordinates
    for j in [383,767]:  # Looping over y-coordinates
        print(i,j)  # Printing the coordinates
        pyautogui.moveTo(i, j)  # Moving the mouse pointer to the specified coordinates
        input("Press Enter to continue...")  # Waiting for user input
        pyautogui.moveTo(i, j)  # Moving the mouse pointer again to ensure focus
        getEye(times = 10, coords=(i,j),counterStart=0, folder = "testeyes")  # Capturing eye images

# getEye(times = 1, coords=(360,225),counterStart=0)  # Example of capturing a single eye image
# getWebcam(True)  # Example of displaying webcam feed

