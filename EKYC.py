import streamlit as st
import cv2
import numpy as np
import mediapipe as mp  # You may need to install this library using pip
import time
import random
import dlib
from scipy.spatial import distance as dist


st.title('Finger Numbers Instructions')

# Create a VideoCapture object for the default camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error('Error: Could not open the camera.')
else:
    st.success('Camera opened successfully.')

# Define video recording settings
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, (frame_width, frame_height))

# Create a placeholder for displaying the video feed
video_placeholder = st.empty()

# Create a list of instructions
instructions = [random.randint(1, 5), random.randint(1, 5), random.randint(1, 5), random.randint(1, 5)]

# Initialize the instruction index
instruction_index = 0

# Define the time interval (in seconds) for changing instructions
interval = 5  # Change instructions every 5 seconds

# Create an HTML placeholder for displaying the animated number
html_placeholder = st.empty()

instruction_change_time = time.time()

start_time = time.time()

Thankyou = "Thank You!"

frame_list = []

end = False

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        st.error('Error: Could not read a frame.')
        break

    out.write(frame)

    # Overlay numbers on detected hand region
    # Check if it's time to change the instruction
    if time.time() - instruction_change_time >= interval:
        instruction_change_time = time.time()
        instruction_index = (instruction_index + 1) % len(instructions)  # Wrap around to the beginning if needed

    if time.time() - start_time < interval * (len(instructions)):
        # Display the animated number using HTML/CSS and JavaScript
        html_code = f"""
        <div id="number-container">
            <div class="number">{instructions[instruction_index]}</div>
        </div>
        <style>
            #number-container {{
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100%;
            }}
            .number {{
                font-size: 72px;
                animation: fade 2s linear infinite;
            }}
            @keyframes fade {{
                0% {{ opacity: 0; }}
                50% {{ opacity: 1; }}
                100% {{ opacity: 0; }}
            }}
        </style>
        <script>
            // JavaScript code can be added here for more interactivity
        </script>
        """
    else:
        end = True
        html_code = f"""
            <div id="string-container">
                <div class="string">{Thankyou}</div>
            </div>
            <style>
                #string-container {{
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100%;
                }}
                .string {{
                    font-size: 36px;
                    animation: fade 2s linear infinite;
                }}
                @keyframes fade {{
                    0% {{ opacity: 0; }}
                    50% {{ opacity: 1; }}
                    100% {{ opacity: 0; }}
                }}
            </style>
            <script>
                // JavaScript code can be added here for more interactivity
            </script>
            """

    # Use write() with unsafe_allow_html=True to display the HTML
    html_placeholder.write(html_code, unsafe_allow_html=True)

    # Display the processed frame
    video_placeholder.image(frame, channels='BGR')

    if not end:
        # Storing the data to process
        frame_list.append(instructions[instruction_index])

    if end:
        break

out.release()

# Release the camera and close the app
cap.release()


# Load the pre-trained shape predictor model for facial landmarks
shape_predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)


# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Open the video file
video_path = 'output.mp4'
cap = cv2.VideoCapture(video_path)

ii = 0

reliability_list = []


def is_monotonic(numbers):
    increasing = decreasing = True

    for i in range(1, len(numbers)):
        if numbers[i] > numbers[i - 1]:
            decreasing = False
        elif numbers[i] < numbers[i - 1]:
            increasing = False

    return increasing or decreasing


def check_monotonicity_thumb(coordinate_vector: list, two_points_check: list):
    x_coordinate1 = [X.x for X in coordinate_vector]
    x_coordinate2 = [X.x for X in two_points_check]
    y_coordinate = [Y.y for Y in coordinate_vector]

    return is_monotonic(x_coordinate1) and is_monotonic(y_coordinate)


def check_monotonicity_fingers(coordinate_vector: list):
    y_coordinate = [Y.y for Y in coordinate_vector]

    return is_monotonic(y_coordinate)


def check_fingers(hand_landmarks):
    thumb = [mp_hands.HandLandmark.WRIST, mp_hands.HandLandmark.THUMB_CMC, mp_hands.HandLandmark.THUMB_MCP,
             mp_hands.HandLandmark.THUMB_IP, mp_hands.HandLandmark.THUMB_TIP]
    index_finger = [mp_hands.HandLandmark.WRIST, mp_hands.HandLandmark.INDEX_FINGER_MCP,
                    mp_hands.HandLandmark.INDEX_FINGER_PIP, mp_hands.HandLandmark.INDEX_FINGER_DIP,
                    mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_finger = [mp_hands.HandLandmark.WRIST, mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
                     mp_hands.HandLandmark.MIDDLE_FINGER_PIP, mp_hands.HandLandmark.MIDDLE_FINGER_DIP,
                     mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_finger = [mp_hands.HandLandmark.WRIST, mp_hands.HandLandmark.RING_FINGER_MCP,
                   mp_hands.HandLandmark.RING_FINGER_PIP, mp_hands.HandLandmark.RING_FINGER_DIP,
                   mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_finger = [mp_hands.HandLandmark.WRIST, mp_hands.HandLandmark.PINKY_MCP, mp_hands.HandLandmark.PINKY_PIP,
                    mp_hands.HandLandmark.PINKY_DIP, mp_hands.HandLandmark.PINKY_TIP]
    thumb_MCP = [mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_MCP, mp_hands.HandLandmark.MIDDLE_FINGER_MCP, mp_hands.HandLandmark.RING_FINGER_MCP, mp_hands.HandLandmark.PINKY_MCP]

    thumb_coor = [hand_landmarks.landmark[x] for x in thumb]
    thumb_coor_MCP = [hand_landmarks.landmark[x] for x in thumb_MCP]
    index_finge_coor = [hand_landmarks.landmark[x] for x in index_finger]
    middle_finger_coor = [hand_landmarks.landmark[x] for x in middle_finger]
    ring_finger_coor = [hand_landmarks.landmark[x] for x in ring_finger]
    pinky_finger_coor = [hand_landmarks.landmark[x] for x in pinky_finger]

    finger = 0
    if check_monotonicity_thumb(thumb_coor, thumb_coor_MCP):
        finger = finger + 1
    if check_monotonicity_fingers(index_finge_coor):
        finger = finger + 1
    if check_monotonicity_fingers(middle_finger_coor):
        finger = finger + 1
    if check_monotonicity_fingers(ring_finger_coor):
        finger = finger + 1
    if check_monotonicity_fingers(pinky_finger_coor):
        finger = finger + 1

    return finger


def eye_aspect_ratio(eye):
    # Compute the Euclidean distances between the two sets of vertical eye landmarks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # Compute the Euclidean distance between the horizontal eye landmarks
    C = dist.euclidean(eye[0], eye[3])

    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear


# Constants for EAR and blink detection
EAR_THRESHOLD = 0.22  # Adjust this threshold based on your conditions
CONSECUTIVE_FRAMES_BLINK = 2

blink_counter = 0

closed_eye = False

jj = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    ii = ii + 1

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = detector(gray_frame)

    # Loop over the faces
    for face in faces:
        # Use the shape predictor to get facial landmarks
        landmarks = predictor(gray_frame, face)
        # Extract coordinates for the eyes
        left_eye_coords = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
        right_eye_coords = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

        # Calculate EAR for each eye
        left_ear = eye_aspect_ratio(left_eye_coords)
        right_ear = eye_aspect_ratio(right_eye_coords)

        # Average EAR for both eyes
        ear = (left_ear + right_ear) / 2.0

        # Check for blinking
        if ear < EAR_THRESHOLD:
            blink_counter += 1
        else:
            blink_counter = 0

    # Check for consecutive blinks
    if blink_counter == CONSECUTIVE_FRAMES_BLINK:
        st.write("Blink detected!")
    elif blink_counter == 20:
        closed_eye = True
        st.write("Closed Eye detected")

    if closed_eye and (blink_counter == 0):
        closed_eye = False
        st.write("eye opened")


    # Process the image with MediaPipe Hands
    results = hands.process(rgb_frame)

    # Check if hand landmarks are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmarks for each finger tip

            # Check the y-coordinate of each finger tip
            finger_count = check_fingers(hand_landmarks)

            # if (finger_count != frame_list[ii]) & (jj%50 == 0):
            #     st.image(frame)
            #     st.write(finger_count)
            #     st.write(frame_list[ii])
            #     jj = jj + 1

            # Compare the no. of fingers in the given frame.
            if ii >= len(frame_list):
                ii = len(frame_list)-1
            if finger_count == frame_list[ii]:
                reliability_list.append(1)
            else:
                reliability_list.append(0)

            # Display the finger count
            cv2.putText(frame, f'Finger Count: {finger_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw landmarks on the hand
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame
    cv2.imshow('Hand Tracking', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

try:
    reliability_score = sum(reliability_list) / len(reliability_list)
    st.write(f"{reliability_score * 100}%")
except:
    reliability_score = 0
    st.write("len is zero")

# Release the VideoCapture and close all windows
cap.release()
cv2.destroyAllWindows()

if reliability_score > 0.8:
    st.write("Passed")
else:
    st.write("Failed")
