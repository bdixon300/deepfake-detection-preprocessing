import dlib
import cv2
import pandas as pd
import os
from PIL import Image
from imutils import face_utils, resize
import numpy as np
import shutil


"""
FACIAL_LANDMARKS_IDXS
	("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 35)),
	("jaw", (0, 17))
"""

# Setup and load face detection/facial landmark detection
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

# Setup dataframe for generating csv of labels for sequences/frames
df = pd.DataFrame({'sequence': [], 'label': [], 'landmarks': []})

# Paths to videos for preprocessing: SET THIS TO THE CORRECT PATH
path = ""
#path = '../../dataset/FaceForensicsDatasetMouthFiltered/training/real_new/'
#path = '../../dataset/FaceForensicsDatasetMouthFiltered/training/manipulated_sequences/'
#path = '../../dataset/FaceForensicsDatasetMouthFiltered/training/original_sequences/'
#path = '../../dataset/FaceForensicsDatasetMouthFiltered/validation/manipulated_sequences/'
#path = '../../dataset/FaceForensicsDatasetMouthFiltered/validation/original_sequences/'
#path = '../../dataset/FaceForensicsDatasetMouthFiltered/testing/manipulated_sequences/'
#path = '../../dataset/FaceForensicsDatasetMouthFiltered/testing/original_sequences/'

# Label for the sequences in preprocessing, 0 is Fake, whereas 1 is real
label = 0
video_number = 1

# Iterate over videos
for filename in os.listdir(path):
    print("Generating faces from: {}, video_number: {}".format(filename, video_number))

    # Iterate only over select range, so program can be run concurrently, preprocessing selected ranges in the dataset
    if  not(video_number < 21 and video_number >= 1):
        print("skipping video: {}".format(filename))
        video_number += 1
        continue

    # Extract frames
    vidcap = cv2.VideoCapture(path + filename)
    success = True

    # Counts used to generate filenames for sequences/frames
    frame_count = 0
    frame_sequence = 0
    frame_sequence_count = 0

    # Temporary list of eye landmarks to append to csv
    eye_landmarks = []

    while success:
        success,image = vidcap.read()
        # Only extract first 30 seconds of video
        if not success or frame_count >= 720:
            break
        # Extract faces
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = detector(gray, 0)
        # Move to next frame if no faces were detected
        if len(faces) == 0:
            continue
        # Generate sequence directories to store the frames
        sequence_directory = filename[:-4] + '_{}/'.format(frame_sequence)
        if not os.path.exists('frames_faces/{}'.format(sequence_directory)): 
            # Create new row
            os.mkdir('frames_faces/{}'.format(sequence_directory))
            df = df.append({'sequence': sequence_directory[:-1], 'label': label, 'landmarks': []}, ignore_index=True)

        largest_face_size = 0
        largest_shape = []

        # Only extract the largest face in the frame of the video
        for (i, face) in enumerate(faces):
            # Make the prediction and transfom it to numpy array
            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)
            size = face.width() * face.height()
            if largest_face_size < size:
                largest_shape = shape
                largest_face_size = size
                (x, y, w, h) = face_utils.rect_to_bb(face)
                face_boundary = image[y:y+h, x:x+w]
                roi = cv2.resize(face_boundary, (224,224))
        # Extract eye landmarks for lstm use
        eye_landmarks.append(largest_shape[36:48].flatten().tolist())
        
        # Create face area image files
        frame_file_name = 'frame_{}.jpg'.format(frame_sequence_count)
        cv2.imwrite('frames_faces/' + sequence_directory + frame_file_name, roi)
        
        # Update counts
        frame_count += 1
        frame_sequence = int(frame_count / 20)
        frame_sequence_count = int(frame_count % 20)

        # Append eye area data to csv
        if len(eye_landmarks) == 20:
            # Add landmark data to dataframe
            df.at[df.index[-1], 'landmarks'] = eye_landmarks
            eye_landmarks = []     

    video_number += 1

    # Remove sequences that are not exactly 20 frames in length
    if  frame_sequence_count < 19:
        shutil.rmtree('frames_faces/{}'.format(sequence_directory))
        df = df.drop(df[df["sequence"] == sequence_directory[:-1]].index)

    # Append to exiting CSV
    if os.path.isfile('labels.csv'):
        print("appending to csv")
        df.to_csv('labels.csv', header=None, mode='a', index=False)
        df = pd.DataFrame({'sequence': [], 'label': [], 'landmarks': []})
    # Append to new CSV
    else:
        print("Creating new csv")
        df.to_csv('labels.csv', index=False)
        df = pd.DataFrame({'sequence': [], 'label': [], 'landmarks': []})
