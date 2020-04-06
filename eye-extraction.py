import dlib
import cv2
import pandas as pd
import os
from PIL import Image
from imutils import face_utils, resize
import numpy as np
import shutil

"""FACIAL_LANDMARKS_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 35)),
	("jaw", (0, 17))
])"""


p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

df = pd.DataFrame({'sequence': [], 'label': []})
#path='test_videos_fake/'
#path = '../../dataset/FaceForensicsDatasetMouthFiltered/training/real_new/'
#path = '../../dataset/FaceForensicsDatasetMouthFiltered/training/manipulated_sequences/'
#path = '../../dataset/FaceForensicsDatasetMouthFiltered/training/original_sequences/'
#path = '../../dataset/FaceForensicsDatasetMouthFiltered/validation/manipulated_sequences/'
path = '../../dataset/FaceForensicsDatasetMouthFiltered/validation/original_sequences/'
#path = '../../dataset/FaceForensicsDatasetMouthFiltered/testing/manipulated_sequences/'
#path = '../../dataset/FaceForensicsDatasetMouthFiltered/testing/original_sequences/'
#path = '../../dataset/FaceForensicsDatasetMouthFilteredFull/training/original_sequences/'
#path = '../../dataset/FaceForensicsDatasetMouthFilteredFull/training/manipulated_sequences/'
#path = '../../dataset/FaceForensicsDatasetMouthFilteredFull/validation/original_sequences/'
#path = '../../dataset/FaceForensicsDatasetMouthFilteredFull/validation/manipulated_sequences/'
#path = '../../dataset/FaceForensicsDatasetMouthFilteredFull/testing/manipulated_sequences/'
#path = '../../dataset/FaceForensicsDatasetMouthFilteredFull/testing/original_sequences/'

label = 1
video_number = 1

for filename in os.listdir(path):

    # Execute preprocessing on one video
    """if  filename != "11__outside_talking_pan_laughing.mp4":
        print("skipping video: {}".format(filename))
        video_number += 1
        continue"""

    # Skip first videos not in range if required
    # Training
    # 1 - 304
    # 304 - 608
    # 608 - 916

    # 1 - 35
    #35 - 70
    # 70 - 105

    # Validation
    # 1 - 50
    # 50 - 100
    # 100 - 150
    # 150 - 196

    if  not(video_number < 10 and video_number >= 1):
        print("skipping video: {}".format(filename))
        video_number += 1
        continue

    print("Generating eyes from: {}, video_number: {}".format(filename, video_number))
    vidcap = cv2.VideoCapture(path + filename)
    success = True

    frame_count = -1
    frame_sequence = 0
    frame_sequence_count = 0

    while success:
        success,image = vidcap.read()
        # Only extract first 30 seconds of video
        if not(success) or frame_count >= 720:
            break

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = detector(gray, 0)

        # Move to next frame if no faces were detected
        if len(faces) == 0:
            continue
        else:
            frame_count += 1
            frame_sequence = int(frame_count / 20)
            frame_sequence_count = int(frame_count % 20)

        sequence_directory = filename[:-4] + '_{}/'.format(frame_sequence)
        if not os.path.exists('frames_eyes/{}'.format(sequence_directory)):
            os.mkdir('frames_eyes/{}'.format(sequence_directory))
        df = df.append({'sequence': sequence_directory[:-1], 'label': label}, ignore_index=True)
        df.drop_duplicates(inplace=True)

        largest_face_size = 0
        for (i, face) in enumerate(faces):
            # Make the prediction and transfom it to numpy array
            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)
            size = face.width() * face.height()
            if largest_face_size < size:
                largest_face_size = size

                # Eye region uses these indices for dlib
                (i, j) = (36, 48)
                # clone the original image so we can draw on it, then
                # display the name of the face part on the image
                clone = image.copy()

                # loop over the subset of facial landmarks, drawing the
                # specific face part
                for (x, y) in shape[i:j]:                    
                    cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
                    (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
                    roi = image[y:y + h, x:x + w]
                    roi = cv2.resize(roi, (224,224))
        
        frame_file_name = 'frame_{}.jpg'.format(frame_sequence_count)
        cv2.imwrite('frames_eyes/' + sequence_directory + frame_file_name, roi)
    video_number += 1

    # Remove sequences that are not exactly 20 frames in length
    if  frame_sequence_count < 19:
        shutil.rmtree('frames_eyes/{}'.format(sequence_directory))
        df = df.drop(df[df["sequence"] == sequence_directory[:-1]].index)

    # Append video sequences to csv
    if os.path.isfile('labels.csv'):
        print("appending to csv")
        pd.read_csv('labels.csv').append(df).drop_duplicates().to_csv('labels.csv', index=False)
    else:
        print("Creating new csv")
        df.to_csv('labels.csv', index=False)
    df = pd.DataFrame({'sequence': [], 'label': []})
        
