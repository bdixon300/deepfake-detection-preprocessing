import dlib
import cv2
from PIL import Image
from imutils import face_utils, resize

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

df = pd.DataFrame({'frame': [], 'label': []})
path = ''
label = 0

for filename in os.listdir(path):
    vidcap = cv2.VideoCapture(filename)
    success = True

    frame_count = 0
    frame_sequence = 0
    frame_sequence_count = 0

    while success:
        success,image = vidcap.read()
        # Only extract first 30 seconds of video
        if not success or frame_count >= 720:
            break
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = detector(gray, 0)

        largest_face_size = 0
        for (i, face) in enumerate(faces):
            # Make the prediction and transfom it to numpy array
            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)
            size = face.width() * face.height()
            if largest_face_size < size:
                largest_face_size = size

                # Mouth region uses these indices for dlib
                (i, j) = (48, 68)
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
              frame_file_name = '{}_frame_{}_{}.jpg'.format(filename[:4], frame_sequence, frame_sequence_count)
              df = df.append({'frame': frame_file_name, 'label': label}, ignore_index=True)
              cv2.imwrite('frames_mouths/' + frame_file_name, roi)
              frame_count += 1
              frame_sequence = int(frame_count / 20)
              frame_sequence_count = int(frame_count % 20)

    if os.path.isfile('labels.csv'):
        print("appending to csv")
        df.to_csv('labels.csv', header=None, mode='a')
    else:
        print("Creating new csv")
        df.to_csv('labels.csv')

