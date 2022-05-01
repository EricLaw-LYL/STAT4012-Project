import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
import tensorflow as tf

# Import the models
detector = MTCNN()
# trained_model = tf.keras.models.load_model("emotion_detection/src/model/trained_model.h5", compile = False)
trained_model = tf.keras.models.load_model("emotion_detection/src/model/CNN_model.h5", compile = False)

# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

# dictionary which assigns each label an emotion (alphabetical order)
# emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprise"}
# IMG_SIZE = (96, 96)
IMG_SIZE = (48, 48)

# Import video
video = cv2.VideoCapture("emotion_detection/video/Will_Smith.mp4")

# We need to check if camera
# is opened previously or not
if (video.isOpened() == False):
    print("Error reading video file")

# We need to set resolutions.
# so, convert them from float to integer.
frame_width = int(video.get(3))
frame_height = int(video.get(4))

size = (frame_width, frame_height)

# Below VideoWriter object will create
# a frame of above defined The output
result = cv2.VideoWriter("emotion_detection/video/Will_Smith_demo.mp4", cv2.VideoWriter_fourcc(*'MJPG'), 29, size)
frame_num=0
while (True):
    # if frame_num >= 200:
    #     break
    
    ret, frame = video.read()
    frame_num += 1
    print(frame_num)
    
    if ret == True:
        location = detector.detect_faces(frame)
        if len(location) > 0:
            for face in location:
                x, y, width, height = face['box']
                x2, y2 = x + width, y + height
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 4)
                
                face_ = frame[y:y2, x:x2]
                cropped_img = cv2.resize(face_, IMG_SIZE)
                cropped_img_expanded = np.expand_dims(cropped_img, axis = 0)
                cropped_img_float = cropped_img_expanded.astype(float)
                prediction = trained_model.predict(cropped_img_float)
                print(prediction)
                maxindex = int(np.argmax(prediction))
                cv2.putText(frame, emotion_dict[maxindex], (x + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
        result.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break


video.release()
result.release()

# Closes all the frames
cv2.destroyAllWindows()

print("The video was successfully saved")