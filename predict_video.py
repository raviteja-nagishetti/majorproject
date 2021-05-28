import matplotlib
matplotlib.use("Agg")
from keras.models import load_model
from collections import deque
import numpy as np
import pickle
import cv2
 
print("[INFO] loading model")
model = load_model("model")
CLASSES = open("action_label.txt").read().strip().split("\n")

#mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")

writer = None
(W, H) = (None, None)
cap = cv2.VideoCapture("data//tennis//tennis.mp4")
fps = cap.get(5)
print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
img_rows,img_cols,img_depth=224,224,120
frames = []

for k in range(120):
    ret, frame = cap.read()
    frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
    color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(color)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

input=np.array(frames)

print(input.shape)
ipt= np.rollaxis(np.rollaxis(input,2,0),2,0)
ipt = np.rollaxis(ipt,3,0)
print(ipt.shape)

prediction = model.predict(np.expand_dims(ipt, axis=0))[0]
label = CLASSES[np.argmax(prediction)]
print(label)

# loop over our frames
while True:
    for frame in frames:
        # draw the predicted activity on the frame
        cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
        cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,0.8, (255, 255, 255), 2)

        # display the frame to our screen
        cv2.imshow("Activity Recognition", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

