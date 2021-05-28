import matplotlib
matplotlib.use("Agg")
from keras.models import load_model
from collections import deque
import numpy as np
import pickle
import cv2
 
size = 128
print("[INFO] loading model and label binarizer...")
model = load_model("model")
#lb = pickle.loads(open("label_bin", "rb").read())
with open("label_bin", "rb") as fp:
    lb = pickle.load(fp)
#print(lb.classes_)
mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
Q = deque(maxlen=size)

vs = cv2.VideoCapture("data//soccer//soccer.mp4")
writer = None
(W, H) = (None, None)
 
while True:
	(grabbed, frame) = vs.read()

	if not grabbed:
		break
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	output = frame.copy()
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame = cv2.resize(frame, (16, 16)).astype("float32")
	#frame -= mean

	preds = model.predict(np.expand_dims(frame, axis=0))[0]
	Q.append(preds)

	results = np.array(Q).mean(axis=0)
	i = np.argmax(results)
	label = lb[i]

	text = "activity: {}".format(label)
	cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,
		1.25, (0, 255, 0), 5)

	if writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter("output", fourcc, 30,
			(W, H), True)

	writer.write(output)

	cv2.imshow("output", output)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break

cv2.destroyAllWindows() 
