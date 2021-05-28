import matplotlib
from numpy.core.fromnumeric import shape
matplotlib.use("Agg")

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D

from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils, generic_utils

from keras import backend as K
K.set_image_data_format('channels_first')

import theano
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
#from sklearn import cross_validation
from sklearn import preprocessing

img_rows,img_cols,img_depth=224,224,120

X_tr=[] 

data = os.listdir('data')
for data_type in data:
    listing = os.listdir('data/'+data_type)
    for vid in listing:
        vid = 'data/'+data_type+'/'+vid
        frames = []
        cap = cv2.VideoCapture(vid)
        fps = cap.get(5)
        print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
 
        for k in range(120):
            ret, frame = cap.read()
            frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
            color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(color)

            #plt.imshow(gray, cmap = plt.get_cmap('gray'))
            #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
            #plt.show()
            #cv2.imshow('frame',gray)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

        input=np.array(frames)

        print(input.shape)
        ipt= np.rollaxis(np.rollaxis(input,2,0),2,0)
        ipt = np.rollaxis(ipt,3,0)
        print(ipt.shape)

        X_tr.append(ipt)


X_tr_array = np.array(X_tr)   # convert the frames read into array
num_samples = len(X_tr_array)
print(num_samples)

label=np.ones((num_samples,),dtype = int)
label[0:1]= 0
label[1:2] = 1
label[2:3] = 2

train_data = [X_tr_array,label]

(X_train, y_train) = (train_data[0],train_data[1])
print('X_Train shape:', X_train.shape)
#print(y_train)
#X_Train shape: (3, 16, 16, 15) (3, 3, 16, 16, 15)
#(3, 1, 16, 16, 15) train samples
train_set = X_train
 
patch_size = 15    # img_depth or number of frames used for each video

print(train_set.shape, 'train samples')

batch_size = 1
nb_classes = 3
nb_epoch = 5

Y_train = np_utils.to_categorical(y_train, nb_classes)

# number of convolutional filters to use at each layer
nb_filters = [32, 32]

# level of pooling to perform at each layer (POOL x POOL)
nb_pool = [3, 3]

# level of convolution to perform at each layer (CONV x CONV)
nb_conv = [5,5]

# Pre-processing

train_set = train_set.astype('float32')

train_set -= np.mean(train_set)

train_set /=np.max(train_set)


# Define model

model = Sequential()
model.add(Convolution3D(nb_filters[0], kernel_dim1=nb_conv[0], kernel_dim2=nb_conv[0], kernel_dim3=nb_conv[0],input_shape=(3, img_rows, img_cols, img_depth), activation='relu'))

model.add(MaxPooling3D(pool_size=(nb_pool[0], nb_pool[0], nb_pool[0])))

model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(128, init='normal', activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(nb_classes,init='normal'))

model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['acc'])


# Split the data

X_train_new, X_val_new, y_train_new,y_val_new =  train_test_split(train_set, Y_train, test_size=0.2, random_state=4)

#print(X_val_new.shape)
#print(y_val_new)
#print(X_train_new.shape)
#print(y_train_new)
# Train the model

H = model.fit(X_train_new, y_train_new, validation_data=(X_val_new,y_val_new),
          batch_size=batch_size,nb_epoch = nb_epoch,shuffle=True)


#hist = model.fit(train_set, Y_train, batch_size=batch_size,
#         nb_epoch=nb_epoch,validation_split=0.2, show_accuracy=True,
#           shuffle=True)


 # Evaluate the model
score = model.evaluate(X_val_new, y_val_new, batch_size=batch_size)
print('Test score:', score[0])
print('Test accuracy:', score[1])

N = nb_epoch
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")

print("[INFO] serializing network...")
model.save("model")




