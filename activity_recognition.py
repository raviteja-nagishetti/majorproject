import matplotlib
from numpy.core.fromnumeric import shape
matplotlib.use("Agg")
from keras.utils import np_utils
from keras.layers import Conv3D, MaxPool3D, Flatten, Dense
from keras.layers import Dropout, Input, BatchNormalization

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D

from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils, generic_utils

from tensorflow import keras
from tensorflow.keras import layers

from keras import backend as K
#K.set_image_data_format('channels_first')

import theano
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
#from sklearn import cross_validation
from sklearn import preprocessing

img_rows,img_cols,img_depth=128,128,64

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
 
        for k in range(64):
            ret, frame = cap.read()
            frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
            color = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
        #ipt = np.rollaxis(ipt,3,0)
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

train_set = np.zeros((num_samples, img_rows, img_cols, img_depth, 1))

for h in range(num_samples):
    for i in range(128):
        for j in range(128):
            for k in range(64):
                train_set[h][i][j][k][0] = X_train[h][i][j][k]
 
patch_size = 64    # img_depth or number of frames used for each video

print(train_set.shape, 'train samples')

batch_size = 1
nb_classes = 3
nb_epoch = 2

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


def get_model(width=128, height=128, depth=64):
    """Build a 3D convolutional neural network model."""

    input_layer = Input((128, 128, 64, 1))
    
    conv_layer1 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu')(input_layer)
    pooling_layer1 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer1)

    pooling_layer1 = BatchNormalization()(pooling_layer1)  
    conv_layer2 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu')(pooling_layer1)
    pooling_layer2 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer2)
    pooling_layer2 = BatchNormalization()(pooling_layer2)
    conv_layer3 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu')(pooling_layer1)
    pooling_layer3 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer3)
    pooling_layer3 = BatchNormalization()(pooling_layer3)
    conv_layer4 = Conv3D(filters=128, kernel_size=(3, 3, 3), activation='relu')(pooling_layer3)
    pooling_layer4 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer4)
    pooling_layer4 = BatchNormalization()(pooling_layer4)
    conv_layer5 = Conv3D(filters=256, kernel_size=(3, 3, 3), activation='relu')(pooling_layer4)
    pooling_layer5 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer5)
    
    pooling_layer9 = BatchNormalization()(pooling_layer5)
    flatten_layer = Flatten()(pooling_layer9)
    
    dense_layer3 = Dense(units=512, activation='relu')(flatten_layer)
    dense_layer3 = Dropout(0.4)(dense_layer3)

    dense_layer4 = Dense(units=256, activation='relu')(dense_layer3)
    dense_layer4 = Dropout(0.4)(dense_layer3)
  
    output_layer = Dense(units=num_samples, activation='softmax')(dense_layer4)

    model = Model(inputs=input_layer, outputs=output_layer)

    model.compile(loss='mae', optimizer=SGD(lr=1e-06, momentum=0.99, decay=0.0, nesterov=False), metrics=['acc']) 
    
    return model

# Build model.
model = None
model = get_model()
model.summary()


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




