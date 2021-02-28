import cv2
import os
import tensorflow as tf
import numpy as np
from keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Dropout,Flatten,MaxPooling2D
import pickle as pkl
def load_images_from_folder(location):
    images = []
    for filename in os.listdir(location):
        img = cv2.imread(os.path.join(location,filename))
        if img is not None:
            img = np.resize(img,(80,88,3))
            images.append(img)
    return images
def load_data_set(location):
    labels = {}
    currentLabel = 0
    trainX = []
    trainY = []
    for folder in os.listdir(location):
        #We have the folder name as folder
        labels[currentLabel] = folder
        imagesOfLabel = load_images_from_folder(os.path.join(location,folder))
        trainX = trainX+imagesOfLabel
        trainY = trainY+[currentLabel]*(len(imagesOfLabel))
        currentLabel+=1 
    trainX = np.array(trainX) 
    trainY = np.array(trainY)
    trainY = to_categorical(trainY)
    return trainX,trainY,labels

def preprocessTheImages(trainX):
    # converting datatype to float32
    #print("----------------------",trainX.dtype)
    trainX = trainX.astype('float32')
    #print("----------------------",trainX.dtype)
    # normalizing into range (0,255)
    trainX = trainX/255.0
    return trainX

def createModel(trainX,trainY) :
    model = Sequential()
    model.add(Conv2D(100, kernel_size=(3,3), input_shape=(trainX.shape[1],trainX.shape[2],trainX.shape[3])))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(100,kernel_size=(3,3),activation='relu'))
    model.add(Flatten()) 
    model.add(Dense(150, activation=tf.nn.relu))
    model.add(Dense(trainY.shape[1],activation=tf.nn.softmax))
    return model


trainX,trainY,labels = load_data_set("MyDataset")
trainX = preprocessTheImages(trainX)
print("Data Set Size ",trainY.shape[0])
model = createModel(trainX,trainY)
opt = 'adam'
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])

history = model.fit(x=trainX,y=trainY, epochs=45)

modelJsonFile = "Models/"+"modelMAIN"+opt+".json"
modelWeightsFile = "Models/"+"modelMAIN"+opt+".h5"
historyFile = "Models/"+"historyMAIN"+opt+".pickle"
"""Saving the Class Labels"""
with open('labels.pickle', 'wb') as file_labels:
    pkl.dump(labels,file_labels)

"""Saving the Model"""
model_json = model.to_json()
with open(modelJsonFile, "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(modelWeightsFile)

"""Saving the History of Model"""
with open(historyFile, 'wb') as file_history:
    pkl.dump(history.history,file_history)
