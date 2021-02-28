from keras.models import model_from_json
import cv2
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt 

opt = "adam"

#Saved Files
modelJsonFile = "Models/"+"modelMAIN"+opt+".json"
modelWeightsFile = "Models/"+"modelMAIN"+opt+".h5"
historyFile = "Models/"+"historyMAIN"+opt+".pickle"
#Loading the History
with open(historyFile,'rb') as file_history:
    history = pkl.load(file_history)
print(history)
plt.plot(history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
plt.show()