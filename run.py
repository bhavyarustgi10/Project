from keras.models import model_from_json
import cv2
import numpy as np
import pickle as pkl
THRESHOLD = 0.7

#Saved Files
opt = 'adam'
modelJsonFile = "Models/"+"model"+opt+".json"
modelWeightsFile = "Models/"+"model"+opt+".h5"
historyFile = "Models/"+"history"+opt+".pickle"

with open("labels.pickle",'rb') as file_labels:
    labels = pkl.load(file_labels)
#Loading the Model
json_file = open(modelJsonFile, 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights(modelWeightsFile)

def getMyLabel(img):
    y = model.predict(img)
    i,j = np.where(np.isclose(y,y.max()))
    if y.max()>=THRESHOLD:
        return labels[j[0]]
    else:
        return "No Label Found"

#Read the Image and Reshape it to (80,88,3)
img = cv2.imread("MyDataset\Orange\img102.jpg")
img = np.resize(img,(80,88,3))
img = np.reshape(img,(1,80,88,3))
print(getMyLabel(img))
