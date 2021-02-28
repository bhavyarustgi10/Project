import cv2
import os 

def getVideoSnaps(location,folder):
    cam = cv2.VideoCapture(location) 
    currentframe = 0
    while(True): 
            # reading from frame 
            ret,frame = cam.read() 
            print("Hello")
            if ret: 
                name = './MyDataset/'+folder+'/img'+str(currentframe) + '.jpg'
                cv2.imwrite(name, frame) 
                currentframe += 1
            else: 
                break
    cam.release() 
    cv2.destroyAllWindows() 

def getVideos(location):
    for video in os.listdir(location):
        #We have the folder name as folder
        getVideoSnaps(os.path.join(location,video),video[:len(video)-4])

getVideos("FruitVideo")
