import numpy as np

import cv2

import os

from skimage.measure import label  

from sklearn.neighbors import KNeighborsClassifier

from skimage import img_as_ubyte 

from sklearn.datasets import fetch_mldata

from skimage.morphology import disk

from skimage.measure import regionprops 

#bbox[0] = min_row = x1,
#bbox[1] = min_col = y1,
#bbox[2] = max_row = x2,
#bbox[3] = max_col = y2,
#bbox[2] - bbox[0] = visina,
#bbox[3] - bbox[1] = sirina

def reshape(bbox, img):
    number = np.zeros((28, 28));
    for i in range(0, bbox[2]-bbox[0]):
        for j in range(0, bbox[3]-bbox[1]):
            number[i, j] = img[bbox[0]+i+1, bbox[1]+j+1]
    return number

def transform(mnistData):
    close_kernel = np.ones((5, 5), np.uint8)
    r = range(0, len(mnistData))
    for i in r:
        number = mnistData[i].reshape(28, 28)
        th = cv2.inRange(number, 150, 255)
        closing = cv2.morphologyEx(th, cv2.MORPH_CLOSE, close_kernel)
        labeled = label(closing)
        regions = regionprops(labeled)
        if(len(regions) > 1):
            max_width = 0
            max_height = 0
            for j in range(0, len(regions)):
                tempBbox = regions[j].bbox
                if(tempBbox[3] - tempBbox[1] > max_width and  tempBbox[2] - tempBbox[0] > max_height):
                    max_height = tempBbox[3] - tempBbox[1]
                    max_width = tempBbox[2] - tempBbox[0]
                    bbox = tempBbox
        else:
            bbox = regions[0].bbox
        img = np.zeros((28, 28))
        x = 0
        for w in range(bbox[0], bbox[2]):
            y = 0
            for h in range(bbox[1], bbox[3]):
                img[x, y] = number[w, h]
                y += 1
            x += 1
        mnistData[i] = img.reshape(1, 28*28)

def lineEq(x, y):
    line = x*y2 - x*y1 + y*x1 - y*x2 + x2*y1 - x1*y2
    return line
    
def lineIntersection(bbox):
    if(bbox[2]+4 < y2 or bbox[3]+4 < x1 or bbox[1] > x2):
        return False
    recEnd = lineEq(bbox[3]+1, bbox[2]+1)
    if(recEnd <= 0): 
        return False
    if(lineEq(bbox[1], bbox[0]) > 0 and lineEq(bbox[3]+4, bbox[0]) > 0):
        if(lineEq(bbox[1], bbox[2]+4) > 0 and lineEq(bbox[3]+4, bbox[2]+4) > 0):
            return False
    elif(lineEq(bbox[1], bbox[0]) < 0 and lineEq(bbox[3]+4, bbox[0]) < 0):
        if(lineEq(bbox[1], bbox[2]+4) < 0 and lineEq(bbox[3]+4, bbox[2]+4) < 0):
            return False
    else:
        return True


def checkNumber(brojevi,knn_num,bbox):   
    for clan in brojevi:
        #Ako se dati broj nalazi dijagonalno ispod trenutnog onda ga ignorisemo
        if(clan[0] == knn_num and clan[1] < bbox[1]+5  and clan[2] < bbox[0]+5 and clan[3] == bbox[3]-bbox[1]):
            brojevi.remove(clan)
            brojevi.append((knn_num, bbox[1], bbox[0], bbox[3]-bbox[1]))
            return False;
    brojevi.append((knn_num, bbox[1], bbox[0], bbox[3]-bbox[1]))


file = open('out.txt','w');

file.write('PIN BET\nfile sum\n')

str_elem_line = disk(2);
str_elem_number = disk(1);
mnist = fetch_mldata('MNIST original')  #skidanje baze rucno pisanih brojeva
train = mnist.data #train set 

transform(train)    

train_labels = mnist.target
knn = KNeighborsClassifier(n_neighbors=5, algorithm='auto').fit(train, train_labels)

for videoIndex in range(0,10):
    
    videoName = 'video-' + str(videoIndex) + '.avi';
    VIDEO_PATH= os.path.join(os.getcwd(), videoName)
    cap = cv2.VideoCapture(VIDEO_PATH);
    
    frameIndex = 0;
    suma = 0;
    brojevi = []
    print ('video-' + str(videoIndex));
    while (cap.isOpened()):
        #citanje framea
        ret, frame = cap.read();

        if ret != True:
            break
        else:
                    grayScale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY);
                    #pronalazenje linije Hough-transformacijom    
                    if(frameIndex == 0):
                        ret,thresh1 = cv2.threshold(grayScale,4,58,cv2.THRESH_BINARY)
                        erosion = cv2.erode(thresh1, str_elem_line, iterations=1)
                        byte = img_as_ubyte(erosion)
                        lines = cv2.HoughLinesP(byte, 1, np.pi/180, 100, minLineLength=1, maxLineGap=100)
                        #Koordinate prave
                        for line in lines[0]:
                            x1, y1, x2, y2 = line
                    #Izdvajanje regiona za brojeve
                    ret,thresh2 = cv2.threshold(grayScale,150,230,cv2.THRESH_BINARY)
                    thresh2_closing = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, str_elem_number)
                    labeled = label(thresh2_closing)
                    regions = regionprops(labeled)
                    
                    for region in regions:
                        bbox = region.bbox
                        if(bbox[2] - bbox[0] < 8): #visina<8
                            continue;
                        if(lineIntersection(bbox) == False):
                            continue;
                        #regiona na 28x28, pa na 1x784 - prima kao vektor
                        number = reshape(bbox,thresh2_closing)
                        knn_num = int(knn.predict(number.reshape(1,784)))
                        if(checkNumber(brojevi,knn_num,bbox)==False):
                            continue;
                        print 'Broj ' + str(knn_num);
        frameIndex = frameIndex + 2
     
    for clan in brojevi:
        suma = suma + clan[0]
    file.write(videoName + '\t' + str(suma) + '\n');

cap.release()
cv2.destroyAllWindows()
file.close();