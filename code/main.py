from genericpath import exists
import os
import argparse
import datetime
import cv2
import numpy as np
import pickle as pkl

from sklearn.ensemble import RandomForestRegressor

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data

from ITrackerModel import ITrackerModel
from Dataset import Dataset

#Arguments for program
def parse_args():
    parser = argparse.ArgumentParser(prog='main.py', description='Project') 
    parser.add_argument('--datapath', default='./', type=str, required=True,help='Data path')
    parser.add_argument('--train', default=False, type=bool, required=False,help='train model or not')
    parser.add_argument('--test', default=False, type=bool, required=False,help='test model or not')
    parser.add_argument('--produceTrain', default=False, type=bool, required=False,help='prodeuce the train data or not')
    parser.add_argument('--produceTest', default=False, type=bool, required=False,help='prodeuce the test data or not')

    return parser.parse_args()

class FaceDetector():

    def __init__(self,detectionFilePath,size = (224,224)):
        #Load Face detection caffemodel
        prototxt = f"{detectionFilePath}/deploy.prototxt"
        caffemodel = f"{detectionFilePath}/res10_300x300_ssd_iter_140000.caffemodel"
        self.net = cv2.dnn.readNetFromCaffe(prototxt=prototxt, caffeModel=caffemodel)
        self.size = size

    #Get face detection bounding box
    def detect(self,img, min_confidence=0.5):
        (h, w) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detectors = self.net.forward()
        rects = []
        for i in range(0, detectors.shape[2]):
            confidence = detectors[0, 0, i, 2]
            if confidence < min_confidence:
                continue
            box = detectors[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x0, y0, x1, y1) = box.astype("int")
            rects.append({"box": (x0, y0, x1 - x0, y1 - y0), "confidence": confidence})

        return rects

    #Get face image and facegrid array
    def faceDetection(self,imgPath):
        try:
            face = cv2.imread(imgPath)
            rects = self.detect(face)
            faceGrid = np.zeros((len(face),len(face[0])),dtype=np.uint8)
            for rect in rects:
                (x, y, w, h) = rect["box"]
                face = face[y:y+h, x:x+w]
                faceGrid[y:y+h, x:x+w] = 1
            face = cv2.resize(face,self.size)
            faceGrid = cv2.resize(faceGrid,(25,25))
            faceGrid = np.reshape(faceGrid,(1,625))[0]
            return face,faceGrid
        except:
            return None,None
    #Same as faceDetection but use cv2 image as input
    def faceDetection2(self,face):
        try:
            rects = self.detect(face)
            faceGrid = np.zeros((len(face),len(face[0])),dtype=np.uint8)
            for rect in rects:
                (x, y, w, h) = rect["box"]
                face = face[y:y+h, x:x+w]
                faceGrid[y:y+h, x:x+w] = 1
            face = cv2.resize(face,self.size)
            faceGrid = cv2.resize(faceGrid,(25,25))
            faceGrid = np.reshape(faceGrid,(1,625))[0]
            return face,faceGrid
        except:
            return None,None
    #Get and save left and right eye images
    def getEye(self,img):
        
        rx = 5
        lx = 120
        y = 70
        w = 90
        h = 40
        leye = cv2.resize(img[y:y+h,lx:lx+w],(224,224))
        reye = cv2.resize(img[y:y+h,rx:rx+w],(224,224))    
        return leye,reye    
        
class ImageProcess():

    def __init__(self):
        pass

    #Rename images in 'images' folder for further process
    def renameImages(self,path,begin):
        dirs = os.listdir(path)
        counter = 0
        for i in dirs:
            os.rename(f'{path}/{i}',path + '/IMG%05d.jpeg' % (counter + begin,))
            counter += 1
    #Get image paths
    def getImagePaths(self,path):
        dirs = os.listdir(path)
        paths = []
        for i in dirs:
            paths.append(f'{path}/{i}')
        return paths
    #Save face images and facegrids for training or testing
    def getFeatures(self,path,dependencyFilesPath):
        imagePath = os.path.join(path,'images')
        facePath = os.path.join(path,'face')
        faceGridPath = os.path.join(path,'faceGrid')

        faceDetection = FaceDetector(dependencyFilesPath)
        imgPaths = self.getImagePaths(path = imagePath)
        
        for i,imgPath in enumerate(imgPaths):
            print(i)
            imgPath = str(imgPath)
            face,faceGrid = faceDetection.faceDetection(imgPath)
            faceName = f'{facePath}/'+'face%05d.jpeg'%(i,)
            gridName = f'{faceGridPath}/'+'faceGrid%05d.pkl'%(i,)
            try:
                cv2.imwrite(faceName,face)
                pkl.dump(faceGrid,open(gridName,'wb'))
            except:        
                faceDirs = os.listdir(facePath)
                faceGridDirs = os.listdir(faceGridPath)
                if('face%05d.jpeg'%(i,) in faceDirs):
                    os.remove(faceName)
                if('faceGrid%05d.pkl'%(i,) in faceGridDirs):
                    os.remove(gridName)

    #Get and save left and right eye images
    def getEye(self,path):
        facePath = os.path.join(path,'face')
        leftEyePath = os.path.join(path,'leftEye')
        rightEyePath = os.path.join(path,'rightEye')

        facePaths = [i for i in os.listdir(facePath)]
        rx = 5
        lx = 120
        y = 70
        w = 90
        h = 40
        
        for i in facePaths:
            img = cv2.imread(f'{facePath}/{i}')
            leye = cv2.resize(img[y:y+h,lx:lx+w],(224,224))
            cv2.imwrite(f'{leftEyePath}/leye{i[4:]}',leye)
            
        for i in facePaths:
            img = cv2.imread(f'{facePath}/{i}')
            reye = cv2.resize(img[y:y+h,rx:rx+w],(224,224))        
            cv2.imwrite(f'{rightEyePath}/reye{i[4:]}',reye)
    #Produce labels for training(please rewrite this code for your own data)
    def genlabel():
        frames = 10
        pass

#check if necessary folders is exist
def checkExist(path):
    folderNames = ['images','face','leftEye','rightEye','faceGrid']
    trainDataPath = os.path.join(path,'train')
    testDataPath = os.path.join(path,'test')
    for i in folderNames:
        folderName = os.path.join(trainDataPath,i)
        if not exists(folderName):
            os.mkdir(folderName)
        folderName = os.path.join(testDataPath,i)
        if not exists(folderName):
            os.mkdir(folderName)
    logPath = 'log'
    if not exists(logPath):
        os.mkdir(logPath)

#transfer learning(use both train data and test data)     
def train(trainData,testData,workers,epochs,batch_size):

    logName = datetime.datetime.now().strftime("%Y%m%d%H%M%S") + '.pkl'

    minLoss = 99999
    
    losses = []

    #load the base pre-train model(model name is 'model')
    model = ITrackerModel()
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    model.load_state_dict(torch.load("model"))

    #Build train data loader
    train_loader = torch.utils.data.DataLoader(
        trainData,
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)
    #Build test data loader
    test_loader = torch.utils.data.DataLoader(
        testData,
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)
    #Set training parameter
    lr = 0.00001
    momentum = 0.9
    weight_decay = 1e-4    
    criterion = nn.MSELoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)

    for epoch in range(epochs): 
        print('Epoch: {}'.format(epoch))
        for imFace, imEyeL, imEyeR, faceGrid, gaze in train_loader:
            imFace = imFace.cuda()
            imEyeL = imEyeL.cuda()
            imEyeR = imEyeR.cuda()
            faceGrid = faceGrid.cuda()
            gaze = gaze.cuda()
            
            imFace = torch.autograd.Variable(imFace, requires_grad = True)
            imEyeL = torch.autograd.Variable(imEyeL, requires_grad = True)
            imEyeR = torch.autograd.Variable(imEyeR, requires_grad = True)
            faceGrid = torch.autograd.Variable(faceGrid, requires_grad = True)
            gaze = torch.autograd.Variable(gaze, requires_grad = False)

            # compute output
            output = model(imFace, imEyeL, imEyeR, faceGrid)

            loss = criterion(output, gaze)
            
            print(loss)
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        testOriginErr,testCalibrateErr = test(model,test_loader=test_loader)
        losses.append({'testOriginErr':testOriginErr,'testCalibrateErr':testCalibrateErr})
        if(testCalibrateErr < minLoss):
            minLoss = testCalibrateErr
            torch.save(model.state_dict(), 'transferlearning_model')
        with open(logName,'wb') as f:
            pkl.dump(losses,f)
#test the model(only use test data)
def test(model,test_loader):

    outputs = []
    gazes = []

    #Get original predict result from transferlearning
    for imFace, imEyeL, imEyeR, faceGrid, gaze in test_loader:

        imFace = imFace.cuda()
        imEyeL = imEyeL.cuda()
        imEyeR = imEyeR.cuda()
        faceGrid = faceGrid.cuda()
        gaze = gaze.cuda()
        
        imFace = torch.autograd.Variable(imFace, requires_grad = True)
        imEyeL = torch.autograd.Variable(imEyeL, requires_grad = True)
        imEyeR = torch.autograd.Variable(imEyeR, requires_grad = True)
        faceGrid = torch.autograd.Variable(faceGrid, requires_grad = True)
        gaze = torch.autograd.Variable(gaze, requires_grad = False)

        # compute output
        output = model(imFace, imEyeL, imEyeR, faceGrid)

        output = output.detach().cpu().numpy()
        gaze = gaze.detach().cpu().numpy()
        
        outputs.extend(output)
        gazes.extend(gaze)

    
    outputs = np.array(outputs)
    gazes = np.array(gazes)

    original_err = ((outputs[:,0] - gazes[:,0]) ** 2 + (outputs[:,1] - gazes[:,1]) ** 2) ** 0.5

    #Fit calibration ML model
    clfx = RandomForestRegressor(random_state=0)
    clfy = RandomForestRegressor(random_state=0)

    clfx.fit(outputs,gazes[:,[0]])
    clfy.fit(outputs,gazes[:,[1]])

    #Save calibration model
    with open('clfx.pkl', 'wb') as f:
        pkl.dump(clfx, f)
    with open('clfy.pkl', 'wb') as f:
        pkl.dump(clfy, f)

    prex = np.array(clfx.predict(outputs))
    prey = np.array(clfy.predict(outputs))

    prex = np.reshape(prex,(-1,1))
    prey = np.reshape(prey,(-1,1))

    calibrate = np.concatenate([prex,prey],axis = 1)

    calibrate_err = ((calibrate[:,0] - gazes[:,0]) ** 2 + (calibrate[:,1] - gazes[:,1]) ** 2) ** 0.5

    
    original_err = np.sum(original_err)/len(outputs)
    calibrate_err = np.sum(calibrate_err)/len(outputs)
    
    print(f'{original_err}\t{calibrate_err}\t{len(outputs)}')

    return original_err,calibrate_err

def main():
    isGetTrainData = args.produceTrain
    isGetTestData = args.produceTest
    isTrain = args.train
    isTest = args.test
    datapath = args.datapath

    trainDataPath = os.path.join(datapath,'train')
    testDataPath = os.path.join(datapath,'test')    
    dependencyFilesPath = 'dependencyfiles'

    checkExist(path=datapath)

    imgProcess = ImageProcess()
    if(isGetTrainData):
        imgProcess.getFeatures(path=trainDataPath,dependencyFilesPath = dependencyFilesPath)
        imgProcess.getEye(path=trainDataPath)
    if(isGetTestData):
        imgProcess.getFeatures(path=testDataPath,dependencyFilesPath = dependencyFilesPath)
        imgProcess.getEye(path=testDataPath)
    
    if(isTrain):
        #Get trainData
        trainData = Dataset(dataPath = trainDataPath,dependencyFilesPath=dependencyFilesPath)
        testData = Dataset(dataPath = testDataPath,dependencyFilesPath=dependencyFilesPath)
        train(trainData=trainData,testData=testData,workers=3,epochs=60,batch_size=64)
    if(isTest):
        #Get testData
        #load the base tranfer learning model(model name is 'transferlearning_model')
        model = ITrackerModel()
        model = torch.nn.DataParallel(model)
        model = model.cuda()
        model.load_state_dict(torch.load("transferlearning_model"))
        testData = Dataset(dataPath = testDataPath,dependencyFilesPath=dependencyFilesPath)
        test_loader = torch.utils.data.DataLoader(
            testData,
            batch_size=64, shuffle=True,
            num_workers=3, pin_memory=True)
        test(model=model,test_loader=test_loader)
        
if __name__ == '__main__':
    args = parse_args()
    main()