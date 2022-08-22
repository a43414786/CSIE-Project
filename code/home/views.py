from re import X
from django.shortcuts import redirect, render ,HttpResponse
from bs4 import BeautifulSoup
import requests
from .form import UploadModelForm
from .models import Photo
from PIL import Image
import json

from urllib.request import urlretrieve
import numpy as np

import scipy.io as sio
from PIL import Image
import os
import os.path
from os.path import exists
import torch
import pickle as pkl
import cv2

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

from ITrackerModel import ITrackerModel
from django.http import JsonResponse

from main import FaceDetector
from Dataset import SubtractMean,loadMetadata

trainDataPath = '../data/train/images'
testDataPath = '../data/test/images'

class Preprocess:
    def __init__(self,dependencyFilesPath = 'dependencyfiles',imSize=(224,224)):

        self.faceMean = loadMetadata(os.path.join(dependencyFilesPath,'mean_face_224.mat'))['image_mean']
        self.eyeLeftMean = loadMetadata(os.path.join(dependencyFilesPath,'mean_left_224.mat'))['image_mean']
        self.eyeRightMean = loadMetadata(os.path.join(dependencyFilesPath,'mean_right_224.mat'))['image_mean']
        
        #Normalize
        self.transformFace = transforms.Compose([
            transforms.Resize(imSize),
            transforms.ToTensor(),
            SubtractMean(meanImg=self.faceMean),
        ])
        self.transformEyeL = transforms.Compose([
            transforms.Resize(imSize),
            transforms.ToTensor(),
            SubtractMean(meanImg=self.eyeLeftMean),
        ])
        self.transformEyeR = transforms.Compose([
            transforms.Resize(imSize),
            transforms.ToTensor(),
            SubtractMean(meanImg=self.eyeRightMean),
        ])

    def loadImage(self, img):
        try:
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))    
        except OSError:
            raise RuntimeError('Could not get image: ')
        
        return img

    def process(self, face,leye,reye,faceGrid):
        
        imFace = self.loadImage(face)
        imEyeL = self.loadImage(leye)
        imEyeR = self.loadImage(reye)

        imFace = self.transformFace(imFace).unsqueeze(0)
        imEyeL = self.transformEyeL(imEyeL).unsqueeze(0)
        imEyeR = self.transformEyeR(imEyeR).unsqueeze(0)
        faceGrid = torch.FloatTensor(faceGrid).unsqueeze(0)
    

        return imFace, imEyeL, imEyeR, faceGrid

t = os.listdir(trainDataPath)
img = Image.open(os.path.join(trainDataPath,t[0]))

model = ITrackerModel()
model = torch.nn.DataParallel(model)
model = model.cuda()
model.load_state_dict(torch.load("model"))

clfx = pkl.load(open('clfx.pkl', 'rb'))
clfy = pkl.load(open('clfy.pkl', 'rb'))

preprocess = Preprocess()
faceDetector = FaceDetector('dependencyfiles')

def predict(img):
    img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
    cv2.imshow('img',img)
    cv2.waitKey()
    face,facegrid = faceDetector.faceDetection2(img)
    leye,reye = faceDetector.getEye(face)
    face,leye,reye,facegrid = preprocess.process(face,leye,reye,facegrid)
    output = model(face, leye, reye, facegrid)
    output = output.detach().cpu().numpy()
    x = clfx.predict(output)
    y = clfy.predict(output)
    x = 258 - x * 60
    y = 18 - y * 60
    return x[0],y[0]
    
x_label = [215,265,305,335,360,375,385,390,390,390,385,375,360,335,305,265,215,165,125,95,70,55,45,40,40,40,45,55,70,95,125,165]
y_label = [25,40,75,115,160,210,260,305,350,395,440,490,540,585,625,660,675,660,625,585,540,490,440,395,350,305,260,210,160,115,75,40]

counter = 0
getTrainImage = False
x = 0
y = 0

def test(request):
    global counter,x,y,getTrainImage
    if request.method == "POST":
        img = Image.open(request.FILES['image'])
        counter += 1
        if(getTrainImage):
            img.save('IMG{:05d}.jpeg'.format(counter))
        
        if(not getTrainImage):
            data = {
                    'x':x,
                    'y':y
                    }

            idx = int(request.POST['timer'])

            try:
                prex,prey = predict(img)
            except:
                return JsonResponse(data)

            err = (((prex - x_label[idx]) ** 2 + (prey - y_label[idx]) ** 2) ** 0.5)/60

            print(err)

            if(isReturned):
                return JsonResponse(data)

            if(err < 0.7):
                isReturned = 1
                x = prex
                y = prey
                data = {
                        'x':prex,
                        'y':prey
                        }
    return JsonResponse({})
