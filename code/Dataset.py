import torch.utils.data as data
import scipy.io as sio
from PIL import Image
import os
import os.path
import torchvision.transforms as transforms
import torch
import pickle as pkl
import cv2

MEAN_PATH = './'

def loadMetadata(filename, silent = False):
    try:
        # http://stackoverflow.com/questions/6273634/access-array-contents-from-a-mat-file-loaded-using-scipy-io-loadmat-python
        if not silent:
            print('\tReading metadata from %s...' % filename)
        metadata = sio.loadmat(filename, squeeze_me=True, struct_as_record=False)
    except:
        print('\tFailed to read the meta file "%s"!' % filename)
        return None
    return metadata

class SubtractMean(object):
    """Normalize an tensor image with mean.
    """

    def __init__(self, meanImg):
        self.meanImg = transforms.ToTensor()(meanImg / 255)

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """       
        return tensor.sub(self.meanImg)

class Dataset(data.Dataset):
    def __init__(self,dataPath = './',dependencyFilesPath = './',imSize=(224,224), gridSize=(25, 25)):

        # self.label = []
        # with open('label.pkl','rb') as f:
        #     self.label = pkl.load(f)
        
        facePath = os.path.join(dataPath,'face')
        leftEyePath = os.path.join(dataPath,'leftEye')
        rightEyePath = os.path.join(dataPath,'rightEye')
        faceGridPath = os.path.join(dataPath,'faceGrid')

        #Get features paths
        self.facePaths = [os.path.join(facePath,i) for i in os.listdir(facePath)]
        self.leyePaths = [os.path.join(leftEyePath,i) for i in os.listdir(leftEyePath)]
        self.reyePaths = [os.path.join(rightEyePath,i) for i in os.listdir(rightEyePath)]
        self.gridPaths = [os.path.join(faceGridPath,i) for i in os.listdir(faceGridPath)]

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

    def loadImage(self, path):
        try:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # img[:,:,0] = 255
            # img[:,:,1] = 255
            # img[:,:,2] = 255
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
        except OSError:
            raise RuntimeError('Could not read image: ' + path)
            #im = Image.new("RGB", self.imSize, "white")
        return img

    def __getitem__(self, index):
        
        imFacePath = self.facePaths[index]
        imEyeLPath = self.leyePaths[index]
        imEyeRPath = self.reyePaths[index]
        with open(self.gridPaths[index],"rb") as f:
            faceGrid = pkl.load(f)
        # gaze = ((258 - self.label[index][0])/60,(18 - self.label[index][1])/60)
        gaze = (0,0)
        imFace = self.loadImage(imFacePath)
        imEyeL = self.loadImage(imEyeLPath)
        imEyeR = self.loadImage(imEyeRPath)

        imFace = self.transformFace(imFace)
        imEyeL = self.transformEyeL(imEyeL)
        imEyeR = self.transformEyeR(imEyeR)


        # to tensor
        faceGrid = torch.FloatTensor(faceGrid)
        gaze = torch.FloatTensor(gaze)
    

        return imFace, imEyeL, imEyeR, faceGrid, gaze
    
        
    def __len__(self):
        return len(self.facePaths)

    