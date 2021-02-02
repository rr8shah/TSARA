import os
import torch
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms
from PIL import Image, ImageFile
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import variance, mean, median, standard_deviation, sum, maximum
from skimage import io
from skimage.color import rgb2gray
from sklearn import preprocessing
import time
from sklearn.linear_model import LogisticRegression
from Model_Filter_Change import EONSS
from patch_generator import DenseSpatialCrop

X_test = ['test_true_10.png','test_fake_24.png']

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    return model
model = load_checkpoint('Model_Weights.pth')

criterion = torch.nn.BCELoss()

data = pd.read_csv('TrainData_Histogram_256stride_variance.csv')
X_train = data.iloc[:,:2]
y_train = data.iloc[:,2]
clf = LogisticRegression()
clf.fit(X_train,y_train)

class training_data(Dataset):
    def __init__(self, X_train, root_dir, transform):
        self.x = X_train
        self.path = root_dir
        self.transform = transform

    def __len__(self):
        return len(len(self.x))

    def __getitem__(self, index):
        img_name = os.path.join(self.path, self.x[index])
        image = Image.open(img_name)
        image = self.transform(image)
        return image

def Evaluation_test(model, dataloader_concat_val_data):
    model.eval()
    y_pred = []
    sig_out = []
    # set the requires_grad flag to false as we are in the test mode
    with torch.no_grad():
        for i, data in enumerate(dataloader_concat_val_data):
            # the model on the data
            #data = Variable(data)
            output = model(data)
            # PREDICTIONS
            sig_out.extend(output.cpu().reshape(-1).tolist())
            pred = np.round(output.cpu())
            y_pred.extend(pred.reshape(-1).tolist())
    return y_pred, sig_out

root_dir = 'E:/ECE 699/Real_Fake/NR-ISA/Test/'
test_transform = torchvision.transforms.Compose([DenseSpatialCrop(235,256)])
y_pred_test_full = []
zeros_test = []
ones_test = []
image_target_test = []
run_time = []
sr_no = []
counter = 0
k = 1 #Number of image required in one go

for i in range(0,len(X_test),k):
    t1 = time.time()
    test_data = []
    counter = counter + 1
    test_part = training_data(X_test[i:i+k], root_dir,test_transform)
    for i, data in enumerate (test_part):
        for batch_idx in range(0,data.size()[0]):
            if (variance(np.array(data[batch_idx][0])) > 0.00005 and variance(np.array(data[batch_idx][1])) > 0.00005 and variance(np.array(data[batch_idx][1])) > 0.00005):
                test_data.append(data[batch_idx])
    if len(test_data) != 0:
        concated_test_data = torch.stack(test_data, dim=0)
        dataloader_concat_test_data = torch.utils.data.DataLoader(concated_test_data, batch_size=32)
        y_pred_test, sig_out = Evaluation_test(model, dataloader_concat_test_data)
        uniqueValues, occurCount_feature = np.unique(y_pred_test, return_counts=True)
        if len(uniqueValues) == 1:
            if uniqueValues[0] == 0:
                ones_test.append(0)
                zeros_test.append(occurCount_feature[0])
            else:
                ones_test.append(occurCount_feature[0])
                zeros_test.append(0)
        else:
            zeros_test.append(occurCount_feature[0])
            ones_test.append(occurCount_feature[1])
        sr_no.append(counter)
df = pd.DataFrame(data={"sr_no":sr_no, "0": zeros_test, "1": ones_test})
X_test_logi = df.iloc[:,1:3]
pred_test = clf.predict(X_test_logi)

for i in range(0,len(pred_test)):
    if int(pred_test[i] == 1):
        print("Image: " + str(i+1) + " is True4K")
    else:
        print("Image: " + str(i+1) + " is Fake4K")
