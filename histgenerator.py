import os
import torch
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms
from PIL import Image, ImageFile
from Model_Filter_Change import EONSS
from patch_generator import DenseSpatialCrop
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plot
from scipy.ndimage import variance, mean, median, standard_deviation, sum, maximum
from skimage import io
from skimage.color import rgb2gray
from sklearn import preprocessing

train = pd.read_csv('data/train_4K.csv')
test = pd.read_csv('data/test_4K.csv')

X = train['image_name']
y = train['score'].values

X_test = test['image_name']
y_test = test['score'].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state = 42)

X_train = X_train.tolist()
y_train = torch.Tensor(y_train)
X_val = X_val.tolist()
y_val = torch.Tensor(y_val)
X_test = X_test.tolist()
y_test = torch.Tensor(y_test)

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    return model

model = load_checkpoint('pretrained_model/Model_Weights.pth')

criterion = torch.nn.BCELoss()
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()

class training_data(Dataset):
    def __init__(self, X_train, y_train, root_dir, transform):
        self.x = X_train
        self.y = y_train
        self.path = root_dir
        self.transform = transform

    def __len__(self):
        return len(len(self.x))

    def __getitem__(self, index):
        img_name = os.path.join(self.path, self.x[index])
        image = Image.open(img_name)
        image = self.transform(image)
        y = self.y[index]
        return image, y

# ## Train and Validation Evaluation function
def Evaluation_train(model, dataloader_concat_val_data, dataloader_concat_val_label):
    model.eval()
    y_true = []
    y_pred = []
    sig_out = []
    # set the requires_grad flag to false as we are in the test mode
    with torch.no_grad():
        for i, (data, label) in enumerate(zip(dataloader_concat_val_data, dataloader_concat_val_label)):
            # the model on the data
            data = Variable(data)
            if torch.cuda.is_available():
                data = data.cuda()

            output = model(data)
            # PREDICTIONS
            sig_out.extend(output.cpu().reshape(-1).tolist())
            pred = np.round(output.cpu())
            target = label
            y_true.extend(target.tolist())
            y_pred.extend(pred.reshape(-1).tolist())
    return y_true, y_pred, sig_out


# ## Test Evaluation function
def Evaluation_test(model, dataloader_concat_val_data, dataloader_concat_val_label):
    model.eval()
    y_true = []
    y_pred = []
    sig_out = []
    # set the requires_grad flag to false as we are in the test mode
    with torch.no_grad():
        for i, (data, label) in enumerate(zip(dataloader_concat_val_data, dataloader_concat_val_label)):
            # the model on the data
            data = Variable(data)
            if torch.cuda.is_available():
                data = data.cuda()

            output = model(data)
            sig_out.extend(output.cpu().reshape(-1).tolist())
            pred = np.round(output.cpu())
            target = label
            y_true.extend(target.tolist())
            y_pred.extend(pred.reshape(-1).tolist())
    return y_true, y_pred, sig_out


# ## Train Prediction and Histogram Generation
root_dir = '/project/6002585/rrshah/Real_Vs_Fake4K_Dataset/Train/'  #Replace this directory adress to your local dataset adress
test_transform = torchvision.transforms.Compose([DenseSpatialCrop(235,256)])
k = 1 #Number of image required in one go
y_true_train_full = []
y_pred_train_full = []
zeros = []
ones = []
image_target = []
counter = 0
for i in range(0,len(X_train),k):
    train_data = []
    train_label = []
    counter = counter + 1
    train_part = training_data(X_train[i:i+k], y_train[i:i+k], root_dir,test_transform)
    for i, (data, feature) in enumerate (train_part):
        for batch_idx in range(0,data.size()[0]):
            if (variance(np.array(data[batch_idx][0])) > 0.00005 and variance(np.array(data[batch_idx][1])) > 0.00005 and variance(np.array(data[batch_idx][1])) > 0.00005):
                train_data.append(data[batch_idx])
                train_label.append(feature)
    if len(train_data) != 0:
        concated_train_data = torch.stack(train_data, dim=0)
        concated_train_label = torch.stack(train_label, dim=0)
        dataloader_concat_train_data = torch.utils.data.DataLoader(concated_train_data, batch_size=32)
        dataloader_concat_train_label = torch.utils.data.DataLoader(concated_train_label, batch_size=32)
        y_true_train, y_pred_train, sig_out = Evaluation_train(model, dataloader_concat_train_data, dataloader_concat_train_label)
        uniqueValues, occurCount_feature = np.unique(y_pred_train, return_counts=True)
        if len(uniqueValues) == 1:
            if uniqueValues[0] == 0:
                ones.append(0)
                zeros.append(occurCount_feature[0])
            else:
                ones.append(occurCount_feature[0])
                zeros.append(0)
        else:
            zeros.append(occurCount_feature[0])
            ones.append(occurCount_feature[1])
        image_target.append(y_true_train[0])
        y_pred_train_full.extend(y_pred_train)
        y_true_train_full.extend(y_true_train)
print("Accuracy on Training set with patches is" , accuracy_score(y_true_train_full,y_pred_train_full))
df = pd.DataFrame(data={"0": zeros, "1": ones, 'target':image_target})
df.to_csv("histogram_data/TrainData_Histogram_256stride_variance.csv", sep=',',index=False)


# ## Validation Prediction and Histogram Generation
root_dir = '/project/6002585/rrshah/Real_Vs_Fake4K_Dataset/Train/' #Replace this directory adress to your local dataset adress
test_transform = torchvision.transforms.Compose([DenseSpatialCrop(235,256)])
k = 1 #Number of image required in one go
y_true_val_full = []
y_pred_val_full = []
zeros = []
ones = []
image_target = []
counter = 0
for i in range(0,len(X_val),k):
    val_data = []
    val_label = []
    counter = counter + 1
    val_part = training_data(X_val[i:i+k], y_val[i:i+k], root_dir,test_transform)
    for i, (data, feature) in enumerate (val_part):
        for batch_idx in range(0,data.size()[0]):
            if (variance(np.array(data[batch_idx][0])) > 0.00005 and variance(np.array(data[batch_idx][1])) > 0.00005 and variance(np.array(data[batch_idx][1])) > 0.00005):
                val_data.append(data[batch_idx])
                val_label.append(feature)
    if len(val_data) != 0:
        concated_val_data = torch.stack(val_data, dim=0)
        concated_val_label = torch.stack(val_label, dim=0)
        dataloader_concat_val_data = torch.utils.data.DataLoader(concated_val_data, batch_size=32)
        dataloader_concat_val_label = torch.utils.data.DataLoader(concated_val_label, batch_size=32)
        y_true_val, y_pred_val, sig_out = Evaluation_train(model, dataloader_concat_val_data, dataloader_concat_val_label)
        uniqueValues, occurCount_feature = np.unique(y_pred_val, return_counts=True)
        if len(uniqueValues) == 1:
            if uniqueValues[0] == 0:
                ones.append(0)
                zeros.append(occurCount_feature[0])
            else:
                ones.append(occurCount_feature[0])
                zeros.append(0)
        else:
            zeros.append(occurCount_feature[0])
            ones.append(occurCount_feature[1])
        image_target.append(y_true_val[0])
        y_pred_val_full.extend(y_pred_val)
        y_true_val_full.extend(y_true_val)
print("Accuracy on Validation set with patches is" , accuracy_score(y_true_val_full,y_pred_val_full))
df = pd.DataFrame(data={"0": zeros, "1": ones, 'target':image_target})
df.to_csv("histogram_data/ValData_Histogram_256stride_variance.csv", sep=',',index=False)


# ## Test Prediction and Histogram Generation
root_dir = '/project/6002585/rrshah/Real_Vs_Fake4K_Dataset/Test/'   #Replace this directory adress to your local dataset adress
test_transform = torchvision.transforms.Compose([DenseSpatialCrop(235,256)])
y_true_test_full = []
y_pred_test_full = []
zeros_test = []
ones_test = []
image_target_test = []
sr_no = []
counter = 0
k = 1 #Number of image required in one go
for i in range(0,len(X_test),k):
    test_data = []
    test_label = []
    counter = counter + 1
    test_part = training_data(X_test[i:i+k], y_test[i:i+k], root_dir,test_transform)
    for i, (data, feature) in enumerate (test_part):
        for batch_idx in range(0,data.size()[0]):
            if (variance(np.array(data[batch_idx][0])) > 0.00005 and variance(np.array(data[batch_idx][1])) > 0.00005 and variance(np.array(data[batch_idx][1])) > 0.00005):
                test_data.append(data[batch_idx])
                test_label.append(feature)
    if len(test_data) != 0:
        concated_test_data = torch.stack(test_data, dim=0)
        concated_test_label = torch.stack(test_label, dim=0)
        dataloader_concat_test_data = torch.utils.data.DataLoader(concated_test_data, batch_size=32)
        dataloader_concat_test_label = torch.utils.data.DataLoader(concated_test_label, batch_size=32)
        y_true_test, y_pred_test, sig_out = Evaluation_test(model, dataloader_concat_test_data, dataloader_concat_test_label)
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
        image_target_test.append(y_true_test[0])
        sr_no.append(counter)
        y_pred_test_full.extend(y_pred_test)
        y_true_test_full.extend(y_true_test)
print("Accuracy on Testing set with patches is" , accuracy_score(y_true_test_full,y_pred_test_full))
df = pd.DataFrame(data={"sr_no":sr_no, "0": zeros_test, "1": ones_test, 'target':image_target_test})
df.to_csv("histogram_data/TestData_Histogram_256stride_variance_srno.csv", sep=',',index=False)
