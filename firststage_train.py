from Model import EONSS
from patch_generator import DenseSpatialCrop
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
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve


# Class geberating the batches of the data
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

# Function resposible for training of the model
def Training (model, dataloader_concat_train_data, dataloader_concat_train_label):
    y_true = []
    y_pred = []
    for i, (data,label) in enumerate(zip(dataloader_concat_train_data, dataloader_concat_train_label)):
        assert not torch.isnan(data).any()
        model.train()
        data = Variable(data)
        label = Variable(label)
        label = label.float()
        if torch.cuda.is_available():
            data = data.cuda()
            label = label.cuda()
        running_loss = 0.0
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs,label)
        loss.backward()
        optimizer.step()
        pred = np.round(outputs.cpu().detach())
        target = np.round(label.cpu().detach())
        y_pred.extend(pred.tolist())
        y_true.extend(target.tolist())
    return y_true, y_pred

# Function predicting the patch level prediction
def Evaluation(model, dataloader_concat_val_data, dataloader_concat_val_label):
    model.eval()
    y_true = []
    y_pred = []
    # set the requires_grad flag to false as we are in the test mode
    with torch.no_grad():
        for i, (data, label) in enumerate(zip(dataloader_concat_val_data, dataloader_concat_val_label)):
            # the model on the data
            data = Variable(data)
            if torch.cuda.is_available():
                data = data.cuda()

            output = model(data)
            # PREDICTIONS
            pred = np.round(output.cpu())
            target = label
            y_true.extend(target.tolist())
            y_pred.extend(pred.reshape(-1).tolist())
    return y_true, y_pred


train = pd.read_csv('data/train_4K.csv')

X = train['image_name']
y = train['score'].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state = 42)

X_train = X_train.tolist()
y_train = torch.Tensor(y_train)
X_val = X_val.tolist()
y_val = torch.Tensor(y_val)

# Initializing the model and required loss function and Optimizer
model = EONSS()
criterion = torch.nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()

root_dir = '/project/6002585/rrshah/Real_Vs_Fake4K_Dataset/Train/'        #To be Replaced by the train dataset local path
train_transform = torchvision.transforms.Compose([DenseSpatialCrop(235,256)])

k = 20 #Number of image required in one go
for a in range (0,10):
  y_true_full = []
  y_pred_full = []
  for i in range(0,len(X_train), k):
      train_data = []
      train_label = []
      train_part = training_data(X_train[i:i+k], y_train[i:i+k], root_dir,train_transform)
      for i, (data, feature) in enumerate (train_part):
          for batch_idx in range(0,data.size()[0]):
              train_data.append(data[batch_idx])
              train_label.append(feature)
      concated_train_data = torch.stack(train_data, dim=0)
      concated_train_label = torch.stack(train_label, dim=0)
      dataloader_concat_train_data = torch.utils.data.DataLoader(concated_train_data, batch_size=32)
      dataloader_concat_train_label = torch.utils.data.DataLoader(concated_train_label, batch_size=32)
      y_true, y_pred = Training (model, dataloader_concat_train_data, dataloader_concat_train_label)
      y_pred_full.extend(y_pred)
      y_true_full.extend(y_true)
  print("Accuracy on training set is" , accuracy_score(y_true_full,y_pred_full))

checkpoint = {'model': EONSS(),
          'state_dict': model.state_dict(),
          'optimizer' : optimizer.state_dict()}

torch.save(checkpoint, 'pretrained_model/Model_Weights.pth')
