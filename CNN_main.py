import PIL.ImageGrab as ImageGrab
from PIL import Image
import torch
import glob
#import torch.functional as F
import torch.nn as nn
import numpy as np
import torchvision
import datetime
from sklearn.utils import shuffle
from pynput.mouse import Listener
import os
from tqdm import tqdm as tqdm
#np.set_printoptions(threshold=999999999999999)

def record():
    def take_pic(x,y,button,pressed):
        picture="screen"+str(np.random.randint(0,999999))+".png"
        n=ImageGrab.grab()
        n=n.resize((1056,594))
        n.save(picture,format="PNG")


    with Listener(on_click=take_pic) as listener:
        listener.join()
x=[]
y=[]
for infile in glob.glob("data/*.png"):
    os.path.splitext(infile)
    print(infile)
    im = Image.open(infile)
    x.append(im)
for i in range(len(x)):
    x[i]=np.array(x[i])
x=torch.FloatTensor(x)
x=x.reshape(148,1,3,594,1056)
for i in range(42):
    y.append([0,1])
for i in range(106):
    y.append([1,0])
print(y)
y=torch.FloatTensor(y)

print(y)
class CNN_Bot(nn.Sequential):
    def __init__(self):
        super(CNN_Bot, self).__init__()
        self.conv1= nn.Conv2d(3,16,12)
        self.max1=nn.MaxPool2d([583,1045])
        self.fc1  = nn.Linear(in_features=16,out_features=16)
        self.fc2 = nn.Linear(16,2)
    def forward(self, x):
        x =  nn.functional.sigmoid(self.conv1(x))
        x=self.max1(x)
        x=x.reshape([16])
        x=self.fc1(x)
        x=nn.functional.softmax(self.fc2(x))

        return x
model=CNN_Bot()


def train(model, in_x, out_y, optimizer, criterion, epochs):

    running_loss = 0.0
    for epoch in tqdm(range(epochs)):
        in_x, out_y = shuffle(in_x, out_y)
        for i in (range(len(in_x))):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output = model.forward(in_x[i])
            loss = criterion(output, out_y[i])
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 20 == 19:  # print every 100 mini-batches

                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0


train(model,x,y,torch.optim.Adam(model.parameters(),lr=1e-5),torch.nn.MSELoss(),20)
torch.save(model.state_dict(), './model.mod')
