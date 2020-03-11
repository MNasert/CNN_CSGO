import torch
import torch.nn as nn
from PIL import ImageGrab as ImageGrab
import numpy as np
import time
from pynput.mouse import Button, Controller
import sys
import pynput.keyboard
from pynput.keyboard import Listener
mouse=Controller()
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
model.load_state_dict(torch.load("model.mod"))
def quit(key):
    key=str(key)
    if key=="Key.end":
        sys.exit()
with Listener(on_press=quit) as listener:
    listener.join()


while(True):
    img=ImageGrab.grab().resize((1056,594))
    img=np.array(img)
    img=torch.FloatTensor(img)
    img=img.resize(1,3,594,1056)
    n=model(img)
    if n[0]-n[1]>0.35:
        print(n[0]-n[1])
        mouse.click(Button.left,1)