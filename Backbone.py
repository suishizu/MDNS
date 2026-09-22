import torch.nn as nn
import math
import torch
import torch.nn.functional as F

class Bottle2neck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, scale=3, is_Double=is_Double):

        super(Bottle2neck, self).__init__()
        self.groups = planes//scale 
        self.len = planes//scale

        if is_Double == True:
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride)
    
            self.bn1 = nn.BatchNorm2d(planes)
            self.dropout1 = nn.Dropout(0.1)
    
            if self.len > 0:
                # DSconv
                self.conv2 = nn.Conv2d(planes//scale, planes//scale, kernel_size=3, groups=planes//scale, padding=1) # 深度卷积
                self.bn2 = nn.BatchNorm2d(planes//scale)
                self.conv3 = nn.Conv2d(planes//scale, planes//scale, kernel_size=1, stride=1)   # 点卷积
                self.bn3 = nn.BatchNorm2d(planes//scale)
    
            self.relu = nn.ReLU(inplace=True)
    
            self.scale = scale
            self.shortcut = None
    
            if stride != 1 or inplanes != planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride),
                )
            self.resnet18 = nn.Sequential(
                nn.Conv2d(planes, planes, kernel_size=3, padding=1),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True),
                nn.Conv2d(planes, planes, kernel_size=3, padding=1),
                nn.BatchNorm2d(planes),
            )

        else:
            self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, stride=stride)
    
            self.bn1 = nn.BatchNorm1d(planes)
            self.dropout1 = nn.Dropout(0.1)
    
            if self.len > 0:
                # DSconv
                self.conv2 = nn.Conv1d(planes//scale, planes//scale, kernel_size=3, groups=planes//scale, padding=1) # 深度卷积
                self.bn2 = nn.BatchNorm1d(planes//scale)
                self.conv3 = nn.Conv1d(planes//scale, planes//scale, kernel_size=1, stride=1)   # 点卷积
                self.bn3 = nn.BatchNorm1d(planes//scale)
    
            self.relu = nn.ReLU(inplace=True)
    
            self.scale = scale
            self.shortcut = None
    
            if stride != 1 or inplanes != planes:
                self.shortcut = nn.Sequential(
                    nn.Conv1d(inplanes, planes, kernel_size=1, stride=stride),
                )
            self.resnet18 = nn.Sequential(
                nn.Conv1d(planes, planes, kernel_size=3, padding=1),
                nn.BatchNorm1d(planes),
                nn.ReLU(inplace=True),
                nn.Conv1d(planes, planes, kernel_size=3, padding=1),
                nn.BatchNorm1d(planes),
            )
          
  

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        # out = self.relu(out)

        if self.scale < 0:
            out = self.resnet18(out)

        else:
            if self.scale > 1:
                spx = torch.split(out, self.len, dim=1) 
                side = self.conv2(spx[1])
                side = self.conv3(side)
                z = torch.cat((spx[0], side), 1)

            else:
                z = self.conv2(out)
                z = self.conv3(z)

            if self.scale > 2:
                for i in range(2, self.scale):
                    sp = side + spx[i]
                    y = self.conv2(sp)
                    y = self.conv3(y)
                    side = y
                    z = torch.cat((z,y),1)
            out = z

        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = out + residual
        return out
    

class Res2Net(nn.Module):

    def __init__(self, block, layers, scale, num_classes, is_Double=is_Double):
        super(Res2Net, self).__init__()

        self.inplanes = 32
        self.scale = scale 

        if is_Double == True:
            self.conv = nn.Conv2d(2, 32, kernel_size=7, stride=2, padding=3)
        else:
            self.conv = nn.Conv1d(2, 32, kernel_size=7, stride=2, padding=3)

        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 8, layers[0], stride=1, is_Double=is_Double)
        self.layer2 = self._make_layer(block, 16, layers[1], stride=1, is_Double=is_Double)
        self.layer3 = self._make_layer(block, 32, layers[2], stride=1, is_Double=is_Double)
        self.layer4 = self._make_layer(block, 64, layers[3], stride=1, is_Double=is_Double)

        if is_Double == True:
            self.avg = nn.AdaptiveAvgPool2d((8, 1))
            self.max = nn.AdaptiveMaxPool2d((8, 1))
    
            self.fin_conv = nn.Sequential(
                nn.BatchNorm2d(128),
                nn.Conv2d(128, 64, kernel_size=1),
                nn.ReLU(),
    
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 64 ,kernel_size=1)
            )

        else:
            self.avg = nn.AdaptiveAvgPool1d(8)
            self.max = nn.AdaptiveMaxPool1d(8)
    
            self.fin_conv = nn.Sequential(
                nn.BatchNorm1d(128),
                nn.Conv1d(128, 64, kernel_size=1),
                nn.ReLU(),
    
                nn.BatchNorm1d(64),
                nn.Conv1d(64, 64 ,kernel_size=1)
            )

          


    def _make_layer(self, block, planes, blocks, stride=1, is_Double=is_Double):

        layers = []
        layers.append(block(self.inplanes, planes, stride, scale=self.scale, is_Double))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, scale=self.scale, is_Double))
        return nn.Sequential(*layers)

    def forward(self, x):      
        x = self.conv(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x_avg = self.avg(x)
        x_max = self.max(x)
        
        x = torch.cat((x_avg, x_max), dim=1)
        x = self.fin_conv(x).flatten(1)

        return x
    
def my_resnet(is_Double):
    # is_Double == True:  二维数据
    # is_Double == False: 一维数据
    model = Res2Net(Bottle2neck, [2, 2, 2, 2], 4, 7, is_Double=is_Double)
    return model
