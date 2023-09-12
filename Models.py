import torchvision
import torch.nn as nn
import torch


class NNet1(nn.Module):
    def __init__(self):
        super(NNet1, self).__init__()

        self.conv1 = nn.Conv2d(1, 128, kernel_size=(4,513))
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 1))
        self.conv2 = nn.Conv2d(128, 128, kernel_size=(4, 1))
        self.bn2 = nn.BatchNorm2d(128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 1))
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(4, 1))
        self.bn3 = nn.BatchNorm2d(256)
        self.avgpool = nn.AvgPool2d(kernel_size=(26, 1))
        self.maxpool = nn.MaxPool2d(kernel_size=(26, 1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.2)
        self.dense1 = nn.Linear(512, 300)
        self.bn4 = nn.BatchNorm1d(300)
        self.dense2 = nn.Linear(300,150)
        self.bn5 = nn.BatchNorm1d(150)
        self.dense3 = nn.Linear(150, 8)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.conv1(x.float())
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x_avg = self.avgpool(x)
        x_max = self.maxpool(x)
        x = torch.cat([x_avg, x_max], dim=1)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.dropout(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.dense3(x)
        x = self.softmax(x)
        return x
    
    
    
    
class NNet2(nn.Module):
    def __init__(self):
        super(NNet2, self).__init__()
        self.drop=nn.Dropout(0.2)
        # STFT spectrogram input: (batch_size, 1, 128, 513)
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(4, 513))
        self.batch1=nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128,128, kernel_size=(4, 1),padding=1)
        self.batch2=nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=(4, 1),padding=2)
        self.batch3=nn.BatchNorm2d(128)
        self.maxpool = nn.MaxPool2d(kernel_size=(128,1))
        self.avgpool = nn.AvgPool2d(kernel_size=(128,1))
        self.fc1 = nn.Linear(250,128)
        self.bn1=nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128,64)
        self.bn2=nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 8)  # 8 classes for genre predictions
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.conv1(x.float())
        x=self.batch1(x)
        x = torch.relu(x)
        y=x
        x = self.conv2(x)
        x=self.batch2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x=self.batch3(x)
        x = torch.relu(x)
        # Sum between the first and third conv layers
        x = x[:, :, :, 0] + y[:, :, :, 0]
        
        x = torch.relu(x)
        x_max = self.maxpool(x)
        x_avg = self.avgpool(x)
        x = torch.cat([x_avg, x_max], dim=1)
        # Flatten the tensor for fully connected layers
        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        x=self.drop(x)
        x=self.bn1(x)
        x = torch.relu(x)
        x = self.fc2(x)        
        x=self.drop(x)
        x=self.bn2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        
        return x



class NNet1_Small(nn.Module):
    def __init__(self):
        super(NNet1_Small, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=(2, 513))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        self.maxpool1 = nn.MaxPool2d(kernel_size=(4, 1))
        self.conv2 = nn.Conv2d(64,128, kernel_size=(2, 1))
        self.bn2 = nn.BatchNorm2d(128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(4, 1))
        self.conv3 = nn.Conv2d(128,64, kernel_size=(4, 1))
        self.bn3 = nn.BatchNorm2d(64)
        self.avgpool = nn.AvgPool2d(kernel_size=(2, 1))
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.3)
        self.dense1 = nn.Linear(256, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.dense3 = nn.Linear(64, 8)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.conv1(x.float())
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x_avg = self.avgpool(x)
        x_max = self.maxpool(x)
        x = torch.cat([x_avg, x_max], dim=1)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.dense3(x)
        x = self.softmax(x)
        return x



class NNet_Raw(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(NNet_Raw, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=16)
        self.conv2 = nn.Conv1d(32, 8, kernel_size=16)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=32)
        self.maxpool1 = nn.MaxPool1d(kernel_size=8)
        self.batchnorm1 = nn.BatchNorm1d(32)
        self.batchnorm2 = nn.BatchNorm1d(8)
        self.batchnorm3 = nn.BatchNorm1d(24)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(248, 24)
        self.fc3 = nn.Linear(24, 8)
        self.softmax=nn.Softmax(dim=1)
        
    def forward(self, x):
        x=self.maxpool1(x.float())
        x = self.conv1(x)
        x = self.relu(x)
        x = self.batchnorm1(x)
        x=self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.batchnorm2(x)     
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.batchnorm3(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x    

    
class Model_ResNet18(nn.Module):
    def __init__(self, pretrained=True):
        super(Model_ResNet18, self).__init__()
        pretrained_model = torchvision.models.resnet18(pretrained=pretrained)
        if(pretrained==True):
            layers=list(pretrained_model.children())[:-3]
            self.fc1 = nn.Linear(256, 128)
            for param in pretrained_model.parameters():
                param.requires_grad = False
        else:
            layers=list(pretrained_model.children())[:-5] 
            self.fc1 = nn.Linear(64, 128)
        #print(layers)
        self.features = nn.Sequential(*layers)
        
        
        
        self.pool= nn.AdaptiveAvgPool2d(1)
        self.flatten=nn.Flatten() 
        self.dropout=nn.Dropout(0.2)
        self.fc1 = nn.Linear(256, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3= nn.Linear(64, 8)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        
        x = self.features(x.float())
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x


class Ensemble(nn.Module):
    def __init__(self, load_weights=False):
        super(Ensemble, self).__init__()
        self.load_weights = load_weights 

        self.raw_net=NNet_Raw()
        self.stft_net=NNet1_Small()
        
        if(self.load_weights == True):
            self.raw_net.load_state_dict(torch.load("./best_models/models/NNet_Raw"), strict=False)
            self.stft_net.load_state_dict(torch.load("./best_models/models/NNet1_Small"), strict=False)
        
        self.dense1 = nn.Linear(16,16)
        self.bn = nn.BatchNorm1d(16)
        self.dense2 = nn.Linear(16, 8)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, input_value):
        
        #TODO:: REMOVE THESE FROM SCREENSHOT ------------------------------
        stft=input_value[0]
        stft=stft.unsqueeze(1)
        raw=input_value[1]
        raw=raw.unsqueeze(1)
        #-----------------------------------
    
        x_raw=self.raw_net(raw)
        x_stft=self.stft_net(stft)
        
        x = torch.cat([x_stft, x_raw], dim=1)
        x=self.bn(x)
        x=self.dense1(x)
        x=self.bn(x)
        x=self.dense2(x)
        x=self.softmax(x)
        
        return x