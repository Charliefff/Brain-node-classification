import torch.nn as nn
import torch.nn.functional as F
from basemodel import BaseNet

class CNN(BaseNet):
    def __init__(self, **kwargs):
        super(CNN, self).__init__(**kwargs)
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.Chans, 16, (1, self.kernLength), padding=0),
            nn.BatchNorm2d(16, affine=False),
            nn.ReLU(),
            self.dropout(self.dropoutRate)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.wavelet, 16, (1, self.kernLength), padding=0),
            nn.BatchNorm2d(16, affine=False),
            nn.ReLU(),
            self.dropout(self.dropoutRate)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, (1, self.kernLength), stride=2, padding=0),
            nn.BatchNorm2d(16, affine=False),
            nn.ReLU(),
            self.dropout(self.dropoutRate)
        )

    def forward(self, x):
        # x = (batch, freq, sample, channel)
        x = x.permute(0, 3, 1, 2)  # x = (batch, channel, freq, sample) -> (batch, freq, sample, channel)
        x = self.conv1(x).permute(0, 2, 1, 3)  # x = (batch, freq, sample, channel) -> (batch, sample, freq, channel)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.flatten(start_dim=1)  # Flatten for FC
        self.build_fc(x, x.device)  # 動態初始化 Fully Connected 層
        return self.fc(x)

            
class CNN_3D(BaseNet):
    def __init__(self, **kwargs):
        super(CNN_3D, self).__init__(**kwargs)
        
        self.conv3d = nn.Sequential(
            nn.Conv3d(1, self.F1, (1, 1, self.kernLength), padding='same', bias=False),
            nn.BatchNorm3d(self.F1, affine=False),
            nn.ReLU(),
            nn.Dropout(self.dropoutRate),

            nn.Conv3d(self.F1, self.F1, (1, self.Chans, 1), bias=False),
            nn.BatchNorm3d(self.F1, affine=False),
            nn.ReLU(),
            nn.Dropout(self.dropoutRate),

            nn.Conv3d(self.F1, self.F1, (self.wavelet, 1, 1), bias=False),
            nn.BatchNorm3d(self.F1, affine=False),
            nn.ReLU(),
            nn.Dropout(self.dropoutRate)
        )
        
    def forward(self, x):
        x = x.unsqueeze(1)  # x = (batch, freq, sample, channel) -> (batch, 1, freq, sample, channel)
        x = x.permute(0, 1, 2, 4, 3)  # x = (batch, 1, freq, sample, channel) -> (batch, 1, freq, channel, sample)
        x = self.conv3d(x)
        x = x.flatten(start_dim=1)
        self.build_fc(x, x.device)
        return self.fc(x)

class waveletEGGNet(BaseNet):
    def __init__(self, **kwargs):
        super(waveletEGGNet, self).__init__(**kwargs)
        
        dropout = nn.Dropout2d if self.dropoutType == 'SpatialDropout2D' else nn.Dropout

        self.conv1 = nn.Sequential(
            nn.Conv2d(30, self.F1, (1, self.kernLength), padding='same', bias=False),
            nn.BatchNorm2d(self.F1),
            nn.Conv2d(self.F1, self.F1 * self.D, (self.Chans, 1), groups=self.F1, bias=False),
            nn.BatchNorm2d(self.F1 * self.D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            dropout(self.dropoutRate)
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(self.F1 * self.D, self.F2, (1, 16), padding='same', bias=False),
            nn.BatchNorm2d(self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            dropout(self.dropoutRate)
        )

    def forward(self, x):
        x = x.permute(0, 1, 3, 2)  # (batch, freq, sample, channel) -> (batch, freq, channel, sample)
        x = self.conv1(x)
        x = self.separableConv(x)
        x = x.flatten(start_dim=1)
        self.build_fc(x, x.device)
        
        return self.fc(x)

    
class EEGNet(BaseNet):
    def __init__(self,**kwargs):
        super(EEGNet, self).__init__(**kwargs)
        
        self.dropout = nn.Dropout2d if self.dropoutType == 'SpatialDropout2D' else nn.Dropout
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, self.kernLength), padding='same', bias=False),
            nn.BatchNorm2d(self.F1),
            nn.ELU(),
            nn.Conv2d(self.F1, self.F1 * self.D, (self.Chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(self.F1 * self.D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            self.dropout(self.dropoutRate)
        )

        self.separableConv = nn.Sequential(
            nn.Conv2d(self.F1 * self.D, self.F2, (1, 16), padding='same', bias=False),
            nn.BatchNorm2d(self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            self.dropout(self.dropoutRate)
        )

    def forward(self, x):
        x = x.unsqueeze(1).permute(0, 1, 3, 2)  # [B, C, T, H] -> [B, 1, H, T]
        x = self.conv1(x)
        x = self.separableConv(x)
        x = x.flatten(start_dim=1)
        self.build_fc(x, x.device)
        return x
    
class modelType(nn.Module):
    def __init__(self, model_type, **kwargs):
        super(modelType, self).__init__()
        
        model_mapping = {
            "EEGNet": EEGNet,
            "waveletEGGNet": waveletEGGNet,
            "CNN": CNN,
            "CNN_3D": CNN_3D
        }

        if model_type not in model_mapping:
            raise ValueError(
                f"Invalid model_type '{model_type}'. Available options are: {list(model_mapping.keys())}"
            )

        self.model = model_mapping[model_type](**kwargs)
        
    def forward(self, x):
        return self.model(x)
