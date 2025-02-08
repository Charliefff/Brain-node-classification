import torch
import torch.nn as nn
import torch.nn.functional as F
from basemodel import BaseNet
from braindecode.models import EEGNetv4, EEGNetv1, Deep4Net
from torch.nn.utils import weight_norm
# from EEGGENet import EEGGENet
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

    def forward(self, x, attention=False):
        # x = (batch, freq, sample, channel)
        x = x.permute(0, 3, 1, 2)  # x = (batch, channel, freq, sample) -> (batch, freq, sample, channel)
        x = self.conv1(x).permute(0, 2, 1, 3)  # x = (batch, freq, sample, channel) -> (batch, sample, freq, channel)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.flatten(start_dim=1)  # Flatten for FC
        self.build_fc(x, x.device)  # 動態初始化 Fully Connected 層
        return self.fc(x), None

            
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
        
    def forward(self, x, attention=False):
        x = x.unsqueeze(1)  # x = (batch, freq, sample, channel) -> (batch, 1, freq, sample, channel)
        x = x.permute(0, 1, 2, 4, 3)  # x = (batch, 1, freq, sample, channel) -> (batch, 1, freq, channel, sample)
        x = self.conv3d(x)
        x = x.flatten(start_dim=1)
        self.build_fc(x, x.device)
        return self.fc(x), None

class waveletEEGNet(BaseNet):
    def __init__(self, **kwargs):
        super(waveletEEGNet, self).__init__(**kwargs)
        
        self.dropout = nn.Dropout2d if self.dropoutType == 'SpatialDropout2D' else nn.Dropout
        self.convtoEGG = nn.Conv2d(30, self.squeeze_dim, (1, 1), padding='same', bias=False) #(b, 30, a, t) -> (b, 1, a, t)
        self.conv1 = nn.Sequential(
            # (batch, 1, time, sample)
            nn.Conv2d(1, self.F1, (1, self.kernLength), padding='same', bias=False), 
            nn.BatchNorm2d(self.F1),
            nn.ELU(),
            nn.Conv2d(self.F1, self.F1 * self.D, (1, self.Chans), groups=self.F1, bias=False),
            nn.BatchNorm2d(self.F1 * self.D),
            nn.ELU(),
            nn.AvgPool2d((4, 1)),
            self.dropout(self.dropoutRate)
        )

        self.separableConv = nn.Sequential(
            nn.Conv2d(self.F1 * self.D, self.F2, (1, 16), padding='same', bias=False),
            nn.BatchNorm2d(self.F2),
            nn.ELU(),
            nn.AvgPool2d((8, 1)),
            self.dropout(self.dropoutRate)
        )
        self.fc = self.fc = weight_norm(nn.Linear(self.F2 * (375 // 32), 2), dim=0) 

    def forward(self, x):
        # x = x.permute(0, 1, 3, 2)  
        x = self.convtoEGG(x)
        x = self.conv1(x) 
        x = self.separableConv(x) 
        x = x.flatten(start_dim=1) 
        x = self.fc(x)
        return x, self.convtoEGG.weight

    
class EEGNet(BaseNet):
    def __init__(self,**kwargs):
        super(EEGNet, self).__init__(**kwargs)
        
        self.dropout = nn.Dropout2d if self.dropoutType == 'SpatialDropout2D' else nn.Dropout
        
        self.conv1 = nn.Sequential(
            # (batch, 1, time, sample)
            nn.Conv2d(1, self.F1, (1, self.kernLength), padding='same', bias=False), 
            nn.BatchNorm2d(self.F1),
            nn.ELU(),
            nn.Conv2d(self.F1, self.F1 * self.D, (self.Chans, 1 ), groups=self.F1, bias=False),
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
        self.fc = self.fc = weight_norm(nn.Linear(self.F2 * (375 // 32), 2), dim=0) 
    def forward(self, x):
        x = x.unsqueeze(1)
        # x = x.permute(0, 1, 3, 2)  # (batch, 1, channel, time) -> (batch, 1, time, channel)
        x = self.conv1(x)
        # print(x.shape)
        x = self.separableConv(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x, None

class DeepCONV(BaseNet):
    def __init__(self, **kwargs):
        super(DeepCONV, self).__init__(**kwargs)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 5, (1, 3), padding='valid', bias=False),
            nn.Conv2d(5, 5, (self.Chans, 1), padding='valid', bias=False),
            nn.BatchNorm2d(5, eps=1e-05, momentum=0.1),
            nn.ELU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(self.dropoutRate)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(5, 10, (1, 3), padding='valid', bias=False),
            nn.BatchNorm2d(10, eps=1e-05, momentum=0.1),
            nn.ELU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(self.dropoutRate)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(10, 20, (1, 3), padding='valid', bias=False),
            nn.BatchNorm2d(20, eps=1e-05, momentum=0.1),
            nn.ELU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(self.dropoutRate)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(20, 40, (1, 3), padding='valid', bias=False),
            nn.BatchNorm2d(40, eps=1e-05, momentum=0.1),
            nn.ELU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(self.dropoutRate)
        )
        self.fc = nn.Linear(920, self.nb_classes, bias=False)
        
        self.max_norm_val = {
            'conv1': 2,
            'conv2': 2,
            'conv3': 2,
            'conv4': 2,
            'fc': 0.5
        }
        
    def forward(self, x):
        x = x.unsqueeze(1)
        # (B, 1, C, T)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x, None
    
    def apply_max_norm(self, eps=1e-8):
        for name, param in self.named_parameters():
            if name in self.max_norm_val and param.requires_grad:
                max_val = self.max_norm_val[name]  # Get the specific max_norm value for this layer
                norm = param.norm(2, dim=0, keepdim=True)  # Calculate L2 norm
                desired = torch.clamp(norm, 0, max_val)  # Limit the norm to the max_val
                param.data = param.data * (desired / (eps + norm))  # Update weights
                
    
class wavDeepCONV(BaseNet):
    def __init__(self, **kwargs):
        super(wavDeepCONV, self).__init__(**kwargs)
        self.convtoEGG = nn.Conv2d(30, self.squeeze_dim, (1, 1), padding='same', bias=False)
        self.model = Deep4Net(
            n_chans=self.Chans,
            n_outputs=self.nb_classes,
            n_times=375,
            final_conv_length="auto",
            n_filters_time=25,
            n_filters_spat=25,
            filter_time_length=5,  # Matches the kernel size of conv1
            pool_time_length=2,  # Matches the pooling length of conv1
            pool_time_stride=1,  # Matches the stride of conv1
            n_filters_2=50,  # Matches conv2
            filter_length_2=5,  # Matches the kernel size of conv2
            n_filters_3=100,  # Matches conv3
            filter_length_3=5,  # Matches the kernel size of conv3
            n_filters_4=200,  # Matches conv4
            filter_length_4=5,  # Matches the kernel size of conv4
            first_conv_nonlin=nn.ELU(alpha=0.4),
            first_pool_mode="max",
            first_pool_nonlin=nn.Identity(),
            later_conv_nonlin=nn.ELU(alpha=0.4),
            later_pool_mode="max",
            later_pool_nonlin=nn.Identity(),
            drop_prob=self.dropoutRate,
            split_first_layer=True,
            batch_norm=True,
            batch_norm_alpha=0.1,
            stride_before_pool=False
        )


    def forward(self, x):
        
        x = self.convtoEGG(x)
        x = x.squeeze(1)
        # x = x.permute(0, 2, 1)
        return self.model(x), None
    
class ShallowConvNet(BaseNet):
    def __init__(self, **kwargs):
        super(ShallowConvNet, self).__init__(**kwargs)
        self.dropout = nn.Dropout2d(p=self.dropoutRate) if self.dropoutType == 'SpatialDropout2D' else nn.Dropout(p=self.dropoutRate)
        
        # Layer 1
        self.conv1 = nn.Sequential(
            weight_norm(nn.Conv2d(1, 20, kernel_size=(1, 13), padding='same', bias=False)),
            weight_norm(nn.Conv2d(20, 20, kernel_size=(self.Chans, 1), padding='valid', bias=False)),
            nn.BatchNorm2d(20, eps=1e-05, momentum=0.1),
            nn.Identity()
        )

        # Layer 2
        self.activation_square = lambda x: x ** 2  # Square activation
        self.pool = nn.AvgPool2d(kernel_size=(1, 35), stride=(1, 7))
        self.activation_log = lambda x: torch.log1p(x)  # Log activation

        # Flatten and classification
        
        self.flatten = nn.Flatten()
        
        self.fc = weight_norm(nn.Linear(980, self.nb_classes, bias=False))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension: (batch, time, channel) -> (batch, 1, time, channel)
        # x = x.permute(0, 1, 3, 2)  # Adjust dimensions for Conv2D: (batch, 1, time, channel) -> (batch, 1, channel, time)
        x = self.conv1(x)
        x = self.activation_square(x)
        x = self.pool(x)
        x = self.activation_log(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x, None
    
class wavShallowConvNet(BaseNet):
    def __init__(self, **kwargs):
        super(wavShallowConvNet, self).__init__(**kwargs)
        self.conEEG = nn.Conv2d(30, self.squeeze_dim, (1, 1), padding='same', bias=False)
        # Layer 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 40, kernel_size=(1, 13), padding='same', bias=False),
            nn.Conv2d(40, 40, kernel_size=(self.Chans, 1), padding='valid', bias=False),
            nn.BatchNorm2d(40, eps=1e-05, momentum=0.1),
            nn.Identity()
        )

        # Layer 2
        self.activation_square = lambda x: x ** 2  # Square activation
        self.pool = nn.AvgPool2d(kernel_size=(1, 35), stride=(1, 7))
        self.activation_log = lambda x: torch.log1p(x)  # Log activation

        # Flatten and classification
        
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=self.dropoutRate)
        self.fc = nn.Linear(1960, self.nb_classes, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        
        x = x.permute(0, 1, 3, 2)  # Adjust dimensions for Conv2D: (batch, 1, time, channel) -> (batch, 1, channel, time)
        x = self.conEEG(x)
        x = self.conv1(x)
        x = self.activation_square(x)
        x = self.pool(x)
        x = self.activation_log(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x, None
    
class EEGNetV4(BaseNet):
    def __init__(self, **kwargs):
        super(EEGNetV4, self).__init__(**kwargs)
        self.model = EEGNetv4(
            n_chans=self.Chans,
            n_outputs=self.nb_classes,
            n_times=375,
            F1=self.F1,
            D=self.D,
            F2=self.F2,
            kernel_length=self.kernLength,
            drop_prob=self.dropoutRate
        )

    def forward(self, x):
        
        x = x.permute(0, 2, 1)
        return self.model(x), None
    

class waveletEEGNetV4(BaseNet):
    def __init__(self, **kwargs):
        super(waveletEEGNetV4, self).__init__(**kwargs)
        self.convtoEGG = nn.Conv2d(30, self.squeeze_dim, (1, 1), padding='same', bias=False) 
        self.model = EEGNetv4(
            n_chans=self.Chans,
            n_outputs=self.nb_classes,
            n_times=375,
            F1=self.F1,
            D=self.D,
            F2=self.F2,
            kernel_length=self.kernLength,
            drop_prob=self.dropoutRate
        )

    def forward(self, x):
        # print(x.shape)
        x = self.convtoEGG(x)
        x = x.squeeze(1)
        x = x.permute(0, 2, 1)
        return self.model(x), None
    
class EEGNetV1(BaseNet):
    def __init__(self, **kwargs):
        super(EEGNetV1, self).__init__(**kwargs)
        self.model = EEGNetv1(
                n_chans=self.Chans,
                n_outputs=self.nb_classes,
                n_times=375,
                final_conv_length='auto',
                pool_mode='max',
                second_kernel_size=(2, 32),
                third_kernel_size=(8, 4),
                drop_prob=self.dropoutRate
            )

    def forward(self, x):
        # print(x.shape)
        return self.model(x), None
    
class modelType(nn.Module):
    def __init__(self, model_type, **kwargs):
        super(modelType, self).__init__()

        model_mapping = {
            "EEGNet": EEGNet,
            "waveletEEGNet": waveletEEGNet,
            "CNN": CNN,
            "CNN_3D": CNN_3D,
            "EEGNetV4": EEGNetV4,
            "EEGNetV1": EEGNetV1,
            "waveletEEGNetV4": waveletEEGNetV4,
            "DeepCONV": DeepCONV, 
            "wavDeepCONV":wavDeepCONV, 
            "ShallowConvNet":ShallowConvNet, 
            "wavShallowConvNet": wavShallowConvNet
        }

        if model_type not in model_mapping:
            raise ValueError(
                f"Invalid model_type '{model_type}'. Available options are: {list(model_mapping.keys())}"
            )

        self.model = model_mapping[model_type](**kwargs)

    def forward(self, x):
        return self.model(x)

    def apply_max_norm(self):
        self.model.apply_max_norm()