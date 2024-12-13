from abc import ABC, abstractmethod
import torch.nn as nn

class BaseNet(nn.Module, ABC):
    def __init__(self, **kwargs):
        super(BaseNet, self).__init__()
        # 提取通用參數
        self.nb_classes = kwargs.get("num_classes", 2)
        self.Chans = kwargs.get("channels", 64)
        self.wavelet = kwargs.get("wavelet", 30)
        self.dropoutRate = kwargs.get("dropoutRate", 0.5)
        self.dropoutType = kwargs.get("dropoutType", 'Dropout')
        self.F1 = kwargs.get("F1", 8)
        self.F2 = kwargs.get("F2", 16)
        self.D = kwargs.get("D", 2)
        self.kernLength = kwargs.get("kernLength", 64)
        self.dropout = nn.Dropout2d if self.dropoutType == 'SpatialDropout2D' else nn.Dropout
        self.fc = None  

    def build_fc(self, x, device):
        
        if self.fc is None:
            self.fc = nn.Linear(x.shape[1], self.nb_classes).to(device)

    @abstractmethod
    def forward(self, x):
        pass
