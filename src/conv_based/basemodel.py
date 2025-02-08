from abc import ABC, abstractmethod
import torch.nn as nn

class BaseNet(nn.Module, ABC):
    def __init__(self, **kwargs):
        super(BaseNet, self).__init__()
        # 提取通用參數
        self.init_w = kwargs.get("init_w", "orthogonal")
        self.nb_classes = kwargs.get("num_classes", 2)
        self.Chans = kwargs.get("channels", 64)
        self.wavelet = kwargs.get("wavelet", 30)
        self.dropoutRate = kwargs.get("dropoutRate", 0.5)
        self.dropoutType = kwargs.get("dropoutType", 'Dropout')
        self.squeeze_dim = kwargs.get("squeeze_dim", 1)
        self.F1 = kwargs.get("F1", 8)
        self.F2 = kwargs.get("F2", 16)
        self.poolKern1 = kwargs.get("poolKern1", 4)
        self.poolKern2 = kwargs.get("poolKern2", 8)
        self.D = kwargs.get("D", 2)
        self.kernLength = kwargs.get("kernLength", 64)
        self.dropout = nn.Dropout2d if self.dropoutType == 'SpatialDropout2D' else nn.Dropout
        self.fc = None  
        self._init_weights()

    def build_fc(self, x, device):
        
        if self.fc is None:
            self.fc = nn.Linear(x.shape[1], self.nb_classes).to(device)
    
    def _init_weights(self): #He initialization
        if self.init_w == "xavier":
            self._init_weights_xavier()
        elif self.init_w == "he":
            self._init_weights_he()
        elif self.init_w == "normal":
            self._init_weights_normal()
        elif self.init_w == "orthogonal":
            self._init_weights_orthogonal()
        else:
            raise ValueError("Invalid weight initialization type")
    def _init_weights_xavier(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _init_weights_he(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _init_weights_normal(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def _init_weights_orthogonal(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
        
 

    @abstractmethod
    def forward(self, x, attention=False):
        pass

