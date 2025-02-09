import numpy as np
import torch
from torch import nn
import os

class Model:
    def __init__(self):
        # You could include a constructor to initialize your model here, but all calls will be made to the load method
        self.model = None
        
    class WSC_1det_struct(nn.Module):
        def __init__(self, 
                    encoder_struct = [4,4,4,4]):
            super(Model.WSC_1det_struct, self).__init__()
            
            self.dep = len(encoder_struct)
            self.encoder_struct = torch.IntTensor(encoder_struct)

            self.relu = nn.ReLU()  # 激活函数
            self.layers = nn.ModuleList()
            self.norm_layers = nn.ModuleList()


            for i in range(self.dep-1):
                layer = nn.Linear(self.encoder_struct[i], self.encoder_struct[i+1])
                nn.init.kaiming_normal_(layer.weight)
                self.layers.append(layer)
                self.norm_layers.append(nn.BatchNorm1d(self.encoder_struct[i+1]))

        def forward(self, x):
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if i < self.dep-2:
                    x = self.relu(x)
                    # x = nn.BatchNorm1d(self.encoder_struct[i+1])(x)
                    x = self.norm_layers[i](x)

            return x

    def predict(self, X):
        # This method should accept an input of any size (of the given input format) and return predictions appropriately
        # Rfft the input X
        # Consider the X in (n,2,200) format and (H,L) in sequence

        if X.shape[-2:] == (200,2):
            X = np.swapaxes(X, -1, -2)
        
        assert X.shape[-2:] == (2,200)

        input = X / np.linalg.norm(X, axis = -1).reshape(-1,2,1)
        input = np.abs(np.fft.rfft(input, axis = -1))

        input = (input / np.linalg.norm(input, axis = -1).reshape(-1,2,1)).reshape(-1,202)

        score = np.sum((nn.Softmax(dim = 1)(self.model(torch.FloatTensor(input))).detach().numpy() * np.array([0,0,1,1,1])), axis = 1)

        return score

    def load(self):
        self.model = self.WSC_1det_struct([202,64,16,5])
        self.model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), 'model.pt')))
        self.model.eval()
