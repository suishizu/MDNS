import math
import torch
import torch.nn as nn

class ADNM(nn.Module):
    def __init__(self, input_size, out_size, M):
        super(ADNM, self).__init__()

        self.input_size = input_size
        # k = torch.rand([out_size, M, input_size])
        w = torch.rand([out_size, M, input_size])
        q = torch.rand([out_size, M, input_size])
        m = torch.rand([out_size, 1, input_size])
        D_k = torch.rand([out_size, M])
        D_q = torch.rand([out_size, M])

        torch.nn.init.constant_(q, 0.1) 
        torch.nn.init.uniform_(m, a=-10, b=10)

        self.params = nn.ParameterDict({'w': nn.Parameter(w)})       
        self.params.update({'q': nn.Parameter(q)})
        self.params.update({'m': nn.Parameter(m)})

        self.params.update({'D_k': nn.Parameter(D_k)})
        self.params.update({'D_q': nn.Parameter(D_q)})

        self.input_ln = nn.LayerNorm(input_size)

    def forward(self, x, y, z):

        # 多模态
        xyz_input = torch.stack((x, y, z), dim=1)
        xyz = xyz_input.unsqueeze(1) 

        # 单模态
        # xyz = x.unsqueeze(0).unsqueeze(0)
 
        xyz = self.input_ln(xyz)

        mul = torch.mul(xyz, self.params['w']) - self.params['q']
        # y = kx + b

        S = torch.sigmoid(mul) 
        A = torch.tanh(self.params['m'])   
        sa = torch.mul(S, A)

        D = torch.sum(sa, -1) 
        D = self.params['D_k'] * D - self.params['D_q']

        D_sigmoid = torch.sigmoid(D)
        O = torch.sum(D_sigmoid, -1) 

        return O

class MDNS(nn.Module):
    def __init__(self, num_class, input_size=512, hidden_size=32, M=3):
        # input_size=512
        # hidden_size=128
        super(ADNS, self).__init__()
        self.DNM_Linear1 = ADNM(input_size, hidden_size, M)

        self.classifiers = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_class),
        )  

    def forward(self, x, y, z):
        x = self.DNM_Linear1(x, y, z)
        classifiers = self.classifiers(x)
        return classifiers
