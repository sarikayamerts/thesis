import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv

class TemporalConvLayer(nn.Module):
    ''' Temporal convolution layer.

    arguments
    ---------
    c_in : int
        The number of input channels (features)
    c_out : int
        The number of output channels (features)
    dia : int
        The dilation size
    '''
    def __init__(self, c_in, c_out, dia = 1):
        super(TemporalConvLayer, self).__init__()
        self.c_out = c_out
        self.c_in = c_in
        self.conv = nn.Conv2d(c_in, c_out, (2, 1), 9, dilation = dia, padding = (0,0))

    def forward(self, x):
        return torch.relu(self.conv(x))


class SpatioConvLayer(nn.Module):
    def __init__(self, c, Lk): # c : hidden dimension Lk: graph matrix
        super(SpatioConvLayer, self).__init__()
        self.g = Lk
        self.gc = GraphConv(c, c, activation=F.relu)

    def init(self):
        stdv = 1. / math.sqrt(self.W.weight.size(1))
        self.W.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x = x.transpose(0, 3)
        x = x.transpose(1, 3)
        output = self.gc(self.g, x)
        output = output.transpose(1, 3)
        output = output.transpose(0, 3)
        return torch.relu(output)

class FullyConvLayer(nn.Module):
    def __init__(self, c):
        super(FullyConvLayer, self).__init__()
        self.conv = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        return self.conv(x)

class OutputLayer(nn.Module):
    def __init__(self, c, T, n):
        super(OutputLayer, self).__init__()
        self.tconv1 = nn.Conv2d(c, c, (T, 1), 1, dilation = 1, padding = (0,0))
        self.ln = nn.LayerNorm([n, c])
        self.tconv2 = nn.Conv2d(c, c, (1, 1), 1, dilation = 1, padding = (0,0))
        self.fc = FullyConvLayer(c)

    def forward(self, x):
        x_t1 = self.tconv1(x)
        x_ln = self.ln(x_t1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_t2 = self.tconv2(x_ln)
        return self.fc(x_t2)

class STGCN_WAVE(nn.Module):
    def __init__(self, c, T, n, Lk, device=torch.device("cpu"), control_str='TNTSTNTST'):
        """
        c: blocks
        - Conv(1, 16)

        """
        super(STGCN_WAVE, self).__init__()
        self.control_str = control_str
        self.num_layers = len(control_str)
        self.layers = nn.ModuleList([])
        cnt = 0
        diapower = 0
        for i in range(self.num_layers):
            i_layer = control_str[i]
            if i_layer == 'T': # Temporal Layer
                self.layers.append(TemporalConvLayer(c[cnt], c[cnt + 1], dia = 2**diapower))
                diapower += 1
                cnt += 1
            if i_layer == 'S': # Spatio Layer
                self.layers.append(SpatioConvLayer(c[cnt], Lk))
            if i_layer == 'N': # Norm Layer
                self.layers.append(nn.LayerNorm([n,c[cnt]]))
        self.output = OutputLayer(c[cnt], T + 1 - 2**(diapower), n)
        for layer in self.layers:
            layer = layer.to(device)

    def forward(self, x):
        for i in range(self.num_layers):
            i_layer = self.control_str[i]
            if i_layer == 'N':
                x = self.layers[i](x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            else:
                x = self.layers[i](x)
        return self.output(x)

def evaluate_model(model, loss, data_iter):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            y_pred = model(x).view(len(x), -1)
            l = loss(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        return l_sum / n


def evaluate_metric(model, data_iter, scaler):
    model.eval()
    with torch.no_grad():
        mae, mape, mse = [], [], []
        for x, y in data_iter:
            y = scaler.inverse_transform(y.cpu().numpy()).reshape(-1)
            y_pred = scaler.inverse_transform(model(x).view(len(x), -1).cpu().numpy()).reshape(-1)
            d = np.abs(y - y_pred)
            mae += d.tolist()
            mape += (d / y).tolist()
            mse += (d ** 2).tolist()
        MAE = np.array(mae).mean()
        MAPE = np.array(mape).mean()
        RMSE = np.sqrt(np.array(mse).mean())
        return MAE, MAPE, RMSE