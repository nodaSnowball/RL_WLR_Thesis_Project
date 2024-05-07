import torch
import torch.nn as nn
import numpy as np
from torch.nn import Module, Linear, BatchNorm1d, Sequential, ReLU, Conv2d
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from gymnasium import spaces
from torchvision import models
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


def simple_conv_and_linear_weights_init(m):
    if type(m) in [
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.ConvTranspose1d,
        nn.ConvTranspose2d,
        nn.ConvTranspose3d,
    ]:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6.0 / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        if m.bias is not None:
            m.bias.data.fill_(0)

class ResNetEmbedder(nn.Module):
    def __init__(self, resnet, pool=True, device='cuda'):
        super().__init__()
        self.model = resnet
        self.pool = pool
        self.model.to(device)
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        if not self.pool:
            return x
        else:
            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
            return x

class ResNetEmbedderC(nn.Module):
    def __init__(self, resnet, input_channel=1, pool=True, device='cuda'):
        super().__init__()
        self.model = resnet
        self.pool = pool
        self.model.conv1 = nn.Conv2d(in_channels=input_channel, out_channels=64, kernel_size=7, stride=3, padding=3, bias=False)
        self.model.to(device)
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        if not self.pool:
            return x
        else:
            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
            return x

class feature_net(Module):
    def __init__(self, device):
        super(feature_net, self).__init__()
        self.device = device

        self.input_dim = 512
        self.resnet = ResNetEmbedderC(
            resnet=models.resnet18(pretrained=True), 
            pool=0,
            device=device)
        self.encoder = self._create_visual_encoder()

    def _create_visual_encoder(self) -> nn.Module:

        self.visual_attention = nn.Sequential(
            nn.Conv2d(self.input_dim, 128, 1,), nn.ReLU(inplace=True), nn.Conv2d(128, 32, 1,),
        ).to(self.device)
        visual_encoder = nn.Sequential(
            nn.Conv2d(self.input_dim, 256, 1,), nn.ReLU(inplace=True), nn.Conv2d(256, 128, 1,),
        ).to(self.device)
        self.visual_attention.apply(simple_conv_and_linear_weights_init)
        visual_encoder.apply(simple_conv_and_linear_weights_init)
        return visual_encoder

    def layer_init(self):
        """Initialize the RNN parameters in the model."""
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

    def forward(self, x):
        bs = x.size(0)
        x = self.resnet(x)
        x = self.encoder(x).view(bs,-1)
        return x

class CustomExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, args, features_dim: int = 3358,
                 device='cuda'):
        super(CustomExtractor, self).__init__(observation_space, features_dim)
        self.device = device
        # self.transform = TF.Compose([TF.Resize((224,224)), TF.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        # self.feature_net = feature_net(device)

        # 1d version
        self.cnn1 = Sequential(
            Conv2d(1,32,kernel_size=(20,1), stride=(9,1), padding=(10,0), bias=False),
            ReLU(),
            Conv2d(32,64,kernel_size=(5,1), stride=(3,1), padding=(5,0), bias=False),
            ReLU(),
            Conv2d(64,8,1),
        )
        self.cnn2 = Sequential(
            Conv2d(1,32,kernel_size=7, stride=3, padding=3, bias=True),
            ReLU(),
            Conv2d(32,64,kernel_size=7, stride=3, padding=3, bias=True),
            ReLU(),
            # Conv2d(128,64,kernel_size=(3,1), stride=(1,1), padding=(1,0), bias=False),
            # ReLU(),
            Conv2d(64,8,1),
        )
        self.mlp13 = Sequential(
            Linear(31, 64),
            ReLU(),
            Linear(64, 16)
        )
        self.mlp11 = Sequential(
            Linear(768,100),
            # ReLU()
            )
        self.mlp12 = Sequential(
            Linear(672,100),
            # ReLU()
            )
        self.mlp2 = Sequential(
            Linear(87, 256),
            ReLU(),
            Linear(256, 100)
        )

        
    def forward(self, state):
        bs = state.size(0)
        state = state.reshape(bs,1,100,-1)
        trj = state[:,:,:,:-56]                 # (bs,1,100,31)
        lidar = state[:,:,:,-56:]               # (bs,1,100,56)
        f11 = self.cnn1(trj)                    # (bs,8,6,31)  = (bs,1296)
        f11 = f11.reshape(-1,31)
        f11 = self.mlp13(f11)                   # (bs*8*6, 16)
        f11 = f11.reshape(bs,-1)                # (bs, 768)
        f11 = self.mlp11(f11)
        f12 = self.cnn2(lidar)                  # (bs,8,12,7) = (bs,672)
        f12 = f12.reshape(bs,-1)
        f12 = self.mlp12(f12)
        f1 = torch.cat((f11,f12), dim=1)        # (bs,512)
        
        current_state = state[:, 0, 0,:]
        f2 = self.mlp2(current_state)           # (bs,512)
        feature = torch.cat((f1, f2), dim=1)

        return feature                          # (bs,1024)