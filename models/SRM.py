from __future__ import absolute_import
from __future__ import division

from torch.nn import init
import torch
import math
from torch import nn
from torch.nn import functional as F
from .av_crossatten import DCNLayer

class SRM(nn.Module):
    def __init__(self):
        super(SRM, self).__init__()

        self.coattn = DCNLayer(256, 128, 2, 0.6)
        
        self.vregressor = nn.Sequential(
            nn.LayerNorm(384),
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

        self.aregressor = nn.Sequential(nn.Linear(512, 128),
                                        nn.ReLU(inplace=True),
                                     nn.Dropout(0.6),
                                 nn.Linear(128, 1))
        
        self.video_gate = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.audio_gate = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
                                           
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)#추가
        self.init_weights()

    def init_weights(net, init_type='xavier', init_gain=1):

        if torch.cuda.is_available():
            net.cuda()

        def init_func(m):  # define the initialization function
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.uniform_(m.weight.data, 0.0, init_gain)
                elif init_type == 'xavier':
                    init.xavier_uniform_(m.weight.data, gain=init_gain)
                elif init_type == 'kaiming':
                    init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=init_gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        print('initialize network with %s' % init_type)
        net.apply(init_func)  # apply the initialization function <init_func>
    
    def forward(self, f1_norm, f2_norm):
        video = F.normalize(f1_norm, dim=-1)
        audio = F.normalize(f2_norm, dim=-1)

        atten_video, atten_audio = self.coattn(video, audio)
        
        # co-attn 결과 stack 유지
        video_stack = torch.stack([video, atten_video[-2], atten_video[-1]], dim=2)
        audio_stack = torch.stack([audio, atten_audio[-2], atten_audio[-1]], dim=2)

        # shape: [B, L, M, D]

        # gating score 계산
        video_score = self.video_gate(video_stack)  # [B, L, M, 1]
        video_weights = torch.softmax(video_score, dim=2)

        fused_video = torch.sum(video_stack * video_weights, dim=2)

        audio_score = self.audio_gate(audio_stack)
        audio_weights = torch.softmax(audio_score, dim=2)

        fused_audio = torch.sum(audio_stack * audio_weights, dim=2)

        audiovisualfeatures = torch.cat((fused_video, fused_audio), -1)

        return audiovisualfeatures
