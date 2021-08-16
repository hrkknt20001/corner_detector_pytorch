import torch
import torch.nn as nn
from torchvision import models

class Model(nn.Module):

    def __init__(self, last_layer=''):

        super(Model, self).__init__()

        self.vgg16 = models.vgg16(pretrained=True)
        
        # keep feature extraction network up to indicated layer
        vgg_feature_layers=['conv1_1','relu1_1','conv1_2','relu1_2','pool1','conv2_1',
                            'relu2_1','conv2_2','relu2_2','pool2','conv3_1','relu3_1',
                            'conv3_2','relu3_2','conv3_3','relu3_3','pool3','conv4_1',
                            'relu4_1','conv4_2','relu4_2','conv4_3','relu4_3','pool4',
                            'conv5_1','relu5_1','conv5_2','relu5_2','conv5_3','relu5_3','pool5']

        if last_layer=='':
            last_layer = 'pool5'
            
        last_layer_idx = vgg_feature_layers.index(last_layer)
        self.vgg16 = nn.Sequential(*list(self.vgg16.features.children())[:last_layer_idx+1])

        self.input_size = 1*512*7*7
        self.hidden_size = 512

        self.fc_lt = nn.Sequential(
            nn.Linear(in_features=self.input_size, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=2, bias=False)
        )

        self.fc_rt = nn.Sequential(
            nn.Linear(in_features=self.input_size, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=2, bias=False)
        )

        self.fc_lb = nn.Sequential(
            nn.Linear(in_features=self.input_size, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=2, bias=False)
        )

        self.fc_rb = nn.Sequential(
            nn.Linear(in_features=self.input_size, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=2, bias=False)
        )


    def forward(self, input):

        # ひとつめのデータを用いて畳み込みの計算
        x = self.vgg16(input)

        # 畳み込み層からの出力を1次元化
        x = x.view(x.size(0), self.input_size)

        # 全結合層に入力して計算
        lt = self.fc_lt(x)
        rt = self.fc_rt(x)
        lb = self.fc_lb(x)
        rb = self.fc_rb(x)
        
        return lt, rt, lb, rb