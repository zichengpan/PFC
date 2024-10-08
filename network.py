import torch
import torch.nn as nn
import torchvision
import torch.nn.utils.weight_norm as weightNorm

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find(
            'ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        try:
            nn.init.zeros_(m.bias)
        except AttributeError:
            pass

class feat_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="linear"):
        super(feat_classifier, self).__init__()
        if type == "linear":
            self.fc = nn.Linear(bottleneck_dim, class_num, bias=False)
        else:
            self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num),
                                 name="weight")
        self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x


class ResNet_FE(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        model_resnet = torchvision.models.resnet50(True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu,
                                            self.maxpool, self.layer1,
                                            self.layer2, self.layer3,
                                            self.layer4)
        self.bottle = nn.Linear(2048, 256)
        self.bn = nn.BatchNorm1d(256)

        self.proto = nn.Parameter(torch.zeros([class_num, 2, 7, 7]), requires_grad=False)


    def forward(self, x):
        feat_maps = self.feature_layers(x)
        out = self.avgpool(feat_maps)
        out = out.view(out.size(0), -1)
        out = self.bn(self.bottle(out))
        return feat_maps, out

