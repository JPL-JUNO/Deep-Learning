"""
@Description: Model Development
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-05-14 21:47:03
"""

import torch.nn.functional as F
import torch.nn as nn
from torchvision import models
vgg16 = models.vgg16(weights=True)
# print(vgg16.classifier)


# wavelow = torch.hub.load('nvidia/DeepLearningExamples:torchhub',
#                          'nvidia_wavelow')
# torch.hub.list('nvidia/DeepLearningExamples:torchhub')
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(2048, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = x.view(-1, 2048)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x
