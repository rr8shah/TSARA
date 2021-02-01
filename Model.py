import math

import torch.nn as nn
import torch.nn.functional as F


class EONSS(nn.Module):
    def __init__(self):
        super(EONSS, self).__init__()

        self.conv1 = nn.Conv2d(3, 4, 5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(4, 8, 5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(8, 16, 5, stride=2, padding=2)
        self.conv4 = nn.Conv2d(16, 32, 3, stride=1, padding=0)

        self.st2_fc1 = nn.Conv2d(32, 16, 1, stride=1, padding=0)
        self.st2_fc2 = nn.Conv2d(16, 1, 1, stride=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        batch_size = x.size()[0]

        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv4(x)), (2, 2))

        y2 = F.relu(self.st2_fc1(x))
        s = self.st2_fc2(y2)
        s = s.view(batch_size, -1)
        out = F.sigmoid(s)
        return out
