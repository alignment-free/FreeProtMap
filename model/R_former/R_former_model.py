from .R_former_parts import *


class R_former(nn.Module):
    def __init__(self, n_channels, bilinear=False):
        super(R_former, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.layer1 = Forward_conv(n_channels, 64)
        self.layer2 = Forward_former(64, 128)
        self.layer3 = Mix_former(128, 64, bilinear)
        self.layer4 = Mix_conv(64, n_channels, bilinear)
        self.lr = nn.Conv2d(n_channels, 1, kernel_size=1,bias=False)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x, L):
        length = x.size(3)
        L = int(L.item())
        x1 = x[:,:,:L,:L]

        x4 = self.layer1(x1)
        x5 = self.layer2(x4)

        x = self.layer3(x5, x4)
        x = self.layer4(x, x1)
        x = self.lr(x)
        x = self.sigmoid(x)
        p3d = (0,length-L,0,length-L)
        x = torch.nn.functional.pad(x,p3d, "constant", 0)                             
        return x

