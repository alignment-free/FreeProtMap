from .R_former_parts import *
from .tri import *



class R_former(nn.Module):
    def __init__(self, n_channels, bilinear=False):
        super(R_former, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.layer1 = Forward_conv(n_channels, 48)
        self.tri_1 = TriangularSelfAttentionBlock(48)
        self.layer2 = Forward_former(48, 128)
        self.layer3 = Mix_former(128, 48, bilinear)
        self.layer4 = Mix_conv(48, n_channels, bilinear)
        self.lr = nn.Conv2d(n_channels, 1, kernel_size=1,bias=False)
        self.pMAE = nn.Conv2d(n_channels, 1, kernel_size=1,bias=False)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1, L):
        #length = x1.size(-1)
        #L = int(L.item())
        #x1 = x[:,:,:L,:L]
        x4 = self.layer1(x1)
        x4 = self.tri_1(x4.permute(0,2,3,1))
        x4 = x4.permute(0,3,1,2)
        x5 = self.layer2(x4)
        
        x = self.layer3(x5, x4)
        x = self.layer4(x, x1)
        pMAE = self.pMAE(x)
        x = self.lr(x)
        
        x = self.sigmoid(x)
        pMAE = self.sigmoid(pMAE)

        #p3d = (0,length-L,0,length-L)
        #x = torch.nn.functional.pad(x,p3d, "constant", 0)       
        #pMAE = torch.nn.functional.pad(pMAE,p3d, "constant", 0)                     
        return x,pMAE

