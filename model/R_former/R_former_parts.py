from einops import rearrange, repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops





def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, model='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, model='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)



class LAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.Nh   =  heads
        self.dk  =  dim
        self.dv  =  dim
        self.scale = dim ** -0.5
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_q.apply(weights_init_classifier)
        self.to_k.apply(weights_init_classifier)
        self.to_v.apply(weights_init_classifier)

        self.o_proj = nn.Linear(dim//self.Nh, dim//self.Nh, bias=True)
        self.g_proj = nn.Linear(dim//self.Nh, dim//self.Nh)
        torch.nn.init.zeros_(self.g_proj.weight)
        torch.nn.init.ones_(self.g_proj.bias)

        torch.nn.init.zeros_(self.o_proj.bias)


        self.weight_r = nn.Linear(dim//self.Nh, dim//self.Nh, bias = False)
        self.weight_alpha = nn.Parameter(torch.randn(dim//self.Nh))
        self.weight_beta = nn.Parameter(torch.randn(dim//self.Nh))






    def forward(self, x,mask = None):
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        q = rearrange(q, 'b s n (h d) -> b s h n d', h=self.heads)
        k = rearrange(k, 'b s n (h d) -> b s h n d', h=self.heads)
        v = rearrange(v, 'b s n (h d) -> b s h n d', h=self.heads)
        # dots = torch.einsum('bshid,bshjd->bshij', q, k) * self.scale
        # attn = dots.softmax(dim=-1)
        # out = torch.einsum('bshij,bshjd->bshid', attn, v)
        #
        # out = rearrange(out, 'b s h n d -> b s n (h d)')


#############################################################################

        b , s , h, n , d = q.size()


        # Caculate the global query
        
        
        alpha_weight = (torch.mul(q, self.weight_alpha) * self.scale)
        alpha_weight = torch.softmax(alpha_weight, dim = -1)
        global_query = q * alpha_weight
        global_query = torch.einsum('b s h n d -> b s h d', global_query)

        # Model the interaction between global query vector and the key vector
        repeat_global_query = einops.repeat(global_query, 'b s h d -> b s h copy d', copy = n)
        p = repeat_global_query * k
        beta_weight = (torch.mul(p, self.weight_beta) * self.scale)
        beta_weight = torch.softmax(beta_weight, dim = -1)
        global_key = p * beta_weight
        global_key = torch.einsum('b s h n d -> b s h d', global_key)

        # key-value
        key_value_interaction = torch.einsum('b s h j, b s h n j -> b s h n j', global_key, v)
        key_value_interaction_out = self.weight_r(key_value_interaction)
        result = key_value_interaction_out + q
        result = rearrange(result, 'b s h n d -> b s n (h d)')
        return result




        # return out


class Rlocalformer(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels,h=4,w=4):
        super().__init__()
        self.patch_size =1
        self.w=w
        self.h=h
        self.attention =  LAttention(mid_channels, heads=8) 
        self.pre_ViT=nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1,stride=1,padding="same")
        self.mix=nn.ConvTranspose2d(in_channels=mid_channels, out_channels=in_channels, kernel_size=1, stride=1)
        self.mid_channels = mid_channels
    def forward(self, x,mask = None):
        H = x.size()[-2]
        W = x.size()[-1]
        x_1 = self.pre_ViT(x)
        x_1 = rearrange(x_1, 'b c (sh h) (sw w) -> b (sh sw) c h w', h=self.h,w=self.w)
        x_1 = rearrange(x_1, 'b s c h w -> b s (h w) c')
        x_1 = self.attention(x_1,None)
        x_1 = rearrange(x_1, 'b (sh sw) (h w) c -> b c (sh h) (sw w)', h=self.h,w=self.w, sw=W//self.w,sh=H//self.h,c=self.mid_channels)
        x_1= self.mix(x_1)
        x = x_1+x
        return x





class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Forward_conv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Mix_conv(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(out_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = x2+ x1
        return self.conv(x)


class Forward_former(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

        self.former = Rlocalformer(in_channels, out_channels, in_channels // 2)

    def forward(self, x):

        if (x.size()[-1]%4==0):
            return self.maxpool_conv(self.former(x))
        else:
            return self.maxpool_conv(x)




class Mix_former(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            self.former = Rlocalformer(out_channels, out_channels, out_channels // 2)
            self.conv = DoubleConv(out_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = x2+ x1

        if (x.size()[-1]%4==0):
            return self.conv(self.former(x))
        else:
            return self.conv(x)
