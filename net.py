import torch
import torch.nn.functional as F
import torch.nn as nn

class Convolution(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride):
        super().__init__()

        self.sub_module = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.sub_module(x)

class Upsample(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode='bilinear')


def Crop(x,x_small):
    diffY = x.size()[2] - x_small.size()[2]
    diffX = x.size()[3] - x_small.size()[3]
    x1 = nn.functional.pad(x_small, (diffX // 2, diffX - diffX // 2,
                                diffY // 2, diffY - diffY // 2))
    print(x1.size())
    exit()
    return x1

class Main(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
            Convolution(3,64,3,1),#570x570
            Convolution(64, 64, 3, 1),#568x568

        )
        self.down1 = nn.Sequential(
            nn.Conv2d(64,64,2,2),
            # nn.MaxPool2d(2)  # 284x284x
        )
        self.layer2 = nn.Sequential(
            Convolution(64,128,3,1),#282x282
            Convolution(128,128,3,1),#280x280
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(128,128,2,2),  # 140
        )
        self.layer3 = nn.Sequential(
            Convolution(128,256,3,1),#138x138
            Convolution(256,256,3,1),#136x136
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(256,256,2,2),  # 68
        )
        self.layer4 = nn.Sequential(
            Convolution(256,512,3,1),#66x66
            Convolution(512,512,3,1),#64x64
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(512,512,2,2),  # 32
        )
        #上采样1
        self.layer5 = nn.Sequential(
            Convolution(512,1024,3,1),#30x30
            Convolution(1024,1024,3,1),#28x28
        )
        self.up1 = nn.Sequential(
            Upsample()  # 64
        )
        self.layer6 = nn.Sequential(
            Convolution(1024+512, 512, 3, 1),  # 62
            Convolution(512, 512, 3, 1),  # 60
        )
        self.up2 = nn.Sequential(
            Upsample()  # 120
        )

        self.layer7 = nn.Sequential(
            Convolution(512+256, 256, 3, 1),  # 136
            Convolution(256, 256, 3, 1),  # 132

        )

        self.up3 = nn.Sequential(
            Upsample()  # 200
        )
        self.layer8 = nn.Sequential(
            Convolution(256+128, 128, 3, 1),  # 198
            Convolution(128, 128, 3, 1),  # 276
        )
        self.up4 = nn.Sequential(
            Upsample()  # 392
        )

        self.layer9 = nn.Sequential(
            Convolution(128+64, 64, 3, 1),  # 390
            Convolution(64, 64, 3, 1),  # 388
            Convolution(64, 2, 1, 1),  # 388
        )
    def forward(self, x):
        y1 = self.layer1(x)
        # print(y1.shape)
        down1 = self.down1(y1)
        # print(down1.shape)
        y2 = self.layer2(down1)
        # print(y2.shape)
        down2 = self.down2(y2)
        # print(down2.shape)
        y3 = self.layer3(down2)
        # print(y3.shape)
        down3  = self.down3(y3)
        # print(down3.shape)
        y4 = self.layer4(down3)
        # print(y4.shape)
        down4 = self.down4(y4)
        # print(down4.shape)
        y5 = self.layer5(down4)
        # print(y5.shape)
        up1 = self.up1(y5)
        # print(up1.shape)
        x1= Crop(y4,up1)
        # print(x1.shape)
        merge6 = torch.cat([x1, y4], dim=1)
        # print(merge6.size())
        y6 = self.layer6(merge6)
        # print('y6',y6.shape)
        up2 = self.up2(y6)
        # print('up2',up2.shape)
        x1 = Crop(y3, up2)
        # print(x1.shape)
        merge7 = torch.cat([x1, y3], dim=1)
        # print(merge7.shape)
        y7  = self.layer7(merge7)
        # print(y7.shape)
        up3 = self.up3(y7)
        # print(up3.shape)
        x1 = Crop(y2, up3)
        # print(x1.shape)
        merge8 = torch.cat([x1, y2], dim=1)
        # print(merge8.shape)
        y8 = self.layer8(merge8)
        # print(y8.shape)
        up4 = self.up4(y8)
        # print(up4.shape)
        x1 = Crop(y1, up4)
        # print(x1.shape)
        merge9 = torch.cat([x1, y1], dim=1)
        # print(merge9.shape)
        y9 = self.layer9(merge9)
        # print(y9.shape)
        return nn.Sigmoid()(y9)


main_net = Main()
x = torch.rand(3,572,572)
x = x.unsqueeze(0)
main_net.forward(x)