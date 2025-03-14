import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
# from torch.nn import Module, Sequential
# from torch.nn import Conv3d, ConvTranspose3d, BatchNorm3d, MaxPool3d, AvgPool1d, Dropout3d
# from torch.nn import ReLU, Sigmoid
'''
the shape of Conv3d input is (batch size, channel, sequence, height, width)
'''

class Conv3d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(Conv3d, self).__init__()

        # First convolutional layer with BatchNorm and ReLU activation
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(inplace=True),
        )

        # Second convolutional layer with BatchNorm and ReLU activation
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(True),
        )
        # Residual connection to maintain gradient flow
        self.residual = nn.Conv3d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False)


    def forward(self, x):
        return self.conv2(self.conv1(x)) + self.residual(x)

class Down(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride):
        super(Down, self).__init__()
        # Down-sampling block using max pooling followed by Conv3d block
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=kernel_size, stride=stride),
            Conv3d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, x1_in, x2_in, out_channel, kernel_size, stride, padding):
        super(Up, self).__init__()

        # Up-sampling using transposed convolution followed by Conv3d block
        self.up = nn.Sequential(
            nn.ConvTranspose3d(x1_in, x1_in, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.ReLU()
        )

        # Feature fusion using convolutional layers
        self.conv = Conv3d(x1_in + x2_in, out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Adjust shape for concatenation
        diffC = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffC // 2, diffC - diffC // 2, diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channel_list, out_channel, kernel_size, stride, padding):
        super(OutConv, self).__init__()

        # Compute the total input channels after up-sampling
        channel_sum = np.sum(np.array(in_channel_list))
        self.up_list = []
        for i,channel in enumerate(in_channel_list):
            if i == len(in_channel_list)-1:
                continue
            self.up_list.append(
                nn.Sequential(
                    nn.ConvTranspose3d(channel, channel, kernel_size=[1,np.power(2,(len(in_channel_list)-1)-i),np.power(2,(len(in_channel_list)-1)-i)],
                                       stride=[1,np.power(2,(len(in_channel_list)-1)-i),np.power(2,(len(in_channel_list)-1)-i)], padding=padding, bias=True),
                    nn.ReLU(),
                    nn.Conv3d(channel, in_channel_list[-1], kernel_size=3, stride=1, padding=1, bias=True),
                    nn.BatchNorm3d(in_channel_list[-1]),
                    nn.ReLU(inplace=True)
                )
            )
        self.up_list = nn.ModuleList(self.up_list)

        # Final convolutional layer to produce output
        self.conv = nn.Sequential(
            nn.Conv3d(in_channel_list[-1], out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(),
            nn.Conv3d(out_channel, out_channel, kernel_size=1, stride=1, padding=0)
        )
    def forward(self, x):
        x6,x7,x8,x9 = tuple(x)
        x6 = self.up_list[0](x6)
        x7 = self.up_list[1](x7)
        x8 = self.up_list[2](x8)
        x_last = torch.cat([x6,x7,x8,x9],dim=2)


        return self.conv(x_last)

class Unet3D(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Unet3D, self).__init__()

        # 3D U-Net architecture with encoder and decoder blocks
        self.inc = Conv3d(in_channel, 16, kernel_size=3, stride=1, padding=1)
        self.down1 = Down(16, 32, kernel_size=[1, 2, 2], stride=[1, 2, 2])
        self.down2 = Down(32, 64, kernel_size=[1, 2, 2], stride=[1, 2, 2])
        self.down3 = Down(64, 128, kernel_size=[2, 2, 2], stride=[2, 2, 2])
        self.down4 = Down(128, 128, kernel_size=[2, 2, 2], stride=[2, 2, 2])

        self.up1 = Up(128, 128, 64, kernel_size=[2, 2, 2], stride=[2, 2, 2], padding=0)
        self.up2 = Up(64, 64, 32, kernel_size=[2, 2, 2], stride=[2, 2, 2], padding=0)
        self.up3 = Up(32, 32, 16, kernel_size=[1, 2, 2], stride=[1, 2, 2], padding=0)
        self.up4 = Up(16, 16, 16, kernel_size=[1, 2, 2], stride=[1, 2, 2], padding=0)

        self.outc = OutConv([64,32,16,16], out_channel, kernel_size=[18, 1, 1], stride=[1, 1, 1], padding=0)

        # self.h2input = nn.Linear(64,16)
        # self.h2b = nn.Linear(64, 128)
        # self.x52fusion = nn.Linear(128 * 2 * 4 * 8, 64)

    def forward(self, x):
        # Forward pass through encoder and decoder
        batch, _, _, _, _ = x.shape
        x1 = self.inc(x)  # [4, 64, 5, 100, 100]
        # output = self.h2input(output)
        # x1 = x1 + output.permute(1, 2, 0).contiguous().unsqueeze(-1).unsqueeze(-1)
        x2 = self.down1(x1)     # [4, 128, 5, 50, 50]
        x3 = self.down2(x2)     # [4, 256, 5, 25, 25
        x4 = self.down3(x3)     # [4, 512, 2, 12, 12]
        x5 = self.down4(x4)     # [4, 512, 1, 6, 6]

        # final_encoder_h = self.h2b(final_encoder_h.view(batch, -1))  # [batch,512]
        # x5 = x5 + final_encoder_h.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        x6 = self.up1(x5, x4)    # [4, 256, 2, 12, 12]
        x7 = self.up2(x6, x3)     # [4, 128, 5, 25, 25]
        x8 = self.up3(x7, x2)     # [4, 64, 5, 50, 50]
        x9 = self.up4(x8, x1)     # [4, 64, 5, 100, 100]

        out = self.outc([x6,x7,x8,x9])      # [4, 1, 1, 100, 100]
        # fusion_feature = self.x52fusion(x5.view(batch, -1))
        return out

if __name__ == '__main__':

    x = torch.randn((96, 1, 8, 64, 64)).cuda()  #
    # final_encoder_h = torch.randn((1, 1, 64)).cuda()
    # output = torch.randn((8, 1, 64)).cuda()
    net = Unet3D(1, 1).cuda()
    out = net(x)
    print(out.shape)


