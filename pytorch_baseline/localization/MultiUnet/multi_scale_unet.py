import torch
import torch.nn as nn
import torch.nn.functional as F



class multiunet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(multiunet, self).__init__()
        print('use MultiUnet ! ')
        self.in_conv = InConv(n_channels, 64)
        self.down1 = DownConv(64, 128)
        self.down2 = DownConv(128, 256)
        self.down3 = DownConv(256, 512)
        self.atrous = AtrousConv(512, 256)
        self.up3 = UpConv(512, 128)
        self.up2_s = UpConv(256, 64)
        self.up2_m = UpConv(256, 64)
        self.up2_l = UpConv(256, 64)
        self.up1_s = UpConv(128, 64, kernel_size=5)
        self.up1_m = UpConv(128, 64, kernel_size=5)
        self.up1_l = UpConv(128, 64, kernel_size=5)
        self.out_conv_s = OutConv(64, n_classes)
        self.out_conv_m = OutConv(64, n_classes)
        self.out_conv_l = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.atrous(x4)
        x = self.up3(x, x3)
        x_s = self.up2_s(x, x2)
        x_m = self.up2_m(x, x2)
        x_l = self.up2_l(x, x2)
        x_s = self.up1_s(x_s, x1)
        x_m = self.up1_m(x_m, x1)
        x_l = self.up1_l(x_l, x1)
        x_s = self.out_conv_s(x_s)
        x_m = self.out_conv_m(x_m)
        x_l = self.out_conv_l(x_l)
        
        # return torch.sigmoid(x_s), torch.sigmoid(x_m), torch.sigmoid(x_l)
        return x_s, x_m, x_l

class InConv(nn.Module):
    def __init__(self, n_in, n_out):
        super(InConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_in, n_out, 5, padding=2),
            nn.BatchNorm2d(n_out),
            nn.ReLU(),
            nn.Conv2d(n_out, n_out, 5, padding=2),
            nn.BatchNorm2d(n_out),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class DownConv(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3):
        super(DownConv, self).__init__()
        self.max_pool = nn.MaxPool2d(2)
        padd = int((kernel_size - 1) / 2)
        self.conv = nn.Sequential(
            nn.Conv2d(n_in, n_out, kernel_size, padding=padd),
            nn.BatchNorm2d(n_out),
            nn.ReLU(),
            nn.Conv2d(n_out, n_out, kernel_size, padding=padd),
            nn.BatchNorm2d(n_out),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.max_pool(x)
        x = self.conv(x)
        return x


class AtrousConv(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3):
        super(AtrousConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_in, n_out, kernel_size, padding=2, dilation=2),
            nn.BatchNorm2d(n_out),
            nn.ReLU(),
            nn.Conv2d(n_out, n_out, kernel_size, padding=1),
            nn.BatchNorm2d(n_out),
            nn.ReLU(),
            nn.Conv2d(n_out, n_out, kernel_size, padding=1),
            nn.BatchNorm2d(n_out),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpConv(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3):
        super(UpConv, self).__init__()
        n_small = n_in // 2
        self.up = nn.ConvTranspose2d(n_small, n_small, 2, stride=2)
        padd = int((kernel_size - 1) / 2)
        self.conv = nn.Sequential(
            nn.Conv2d(n_in, n_out, kernel_size, padding=padd),
            nn.BatchNorm2d(n_out),
            nn.ReLU(),
            nn.Conv2d(n_out, n_out, kernel_size, padding=padd),
            nn.BatchNorm2d(n_out),
            nn.ReLU()
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)

        if x1.size()[2] != x2.size()[2] or x1.size()[3] != x2.size()[3]:
            diff_x = x2.size()[3] - x1.size()[3]
            diff_y = x2.size()[2] - x1.size()[2]

            x1 = F.pad(x1, (diff_x // 2, diff_x - diff_x // 2,
                            diff_y // 2, diff_y - diff_y // 2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, n_in, n_out):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(n_in, n_out, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

