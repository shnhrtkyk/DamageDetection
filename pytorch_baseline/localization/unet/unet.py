import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.in_conv = InConv(n_channels, 64)
        self.down1 = DownConv(64, 128)
        self.down2 = DownConv(128, 256)
        self.down3 = DownConv(256, 512)
        self.atrous = AtrousConv(512, 256)
        self.up3 = UpConv(512, 128)
        self.up2 = UpConv(256, 64)
        self.up1 = UpConv(128, 64, kernel_size=5)
        self.out_conv = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.atrous(x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        x = self.out_conv(x)
        return torch.sigmoid(x)


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

