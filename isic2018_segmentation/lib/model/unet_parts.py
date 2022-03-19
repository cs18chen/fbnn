import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, p=0.2):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.Dropout2d(p),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.Dropout2d(p),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class CARAFE(nn.Module):
    def __init__(self, inC, outC, Kencoder=3, delta=2, Kup=5, Cm=64):  # Kup = Kencoder + 2
        super(CARAFE, self).__init__()
        self.Kencoder = Kencoder
        self.delta = delta
        self.Kup = Kup
        self.down = nn.Conv2d(in_channels=inC, out_channels=Cm, kernel_size=1)  #
        self.encoder = nn.Conv2d(64, self.delta ** 2 * self.Kup ** 2,
                                 self.Kencoder, 1, self.Kencoder // 2)
        self.out = nn.Conv2d(inC, outC, 1)

    def forward(self, in_tensor):
        N, C, H, W = in_tensor.size()

        kernel_tensor = self.down(in_tensor)
        kernel_tensor = self.encoder(kernel_tensor)
        kernel_tensor = F.pixel_shuffle(kernel_tensor, self.delta)
        kernel_tensor = F.softmax(kernel_tensor, dim=1)
        kernel_tensor = kernel_tensor.unfold(2, self.delta, step=self.delta)
        kernel_tensor = kernel_tensor.unfold(3, self.delta, step=self.delta)
        kernel_tensor = kernel_tensor.reshape(N, self.Kup ** 2, H, W, self.delta ** 2)
        kernel_tensor = kernel_tensor.permute(0, 2, 3, 1, 4)

        in_tensor = F.pad(in_tensor, pad=(self.Kup // 2, self.Kup // 2, self.Kup // 2, self.Kup // 2), mode='constant', value=0)
        in_tensor = in_tensor.unfold(dimension=2, size=self.Kup, step=1)
        in_tensor = in_tensor.unfold(3, self.Kup, step=1)
        in_tensor = in_tensor.reshape(N, C, H, W, -1)
        in_tensor = in_tensor.permute(0, 2, 3, 1, 4)

        out_tensor = torch.matmul(in_tensor, kernel_tensor)
        out_tensor = out_tensor.reshape(N, H, W, -1)
        out_tensor = out_tensor.permute(0, 3, 1, 2)
        out_tensor = F.pixel_shuffle(out_tensor, self.delta)
        out_tensor = self.out(out_tensor)
        return out_tensor


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = CARAFE(in_channels , in_channels // 2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)




