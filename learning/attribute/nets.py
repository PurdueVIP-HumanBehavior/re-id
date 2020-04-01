import torch
import torchvision
from torch import nn


class CbrBlock(nn.Module):
    def __init__(self, kernel_size, out_channels, in_channels, padding):
        super(CbrBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              padding=padding)
        self.bn = nn.BatchNorm2d(track_running_stats=True,
                                 num_features=out_channels)
        self.relu = nn.ReLU()

    def __call__(self, x):
        return self.relu(self.bn(self.conv(x)))


class EncodeBlock(nn.Module):
    def __init__(self,
                 kernel_size,
                 out_channels,
                 in_channels,
                 padding,
                 ds_kernel_size=2):
        super(EncodeBlock, self).__init__()
        self.cbr = CbrBlock(kernel_size=kernel_size,
                            in_channels=in_channels,
                            out_channels=out_channels,
                            padding=padding)
        self.maxpool = nn.MaxPool2d(kernel_size=ds_kernel_size)

    def __call__(self, x):
        return self.maxpool(self.cbr(x))


class DecodeBlock(nn.Module):
    def __init__(self,
                 kernel_size,
                 out_channels,
                 in_channels,
                 padding,
                 ds_kernel_size=2):
        super(DecodeBlock, self).__init__()
        self.cbr = CbrBlock(kernel_size=kernel_size,
                            in_channels=in_channels,
                            out_channels=out_channels,
                            padding=padding)
        self.upsample = nn.UpsamplingNearest2d()

    def __call__(self, x):
        return self.upsample(self.cbr(x))


class EDNet(nn.Module):
    def __init__(self,
                 input_shape,
                 num_classes,
                 downsampling_rate=2,
                 num_downsamples=5,
                 depth_growth_rate=2):
        super(EDNet, self).__init__()
        # TODO : (nhendy) sanity check input types and values
        height, width, channels = input_shape
        layers = []
        current_channels = channels
        for i in range(num_downsamples):
            layers.append(
                EncodeBlock(kernel_size=3,
                            in_channels=current_channels,
                            out_channels=16 * 2**i,
                            padding=1,
                            ds_kernel_size=2))
            current_channels = 16 * 2**i

        self.encoder = nn.Sequential(*layers)
        layers = []
        for i in reversed(range(num_downsamples - 1)):
            layers.append(
                DecodeBlock(kernel_size=3,
                            in_channels=current_channels,
                            out_channels=16 * 2**i,
                            padding=1,
                            ds_kernel_size=2))
            current_channels = 16 * 2**i

        self.decoder = nn.Sequential(*layers)
        self.fc = nn.Linear(in_features=(height // 2) * (width // 2) *
                            current_channels,
                            out_features=num_classes)

    def __call__(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


if __name__ == "__main__":
    net = EDNet(input_shape=(300, 320, 3), num_classes=10, num_downsamples=3)
    print(net)
