import torch
import torch.nn as nn
from dotmap import DotMap
from trf import compute_trf


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, conf):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=conf.conv_k, padding=conf.conv_p, stride=conf.conv_s),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=conf.conv_k, padding=conf.conv_p, stride=conf.conv_s),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

        self.n_submodules = 6

    def forward(self, x):
        return self.conv(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c, conf):
        super().__init__()
        self.conv = ConvBlock(in_c, out_c, conf)
        self.pool = nn.MaxPool2d((conf.pool_k, conf.pool_k))

        self.n_submodules = self.conv.n_submodules + 1

    def forward(self, x):
        s = self.conv(x)
        p = self.pool(s)
        return s, p


class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c, conf):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=conf.deconv_k, padding=conf.deconv_p, stride=conf.deconv_s)
        self.conv = ConvBlock(out_c + out_c, out_c, conf)

        self.n_submodules = self.conv.n_submodules + 1

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)

        return x

class BottleNeck(nn.Module):
    def __init__(self, in_c, out_c, conf):
        super().__init__()
        self.conv = ConvBlock(in_c, out_c, conf)
        self.n_submodules = self.conv.n_submodules

    def forward(self, x):
        return self.conv(x)



class UNet(nn.Module):
    def __init__(self, conf: DotMap):
        super().__init__()
        self.config = conf
        c_in = 1 if conf.grayscale else 3

        self.encoders = nn.ModuleList([])
        self.b = BottleNeck(conf.channels[-2], conf.channels[-1], conf)
        self.decoders = nn.ModuleList([])

        for i in range(conf.depth):
            if i == 0:
                self.encoders.append(EncoderBlock(c_in, conf.channels[i], conf))
            else:
                self.encoders.append(EncoderBlock(conf.channels[i-1], conf.channels[i], conf))

        for i in range(conf.depth, 0, -1):
            self.decoders.append(DecoderBlock(conf.channels[i], conf.channels[i-1], conf))

        self.outputs = nn.Conv2d(conf.channels[0], 1, kernel_size=1, padding=0, stride=1)


    def total_parameters(self) -> int:
        """Compute the total number of parameters in the network"""
        return sum(p.numel() for p in self.parameters())


    def max_trf_size(self) -> int:
        """Compute the maximum theoretical receptive field of the network"""
        rf = compute_trf(self, 576)
        return rf[next(reversed(rf))]["max_trf_size"]


    def output_trfs(self):
        rf = compute_trf(self, 576)
        return rf[next(reversed(rf))]["trf"]


    def pixel_trf(self, x, y):
        """Compute the theoretical receptive field of a pixel in the output"""
        rf = compute_trf(self, 576)
        return rf[next(reversed(rf))]["trf"][x, y]


    def center_trf(self):
        """Compute the theoretical receptive field of the center pixel in the output"""
        center = 576 // 2
        return self.pixel_trf(center, center)


    def pixel_erf(self, x, y):
        """Compute the effective receptive field of a pixel in the output"""
        # TODO
        pass


    def forward(self, inputs):
        skip_connections = []

        for e in self.encoders:
            skip, inputs = e(inputs)
            skip_connections.append(skip)

        b = self.b(inputs)

        for d in self.decoders:
            b = d(b, skip_connections.pop())

        outputs = self.outputs(b)

        return outputs
