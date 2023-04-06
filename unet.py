import torch
import torch.nn as nn

from dotmap import DotMap


class ConvBlock(nn.Module):
    """A convolutional block with two convolutional layers and batch normalization"""
    def __init__(self, in_c, out_c, conf1, conf2):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=conf1.k, padding=conf1.p, stride=conf1.s),
            nn.BatchNorm2d(out_c), # TODO: fix batch size >1
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=conf2.k, padding=conf2.p, stride=conf2.s),
            nn.BatchNorm2d(out_c), # TODO: fix batch size >
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c, conf):
        super().__init__()

        self.conv = ConvBlock(in_c, out_c, conf.conv1, conf.conv2)
        self.pool = nn.MaxPool2d((conf.pool_k, conf.pool_k))

    def forward(self, x):
        s = self.conv(x)
        p = self.pool(s)
        return s, p


class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c, conf):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=conf.up.k, padding=conf.up.p, stride=conf.up.s)
        self.conv = ConvBlock(out_c + out_c, out_c, conf.conv1, conf.conv2)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)

        return x


class UNet(nn.Module):
    def __init__(self, conf: DotMap):
        super().__init__()

        c_in = 1 if conf.grayscale else 3

        self.e1 = EncoderBlock(c_in, 64, conf.enc1)
        self.e2 = EncoderBlock(64, 128, conf.enc2)
        self.e3 = EncoderBlock(128, 256, conf.enc3)
        self.e4 = EncoderBlock(256, 512, conf.enc4)

        self.b = ConvBlock(512, 1024, conf.b.conv1, conf.b.conv2)

        self.d1 = DecoderBlock(1024, 512, conf.dec1)
        self.d2 = DecoderBlock(512, 256, conf.dec2)
        self.d3 = DecoderBlock(256, 128, conf.dec3)
        self.d4 = DecoderBlock(128, 64, conf.dec4)

        self.outputs = nn.Conv2d(64, 1, kernel_size=conf.out.k, padding=conf.out.p, stride=conf.out.s)


    def compute_trf(self, x, y):
        """Compute the theoretical receptive field of a pixel in the output"""
        # TODO
        pass


    def compute_erf(self, x, y):
        """Compute the effective receptive field of a pixel in the output"""
        # TODO
        pass


    def forward(self, inputs):
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        b = self.b(p4)

        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        outputs = self.outputs(d4)

        return outputs