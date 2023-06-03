import torch
import torch.nn as nn
from dotmap import DotMap
from trf import compute_trf
from typing import Tuple


class ConvBlock(nn.Module):
    """
    A convolutional block that consists of two convolutional layers with batch normalization and ReLU activation.

    Args:
        in_c (int): The number of input channels.
        out_c (int): The number of output channels.
        conf (DotMap): A configuration object that contains the hyperparameters for the convolutional layers.

    Attributes:
        conv (nn.Sequential): A sequential container that holds the convolutional layers, batch normalization layers, and ReLU activation functions.
        n_submodules (int): The number of submodules in the convolutional block.

    Methods:
        forward(x): Performs a forward pass through the convolutional block.

    """

    def __init__(self, in_c, out_c, conf):
        """
        Initializes the ConvBlock object.

        Args:
            in_c (int): The number of input channels.
            out_c (int): The number of output channels.
            conf (DotMap): A configuration object that contains the hyperparameters for the convolutional layers.

        """
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=conf.conv_k,
                      padding=conf.conv_p, stride=conf.conv_s),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=conf.conv_k,
                      padding=conf.conv_p, stride=conf.conv_s),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

        self.n_submodules = 6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the convolutional block.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        return self.conv(x)


class EncoderBlock(nn.Module):
    """
    An encoder block that consists of a convolutional layer, batch normalization, ReLU activation, and max pooling.

    Args:
        in_c (int): The number of input channels.
        out_c (int): The number of output channels.
        conf (DotMap): A configuration object that contains the hyperparameters for the convolutional layer and max pooling.

    Attributes:
        conv (ConvBlock): A convolutional block that consists of two convolutional layers with batch normalization and ReLU activation.
        pool (nn.MaxPool2d): A max pooling layer that reduces the spatial dimensions of the input tensor.

    Methods:
        forward(x): Performs a forward pass through the encoder block.

    """

    def __init__(self, in_c: int, out_c: int, conf: DotMap):
        """
        Initializes the EncoderBlock object.

        Args:
            in_c (int): The number of input channels.
            out_c (int): The number of output channels.
            conf (DotMap): A configuration object that contains the hyperparameters for the convolutional layer and max pooling.

        """
        super().__init__()

        self.conv = ConvBlock(in_c, out_c, conf)
        self.pool = nn.MaxPool2d((conf.pool_k, conf.pool_k))

        self.n_submodules = self.conv.n_submodules + 1

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass through the encoder block.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the output tensor from the convolutional block and the output tensor from the max pooling layer.

        """
        s = self.conv(x)
        p = self.pool(s)
        return s, p


class DecoderBlock(nn.Module):
    """
    A decoder block that consists of a transposed convolutional layer, concatenation with skip connections, and a convolutional block.

    Args:
        in_c (int): The number of input channels.
        out_c (int): The number of output channels.
        attention (bool): A boolean indicating whether to use an attention block.
        conf (DotMap): A configuration object that contains the hyperparameters for the transposed convolutional layer.

    Attributes:
        has_attention (bool): A boolean indicating whether the decoder block has an attention block.
        attention (AttentionBlock): An attention block that computes attention weights between the skip connection and the transposed convolutional layer.
        up (nn.ConvTranspose2d): A transposed convolutional layer that upsamples the input tensor.
        conv (ConvBlock): A convolutional block that consists of two convolutional layers with batch normalization and ReLU activation.
        n_submodules (int): The number of submodules in the decoder block.

    Methods:
        forward(inputs, skip): Performs a forward pass through the decoder block.

    """

    def __init__(self, in_c: int, out_c: int, attention: bool, conf: DotMap):
        """
        Initializes the DecoderBlock object.

        Args:
            in_c (int): The number of input channels.
            out_c (int): The number of output channels.
            attention (bool): A boolean indicating whether to use an attention block.
            conf (DotMap): A configuration object that contains the hyperparameters for the transposed convolutional layer.

        """
        super().__init__()

        self.has_attention = attention

        if attention:
            self.attention = AttentionBlock(out_c)
        self.up = nn.ConvTranspose2d(
            in_c, out_c, kernel_size=conf.deconv_k, padding=conf.deconv_p, stride=conf.deconv_s)
        self.conv = ConvBlock(out_c + out_c, out_c, conf)

        self.n_submodules = self.conv.n_submodules + 1

    def forward(self, inputs, skip):
        """
        Performs a forward pass through the decoder block.

        Args:
            inputs (torch.Tensor): The input tensor.
            skip (torch.Tensor): The skip connection tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        a = self.up(inputs)

        if self.has_attention:
            b = self.attention(skip, a)
        else:
            b = skip

        x = torch.cat([a, b], axis=1)
        x = self.conv(x)

        return x


class BottleNeck(nn.Module):
    """
    A bottleneck block that consists of a convolutional block.

    Args:
        in_c (int): The number of input channels.
        out_c (int): The number of output channels.
        conf (DotMap): A configuration object that contains the hyperparameters for the convolutional block.

    Attributes:
        conv (ConvBlock): A convolutional block that consists of two convolutional layers with batch normalization and ReLU activation.
        n_submodules (int): The number of submodules in the bottleneck block.

    Methods:
        forward(x): Performs a forward pass through the bottleneck block.

    """

    def __init__(self, in_c: int, out_c: int, conf: DotMap):
        """
        Initializes the BottleNeck object.

        Args:
            in_c (int): The number of input channels.
            out_c (int): The number of output channels.
            conf (DotMap): A configuration object that contains the hyperparameters for the convolutional block.

        """
        super().__init__()

        self.conv = ConvBlock(in_c, out_c, conf)
        self.n_submodules = self.conv.n_submodules

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the bottleneck block.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        return self.conv(x)


class AttentionBlock(nn.Module):
    """
    An attention block that computes attention weights between the skip connection and the gating tensor.

    Args:
        channels (int): The number of channels in the input tensors.

    Attributes:
        conv (nn.Conv2d): A convolutional layer that reduces the number of channels in the input tensors.
        relu (nn.ReLU): A ReLU activation function.
        sigmoid (nn.Sigmoid): A sigmoid activation function.

    Methods:
        forward(skip, gating): Computes attention weights between the skip connection and the gating tensor.

    """

    def __init__(self, channels: int):
        """
        Initializes the AttentionBlock object.

        Args:
            channels (int): The number of channels in the input tensors.

        """
        super().__init__()

        self.conv = nn.Conv2d(channels, channels, kernel_size=(1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, skip: torch.Tensor, gating: torch.Tensor) -> torch.Tensor:
        """
        Computes attention weights between the skip connection and the gating tensor.

        Args:
            skip (torch.Tensor): The skip connection tensor.
            gating (torch.Tensor): The gating tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        a = self.conv(skip)
        b = self.conv(gating)
        x = a + b
        x = self.relu(x)
        x = self.sigmoid(x)
        x = gating * x
        return x


class UNet(nn.Module):
    """
    A UNet model that consists of a series of encoder and decoder blocks.

    Args:
        conf (DotMap): A configuration object that contains the hyperparameters for the model.
        attention (bool): A boolean indicating whether to use attention blocks.
        n_labels (int): The number of output labels.

    Attributes:
        encoders (nn.ModuleList): A list of encoder blocks.
        b (BottleNeck): A bottleneck block that consists of a convolutional block.
        decoders (nn.ModuleList): A list of decoder blocks.
        outputs (nn.Conv2d): A convolutional layer that produces the output tensor.

    Methods:
        total_parameters(): Computes the total number of parameters in the network.
        max_trf_size(): Computes the maximum theoretical receptive field of the network.
        output_trfs(): Computes the theoretical receptive fields of the output tensor.
        pixel_trf(x, y): Computes the theoretical receptive field of a pixel in the output tensor.
        center_trf(): Computes the theoretical receptive field of the center pixel in the output tensor.
        forward(inputs): Performs a forward pass through the UNet model.

    """

    def __init__(self, conf: DotMap, attention: bool = False, n_labels: int = 1):
        """
        Initializes the UNet object.

        Args:
            conf (DotMap): A configuration object that contains the hyperparameters for the model.
            attention (bool): A boolean indicating whether to use attention blocks.
            n_labels (int): The number of output labels.

        """
        super().__init__()
        self.config = conf
        self.attention = attention
        c_in = 1 if conf.grayscale else 3

        self.encoders = nn.ModuleList([])
        self.b = BottleNeck(conf.channels[-2], conf.channels[-1], conf)
        self.decoders = nn.ModuleList([])

        for i in range(conf.depth):
            if i == 0:
                self.encoders.append(EncoderBlock(
                    c_in, conf.channels[i], conf))
            else:
                self.encoders.append(EncoderBlock(
                    conf.channels[i-1], conf.channels[i], conf))

        for i in range(conf.depth, 0, -1):
            self.decoders.append(DecoderBlock(
                conf.channels[i], conf.channels[i-1], attention, conf))

        self.outputs = nn.Conv2d(
            conf.channels[0], n_labels, kernel_size=1, padding=0, stride=1)

    def total_parameters(self) -> int:
        """
        Computes the total number of parameters in the network.

        Returns:
            int: The total number of parameters in the network.

        """
        return sum(p.numel() for p in self.parameters())

    def max_trf_size(self) -> int:
        """
        Computes the maximum theoretical receptive field of the network.

        Returns:
            int: The maximum theoretical receptive field of the network.

        """
        rf = compute_trf(self, 576)
        return rf[next(reversed(rf))]["max_trf_size"]

    def output_trfs(self):
        """
        Computes the theoretical receptive fields of the output tensor.

        Returns:
            dict: A dictionary containing the theoretical receptive fields of the output tensor.

        """
        rf = compute_trf(self, 576)
        return rf[next(reversed(rf))]["trf"]

    def pixel_trf(self, x, y):
        """
        Computes the theoretical receptive field of a pixel in the output tensor.

        Args:
            x (int): The x-coordinate of the pixel.
            y (int): The y-coordinate of the pixel.

        Returns:
            int: The theoretical receptive field of the pixel.

        """
        rf = compute_trf(self, 576)
        return rf[next(reversed(rf))]["trf"][x, y]

    def center_trf(self):
        """
        Computes the theoretical receptive field of the center pixel in the output tensor.

        Returns:
            int: The theoretical receptive field of the center pixel.

        """
        center = 576 // 2
        return self.pixel_trf(center, center)

    def forward(self, inputs):
        """
        Performs a forward pass through the UNet model.

        Args:
            inputs (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        skip_connections = []

        for e in self.encoders:
            skip, inputs = e(inputs)
            skip_connections.append(skip)

        b = self.b(inputs)

        for d in self.decoders:
            b = d(b, skip_connections.pop())

        outputs = self.outputs(b)

        return outputs
