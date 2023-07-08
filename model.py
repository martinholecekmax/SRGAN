import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        discriminator=False,
        use_activation=True,
        use_batchnorm=True,
        **kwargs
    ):
        super().__init__()
        self.use_activation = use_activation
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs, bias=not use_batchnorm)
        self.batchnorm = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.activation = (
            nn.LeakyReLU(0.2, inplace=True)
            if discriminator
            else nn.PReLU(num_parameters=out_channels)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        if self.use_activation:
            x = self.activation(x)
        return x


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, scale_factor=2):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels * scale_factor**2, kernel_size=3, stride=1, padding=1
        )
        self.pixel_shuffle = nn.PixelShuffle(
            scale_factor
        )  # in_channels * 4, H, W -> in_channels, H * 2, W * 2
        self.activation = nn.PReLU(num_parameters=in_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.activation(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block1 = ConvBlock(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.block2 = ConvBlock(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1, use_activation=False
        )

    def forward(self, x):
        residual = x
        x = self.block1(x)
        x = self.block2(x)
        x += residual
        return x


class Generator(nn.Module):
    def __init__(self, in_channels=3, num_channels=64, num_blocks=16):
        super().__init__()
        self.initial = ConvBlock(
            in_channels, num_channels, kernel_size=9, stride=1, padding=4, use_batchnorm=False
        )
        self.residuals = nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_blocks)])
        self.convBlock = ConvBlock(
            num_channels, num_channels, kernel_size=3, stride=1, padding=1, use_activation=False
        )
        self.upsample = nn.Sequential(
            UpsampleBlock(num_channels, scale_factor=2),
            UpsampleBlock(num_channels, scale_factor=2),
        )
        self.final = nn.Conv2d(num_channels, in_channels, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        x = self.initial(x)
        residual = x
        x = self.residuals(x)
        x = self.convBlock(x)
        x += residual
        x = self.upsample(x)
        x = self.final(x)
        return torch.tanh(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 64, 128, 128, 256, 256, 512, 512]):
        super().__init__()
        blocks = []
        for idx, feature in enumerate(features):
            stride = 1 + idx % 2
            use_batchnorm = False if idx < 2 else True
            blocks.append(
                ConvBlock(
                    in_channels,
                    feature,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    discriminator=True,
                    use_activation=True,
                    use_batchnorm=use_batchnorm,
                )
            )
            in_channels = feature

        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(512 * 6 * 6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        x = self.blocks(x)
        x = self.classifier(x)
        return x


def test():
    low_resolution = 24
    with torch.cuda.amp.autocast():
        x = torch.randn((5, 3, low_resolution, low_resolution))
        gen = Generator()
        disc = Discriminator()
        print(gen(x).shape)
        print(disc(x).shape)


if __name__ == "__main__":
    test()
