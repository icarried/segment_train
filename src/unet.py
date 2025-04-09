import torch
import torch.nn.functional as F
import torch.nn as nn
__all__ = ['UNet']


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, channels=[16, 32, 64, 128, 256]):
        super(UNet, self).__init__()
        self.encoder1 = ConvBlock(in_channels, channels[0])
        self.encoder2 = ConvBlock(channels[0], channels[1])
        self.encoder3 = ConvBlock(channels[1], channels[2])
        self.encoder4 = ConvBlock(channels[2], channels[3])
        self.encoder5 = ConvBlock(channels[3], channels[4])

        self.upconv4 = nn.ConvTranspose2d(channels[4], channels[3], kernel_size=2, stride=2)
        self.decoder4 = ConvBlock(channels[3] + channels[3], channels[3])
        
        self.upconv3 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=2, stride=2)
        self.decoder3 = ConvBlock(channels[2] + channels[2], channels[2])
        
        self.upconv2 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=2, stride=2)
        self.decoder2 = ConvBlock(channels[1] + channels[1], channels[1])
        
        self.upconv1 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=2, stride=2)
        self.decoder1 = ConvBlock(channels[0] + channels[0], channels[0])

        self.final_conv = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, 2))
        enc3 = self.encoder3(F.max_pool2d(enc2, 2))
        enc4 = self.encoder4(F.max_pool2d(enc3, 2))
        enc5 = self.encoder5(F.max_pool2d(enc4, 2))

        # Decoder
        up4 = self.upconv4(enc5)
        dec4 = self.decoder4(torch.cat([up4, enc4], dim=1))
        
        up3 = self.upconv3(dec4)
        dec3 = self.decoder3(torch.cat([up3, enc3], dim=1))
        
        up2 = self.upconv2(dec3)
        dec2 = self.decoder2(torch.cat([up2, enc2], dim=1))
        
        up1 = self.upconv1(dec2)
        dec1 = self.decoder1(torch.cat([up1, enc1], dim=1))

        return self.final_conv(dec1)

if __name__ == '__main__':
    input_tensor = torch.randn(4, 3, 256, 256)
    model = UNet(in_channels=3, out_channels=1)
    output_tensor = model(input_tensor)
    print("Input Tensor Shape:", input_tensor.shape)
    print("Output Tensor Shape:", output_tensor.shape)