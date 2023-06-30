import numpy as np
import torch
import torch.nn as nn

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out

class PDSNet(nn.Module):
    def __init__(self):
        super(PDSNet, self).__init__()

        self.Feature_Encoder1_fe = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.Feature_Encoder1_down = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.ReLU())

        self.Feature_Encoder2_fe = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),  nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),  nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.Feature_Encoder2_down = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.ReLU())

        self.Feature_Encoder3_fe = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.Feature_Encoder3_down = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.ReLU())

        self.Feature_Encoder4_fe = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.Feature_Encoder4_down = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.ReLU())

        self.encoder_end = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),nn.ReLU())


        self.cbam1 = CBAM(channel=64)
        self.cbam2 = CBAM(channel=64)
        self.cbam3 = CBAM(channel=64)
        self.cbam4 = CBAM(channel=64)

        self.pool1 = nn.MaxPool2d(1)
        self.pool2 = nn.MaxPool2d(1)
        self.pool3 = nn.MaxPool2d(1)
        self.pool4 = nn.MaxPool2d(1)

        self.decoder_up4 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.Feature_Decoder4 = nn.Sequential(
            nn.Conv2d(128, 64, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1))

        self.decoder_up3 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.Feature_Decoder3 = nn.Sequential(
            nn.Conv2d(128, 64, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1))

        self.decoder_up2 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.Feature_Decoder2 = nn.Sequential(
            nn.Conv2d(128, 64, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1))

        self.decoder_up1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.Feature_Decoder1 = nn.Sequential(
            nn.Conv2d(128, 64, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1))

        self.Feature_Decoder_end = nn.Conv2d(64, 1, 3, padding=1)

        self.delta_1 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.gamma_1 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.eta_1 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.delta_2 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.gamma_2 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.eta_2 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.delta_3 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.gamma_3 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.eta_3 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.delta_4 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.gamma_4 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.eta_4 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.delta_5 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.gamma_5 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.eta_5 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.delta_6 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.gamma_6 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.eta_6 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)

        self.delta_1.data = torch.tensor(0.1)
        self.gamma_1.data = torch.tensor(0.5)
        self.eta_1.data = torch.tensor(0.9)
        self.delta_2.data = torch.tensor(0.1)
        self.gamma_2.data = torch.tensor(0.5)
        self.eta_2.data = torch.tensor(0.9)
        self.delta_3.data = torch.tensor(0.1)
        self.gamma_3.data = torch.tensor(0.5)
        self.eta_3.data = torch.tensor(0.9)
        self.delta_4.data = torch.tensor(0.1)
        self.gamma_4.data = torch.tensor(0.5)
        self.eta_4.data = torch.tensor(0.9)
        self.delta_5.data = torch.tensor(0.1)
        self.gamma_5.data = torch.tensor(0.5)
        self.eta_5.data = torch.tensor(0.9)
        self.delta_6.data = torch.tensor(0.1)
        self.gamma_6.data = torch.tensor(0.5)
        self.eta_6.data = torch.tensor(0.9)

    def forward(self, input):
        x = input
        y = input

        for i in range(6):
            f1 = self.Feature_Encoder1_fe(x)
            dow1 = self.Feature_Encoder1_down(f1)
            down1 = self.cbam1(dow1) + dow1
            pool1 = self.pool1(down1)

            f2 = self.Feature_Encoder2_fe(pool1)
            dow2 = self.Feature_Encoder2_down(f2)
            down2 = self.cbam2(dow2) + dow2
            pool2 = self.pool2(down2)

            f3 = self.Feature_Encoder3_fe(pool2)
            dow3 = self.Feature_Encoder1_down(f3)
            down3 = self.cbam3(dow3) + dow3
            pool3 = self.pool3(down3)

            f4 = self.Feature_Encoder4_fe(pool3)
            dow4 = self.Feature_Encoder2_down(f4)
            down4 = self.cbam4(dow4) + dow4
            pool4 = self.pool4(down4)

            media_end = self.encoder_end(pool4)


            up4 = self.decoder_up4(media_end)
            concat4 = torch.cat([up4, f4], dim=1)
            decoder4 = self.Feature_Decoder4(concat4)

            up3 = self.decoder_up3(decoder4)
            concat3 = torch.cat([up3, f3], dim=1)
            decoder3 = self.Feature_Decoder3(concat3)

            up2 = self.decoder_up2(decoder3)
            concat2 = torch.cat([up2, f2], dim=1)
            decoder2 = self.Feature_Decoder2(concat2)

            up1 = self.decoder_up1(decoder2)
            concat1 = torch.cat([up1, f1], dim=1)
            decoder1 = self.Feature_Decoder1(concat1)

            v = self.Feature_Decoder_end(decoder1)

            v = v + x

            x = self.reconnect(v, x, y, i)

        return x

    def reconnect(self, v, x, y, i):

        i = i + 1
        if i == 1:
            delta = self.delta_1
            eta = self.eta_1
            gamma = self.gamma_1
        if i == 2:
            delta = self.delta_2
            eta = self.eta_2
            gamma = self.gamma_2
        if i == 3:
            delta = self.delta_3
            eta = self.eta_3
            gamma = self.gamma_3
        if i == 4:
            delta = self.delta_4
            eta = self.eta_4
            gamma = self.gamma_4
        if i == 5:
            delta = self.delta_5
            eta = self.eta_5
            gamma = self.gamma_5
        if i == 6:
            delta = self.delta_6
            eta = self.eta_6
            gamma = self.gamma_6

        es = 0
        dk = torch.mul((2 - delta * eta - delta), x) - y - torch.mul((1 - delta - delta * eta), v)
        es = es + dk
        hess = 1 + eta
        recon = x - ((dk + torch.mul(gamma, es)) / hess)
        return recon


if __name__ == '__main__':
    input1 = torch.rand(1, 1, 128, 128)
    net = PSDNet()
    out = net(input1)
    print(out.size())


