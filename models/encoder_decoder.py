import torch
import torch.nn as nn


class illu_correction(nn.Module):
    def __init__(self, channels):
        super(illu_correction, self).__init__()

        self.conv_in = nn.Conv2d(3, channels, kernel_size=3, stride=1, padding=1)

        self.curve = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
                                   nn.LeakyReLU(),
                                   nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
                                   nn.AdaptiveAvgPool2d((1, 1)))

        self.conv_out = nn.Conv2d(channels, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        residual = x

        out = self.conv_in(x)

        amp_factor = self.curve(out)
        out = out * amp_factor

        out = self.conv_out(out) + residual

        return out


class Res_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Res_block, self).__init__()

        sequence = []

        sequence += [
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU(0.2)
        ]

        self.model = nn.Sequential(*sequence)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=0)

    def forward(self, x):
        out = self.model(x) + self.conv(x)

        return out


class upsampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(upsampling, self).__init__()

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1,
                                       output_padding=1)

        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.relu(self.conv(x))
        return out


class channel_down(nn.Module):
    def __init__(self, channels):
        super(channel_down, self).__init__()

        self.conv1 = nn.Conv2d(channels * 8, channels * 4, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv2 = nn.Conv2d(channels * 4, channels * 2, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv3 = nn.Conv2d(channels * 2, channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv4 = nn.Conv2d(channels, 3, kernel_size=(3, 3), stride=(1, 1), padding=1)

        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.relu(self.conv2(self.relu(self.conv1(x))))
        out = self.conv4(self.relu(self.conv3(out)))

        return out


class channel_up(nn.Module):
    def __init__(self, channels):
        super(channel_up, self).__init__()

        self.conv1 = nn.Conv2d(3, channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv2 = nn.Conv2d(channels, channels * 2, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv3 = nn.Conv2d(channels * 2, channels * 4, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv4 = nn.Conv2d(channels * 4, channels * 8, kernel_size=(3, 3), stride=(1, 1), padding=1)

        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.relu(self.conv2(self.relu(self.conv1(self.relu(x)))))
        out = self.conv4(self.relu(self.conv3(out)))

        return out


class RAW_encoder(nn.Module):
    def __init__(self, channels):
        super(RAW_encoder, self).__init__()

        self.convs = nn.Sequential(nn.Conv2d(4, channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
                                   nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=(1, 1), padding=1))

        self.block0 = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
                                    nn.LeakyReLU(0.2),
                                    nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=(1, 1), padding=1))

        self.block1 = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
                                    nn.LeakyReLU(0.2),
                                    nn.Conv2d(channels, channels * 2, kernel_size=(3, 3), stride=(1, 1), padding=1))

        self.block2 = nn.Sequential(nn.Conv2d(channels * 2, channels * 2, kernel_size=(3, 3), stride=(1, 1), padding=1),
                                    nn.LeakyReLU(0.2),
                                    nn.Conv2d(channels * 2, channels * 4, kernel_size=(3, 3), stride=(1, 1), padding=1))

        self.block3 = nn.Sequential(nn.Conv2d(channels * 4, channels * 4, kernel_size=(3, 3), stride=(1, 1), padding=1),
                                    nn.LeakyReLU(0.2),
                                    nn.Conv2d(channels * 4, channels * 8, kernel_size=(3, 3), stride=(1, 1), padding=1))

        self.down0 = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        self.down1 = nn.Conv2d(channels * 2, channels * 2, kernel_size=3, stride=2, padding=1)
        self.down2 = nn.Conv2d(channels * 4, channels * 4, kernel_size=3, stride=2, padding=1)
        self.down3 = nn.Conv2d(channels * 8, channels * 8, kernel_size=3, stride=2, padding=1)

        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        pyramid = {}

        level0 = self.down0(self.block0(self.convs(x)))
        level1 = self.down1(self.block1(level0))
        level2 = self.down2(self.block2(level1))
        level3 = self.down3(self.block3(level2))

        pyramid["down2"] = level0
        pyramid["down4"] = level1
        pyramid["down8"] = level2
        pyramid["down16"] = level3

        return pyramid


class RAW_decoder(nn.Module):
    def __init__(self, channels):
        super(RAW_decoder, self).__init__()

        self.block_up0 = Res_block(channels * 8, channels * 8)
        self.block_up1 = Res_block(channels * 8, channels * 8)
        self.up_sampling0 = upsampling(channels * 8, channels * 4)
        self.block_up2 = Res_block(channels * 4, channels * 4)
        self.block_up3 = Res_block(channels * 4, channels * 4)
        self.up_sampling1 = upsampling(channels * 4, channels * 2)
        self.block_up4 = Res_block(channels * 2, channels * 2)
        self.block_up5 = Res_block(channels * 2, channels * 2)
        self.up_sampling2 = upsampling(channels * 2, channels)
        self.block_up6 = Res_block(channels, channels)
        self.block_up7 = Res_block(channels, channels)
        self.up_sampling3 = upsampling(channels, channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv3 = nn.Conv2d(channels, 4, kernel_size=(1, 1), stride=(1, 1), padding=0)

        self.relu = nn.LeakyReLU(0.2)

    def forward(self, raw_fea_ori, pyramid):
        out_put = {}

        raw_fea2, raw_fea4, raw_fea8, raw_fea16 = \
            pyramid["down2"], pyramid["down4"], pyramid["down8"], pyramid["down16"]

        raw_fea_decode16 = self.block_up1(self.block_up0(raw_fea_ori) + raw_fea16)
        raw_fea_ori_up2 = self.up_sampling0(raw_fea_decode16)
        raw_fea_decode8 = self.block_up3(self.block_up2(raw_fea_ori_up2) + raw_fea8)
        raw_fea_ori_up4 = self.up_sampling1(raw_fea_decode8)
        raw_fea_decode4 = self.block_up5(self.block_up4(raw_fea_ori_up4) + raw_fea4)
        raw_fea_ori_up8 = self.up_sampling2(raw_fea_decode4)
        raw_fea_decode2 = self.block_up7(self.block_up6(raw_fea_ori_up8) + raw_fea2)
        raw_fea_ori_up16 = self.up_sampling3(raw_fea_decode2)

        raw_img_ori = self.conv3(self.relu(self.conv2(raw_fea_ori_up16)))

        return raw_img_ori 


class RGB_encoder(nn.Module):
    def __init__(self, channels):
        super(RGB_encoder, self).__init__()

        self.convs = nn.Sequential(nn.Conv2d(12, channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
                                   nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=(1, 1), padding=1))

        self.block0 = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
                                    nn.LeakyReLU(0.2),
                                    nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=(1, 1), padding=1))

        self.block1 = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
                                    nn.LeakyReLU(0.2),
                                    nn.Conv2d(channels, channels * 2, kernel_size=(3, 3), stride=(1, 1), padding=1))

        self.block2 = nn.Sequential(nn.Conv2d(channels * 2, channels * 2, kernel_size=(3, 3), stride=(1, 1), padding=1),
                                    nn.LeakyReLU(0.2),
                                    nn.Conv2d(channels * 2, channels * 4, kernel_size=(3, 3), stride=(1, 1), padding=1))

        self.block3 = nn.Sequential(nn.Conv2d(channels * 4, channels * 4, kernel_size=(3, 3), stride=(1, 1), padding=1),
                                    nn.LeakyReLU(0.2),
                                    nn.Conv2d(channels * 4, channels * 8, kernel_size=(3, 3), stride=(1, 1), padding=1))

        self.down0 = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        self.down1 = nn.Conv2d(channels * 2, channels * 2, kernel_size=3, stride=2, padding=1)
        self.down2 = nn.Conv2d(channels * 4, channels * 4, kernel_size=3, stride=2, padding=1)
        self.down3 = nn.Conv2d(channels * 8, channels * 8, kernel_size=3, stride=2, padding=1)

        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        level0 = self.down0(self.block0(self.convs(x)))
        level1 = self.down1(self.block1(level0))
        level2 = self.down2(self.block2(level1))
        level3 = self.down3(self.block3(level2))

        return level3


class RGB_decoder(nn.Module):
    def __init__(self, channels):
        super(RGB_decoder, self).__init__()

        self.block_up0 = Res_block(channels * 8, channels * 8)
        self.block_up1 = Res_block(channels * 8, channels * 8)
        self.up_sampling0 = upsampling(channels * 8, channels * 4)
        self.block_up2 = Res_block(channels * 4, channels * 4)
        self.block_up3 = Res_block(channels * 4, channels * 4)
        self.up_sampling1 = upsampling(channels * 4, channels * 2)
        self.block_up4 = Res_block(channels * 2, channels * 2)
        self.block_up5 = Res_block(channels * 2, channels * 2)
        self.up_sampling2 = upsampling(channels * 2, channels)
        self.block_up6 = Res_block(channels, channels)
        self.block_up7 = Res_block(channels, channels)
        self.up_sampling3 = upsampling(channels, channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv3 = nn.Conv2d(channels, 12, kernel_size=(1, 1), stride=(1, 1), padding=0)

        self.relu = nn.LeakyReLU(0.2)

    def forward(self, rgb_fea_ori, raw_pyramid):
        raw_fea2, raw_fea4, raw_fea8, raw_fea16 = \
            raw_pyramid["down2"], raw_pyramid["down4"], \
                raw_pyramid["down8"], raw_pyramid["down16"]

        rgb_fea_ori_up2 = self.up_sampling0(
            self.block_up1(self.block_up0(rgb_fea_ori) + raw_fea16))
        rgb_fea_ori_up4 = self.up_sampling1(
            self.block_up3(self.block_up2(rgb_fea_ori_up2) + raw_fea8))
        rgb_fea_ori_up8 = self.up_sampling2(
            self.block_up5(self.block_up4(rgb_fea_ori_up4) + raw_fea4))
        rgb_fea_ori_up16 = self.up_sampling3(
            self.block_up7(self.block_up6(rgb_fea_ori_up8) + raw_fea2))

        rgb_img_ori = self.conv3(self.relu(self.conv2(rgb_fea_ori_up16)))

        return rgb_img_ori


class ReconNet(nn.Module):
    def __init__(self, channels=64, training_stage='stage1'):
        super(ReconNet, self).__init__()

        self.RAW_encoder = RAW_encoder(channels)
        self.RAW_channel_down = channel_down(channels)
        self.RAW_illu = illu_correction(channels)
        self.RAW_channel_up = channel_up(channels)
        self.RAW_decoder = RAW_decoder(channels)

        self.RGB_encoder = RGB_encoder(channels)
        self.RGB_channel_down = channel_down(channels)
        self.RGB_channel_up = channel_up(channels)
        self.RGB_decoder = RGB_decoder(channels)

        self.training_stage = training_stage

    def forward(self, input, pred_fea=None):
        data_dict = {}

        if self.training_stage == 'stage1':
            low_RAW_img, high_RAW_img, high_RGB_img = \
                input[:, :4, :, :], input[:, 4:8, :, :], input[:, 8:, :, :]

            # ================== low_RAW ==================

            low_raw_pyramid = self.RAW_encoder(low_RAW_img)
            low_raw_fea_16 = low_raw_pyramid["down16"]
            low_raw_fea_down16 = self.RAW_channel_down(low_raw_fea_16)
            low_raw_fea_ori = self.RAW_channel_up(torch.sigmoid(low_raw_fea_down16))
            low_raw_img_ori = self.RAW_decoder(low_raw_fea_ori, low_raw_pyramid)

            low_raw_fea_down16_relight = torch.sigmoid(self.RAW_illu(low_raw_fea_down16))

            # ================== high_RAW ==================
            high_raw_pyramid = self.RAW_encoder(high_RAW_img)
            high_raw_fea_16 = high_raw_pyramid["down16"]
            high_raw_fea_down16 = torch.sigmoid(self.RAW_channel_down(high_raw_fea_16))
            high_raw_fea_ori = self.RAW_channel_up(high_raw_fea_down16)
            high_raw_img_ori = self.RAW_decoder(high_raw_fea_ori, high_raw_pyramid)

            # ================== high_RGB ==================
            high_rgb_fea16 = self.RGB_encoder(high_RGB_img)
            high_rgb_fea_down16 = torch.sigmoid(self.RGB_channel_down(high_rgb_fea16))
            high_rgb_fea_ori = self.RGB_channel_up(high_rgb_fea_down16)
            high_rgb_img_ori = self.RGB_decoder(high_rgb_fea_ori + low_raw_fea_16, low_raw_pyramid)

            data_dict["low_raw_img_ori"] = low_raw_img_ori
            data_dict["high_raw_img_ori"] = high_raw_img_ori
            data_dict["high_rgb_img_ori"] = high_rgb_img_ori

            data_dict["low_raw_fea"] = torch.sigmoid(low_raw_fea_down16)
            data_dict["low_raw_fea_relight"] = low_raw_fea_down16_relight
            data_dict["gt_raw_fea"] = high_raw_fea_down16

        else:
            low_RAW_img, high_RGB_img = input[:, :4, :, :], input[:, 4:, :, :]

            low_raw_pyramid = self.RAW_encoder(low_RAW_img)
            low_raw_fea_16 = low_raw_pyramid["down16"]
            if pred_fea is None:
                low_raw_fea_down16 = self.RAW_channel_down(low_raw_fea_16)
                low_raw_fea_down16 = torch.sigmoid(self.RAW_illu(low_raw_fea_down16))

                high_rgb_fea16 = self.RGB_encoder(high_RGB_img)
                high_rgb_fea_down16 = torch.sigmoid(self.RGB_channel_down(high_rgb_fea16))
                data_dict["raw_fea"] = low_raw_fea_down16
                data_dict["gt_fea"] = high_rgb_fea_down16
            else:
                pred_fea_ori = self.RGB_channel_up(pred_fea)
                pred_img_ori = self.RGB_decoder(pred_fea_ori + low_raw_fea_16, low_raw_pyramid)
                data_dict["pred_img"] = pred_img_ori

        return data_dict
