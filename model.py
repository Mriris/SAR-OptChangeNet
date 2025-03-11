import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torchvision.models as models
from torchvision.models.swin_transformer import swin_v2_t


class BiFPN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BiFPN, self).__init__()

        # 自下而上路径
        self.p3_td = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.p4_td = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.p5_td = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # 自上而下路径
        self.p3_out = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)  # 新增
        self.p4_out = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.p5_out = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        # 如果不需要 p6_out，可以移除
        self.p6_out = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # 权重
        self.w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 1e-4

    def forward(self, inputs):
        p3, p4, p5 = inputs

        # 自下而上
        w1 = F.relu(self.w1)
        w1 = w1 / (torch.sum(w1, dim=0) + self.epsilon)

        p4_td = self.p4_td(p4)
        p5_td = self.p5_td(p5)
        p4_td = w1[0] * p4_td + w1[1] * F.interpolate(p5_td, size=p4.shape[-2:], mode='nearest')
        p4_td = self.p4_out(p4_td)

        p3_td = self.p3_td(p3)
        p3_out = w1[0] * p3_td + w1[1] * F.interpolate(p4_td, size=p3.shape[-2:], mode='nearest')
        p3_out = self.p3_out(p3_out)  # 现在可以正常运行

        # 自上而下
        w2 = F.relu(self.w2)
        w2 = w2 / (torch.sum(w2, dim=0) + self.epsilon)

        p4_out = w2[0] * p4 + w2[1] * p4_td + w2[2] * F.adaptive_max_pool2d(p3_out, output_size=p4.shape[-2:])
        p4_out = self.p4_out(p4_out)

        p5_out = w2[0] * p5 + w2[1] * p5_td + w2[2] * F.adaptive_max_pool2d(p4_out, output_size=p5.shape[-2:])
        p5_out = self.p5_out(p5_out)

        return [p3_out, p4_out, p5_out]


class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super(FeaturePyramidNetwork, self).__init__()

        # 侧边连接
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for _ in range(3)  # 假设有3个尺度级别
        ])

        # 特征融合
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in range(3)
        ])

    def forward(self, features):
        # 从高层到低层的特征列表
        laterals = [
            lateral_conv(feature)
            for lateral_conv, feature in zip(self.lateral_convs, features)
        ]

        # 自顶向下路径
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode='nearest'
            )

        # 应用3x3卷积得到最终输出
        outs = [
            fpn_conv(lateral)
            for fpn_conv, lateral in zip(self.fpn_convs, laterals)
        ]

        return outs


class SwinFeatureExtractor(nn.Module):
    def __init__(self, in_channels=3):
        super(SwinFeatureExtractor, self).__init__()
        self.swin = swin_v2_t(weights="IMAGENET1K_V1")

        # 适应输入通道数
        if in_channels != 3:
            self.swin.features[0][0] = nn.Conv2d(
                in_channels, 96, kernel_size=4, stride=4
            )

        # 移除分类头
        self.feature_extractor = self.swin.features

    def forward(self, x):
        return self.feature_extractor(x)


class ImprovedTransformerChangeDetection(nn.Module):
    def __init__(self, hidden_dim=256, nhead=8, num_encoder_layers=6, dropout=0.1):
        super(ImprovedTransformerChangeDetection, self).__init__()

        self.bifpn = BiFPN(in_channels=hidden_dim, out_channels=hidden_dim)

        # 特征提取器
        self.branch1_encoder = self._build_optical_encoder()  # 时间点1的光学图像
        self.branch2_encoder = self._build_sar_encoder()  # 时间点2的SAR图像 (学生)
        self.branch3_encoder = self._build_optical_encoder()  # 时间点2的光学图像 (教师)

        # 特征投影
        self.projection1 = nn.Conv2d(2048, hidden_dim, kernel_size=1)  # ResNet50输出2048通道
        self.projection2 = nn.Conv2d(2048, hidden_dim, kernel_size=1)
        self.projection3 = nn.Conv2d(2048, hidden_dim, kernel_size=1)

        # Swin Transformer层
        self.swin_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout),
                nn.Dropout(dropout),
            )
            for _ in range(num_encoder_layers)
        ])

        # 增强的差异特征注意力
        self.diff_attention = EnhancedDifferenceAttentionModule(hidden_dim)

        # # 多尺度特征融合
        # self.fpn = FeaturePyramidNetwork(hidden_dim)

        # 改进的解码器
        self.decoder = DeepLabV3PlusDecoder(hidden_dim)

    def forward(self, t1_optical, t2_sar, t2_optical):
        # 特征提取
        f1 = self.branch1_encoder(t1_optical)
        f2 = self.branch2_encoder(t2_sar)
        f3 = self.branch3_encoder(t2_optical)

        # 特征投影
        f1 = self.projection1(f1)
        f2 = self.projection2(f2)
        f3 = self.projection3(f3)

        # 应用差异特征注意力
        diff_features = self.diff_attention(f1, f2)
        teacher_guidance = self.diff_attention(f3, f2)

        # 使用 BiFPN 融合特征
        fused_features = self.bifpn([diff_features, teacher_guidance, f3])

        # 解码
        change_map = self.decoder(fused_features[0], teacher_guidance)  # 选择一个输出

        return change_map, f2, f3

    def _build_optical_encoder(self):
        """光学图像特征提取器，使用 ResNet50"""
        encoder = models.resnet50(weights="IMAGENET1K_V2")
        modules = list(encoder.children())[:-2]  # 移除池化和全连接层
        return nn.Sequential(*modules)

    def _build_sar_encoder(self):
        """SAR 图像特征提取器，使用调整后的 ResNet50"""
        encoder = models.resnet50(weights="IMAGENET1K_V2")
        # 修改输入通道为 1（SAR 图像通常为单通道）
        encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        modules = list(encoder.children())[:-2]
        return nn.Sequential(*modules)


class DeepLabV3PlusDecoder(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(DeepLabV3PlusDecoder, self).__init__()
        self.up1 = nn.ConvTranspose2d(in_channels, 256, kernel_size=4, stride=2, padding=1)  # 8x8 -> 16x16
        self.conv1 = nn.Sequential(
            nn.Conv2d(256 + in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # 16x16 -> 32x32
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)   # 32x32 -> 64x64
        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)    # 64x64 -> 128x128
        self.up5 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)    # 128x128 -> 256x256
        self.conv2 = nn.Conv2d(16, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, diff_features, teacher_guidance):
        x = self.up1(diff_features)  # 8x8 -> 16x16
        teacher_guidance_up = F.interpolate(teacher_guidance, size=x.shape[-2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, teacher_guidance_up], dim=1)
        x = self.conv1(x)
        x = self.up2(x)  # 16x16 -> 32x32
        x = self.up3(x)  # 32x32 -> 64x64
        x = self.up4(x)  # 64x64 -> 128x128
        x = self.up5(x)  # 128x128 -> 256x256
        x = self.conv2(x)
        return self.sigmoid(x)




class EnhancedDifferenceAttentionModule(nn.Module):
    def __init__(self, channels):
        super(EnhancedDifferenceAttentionModule, self).__init__()

        # 空间注意力分支
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        # 通道注意力分支
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 16, channels, kernel_size=1),
            nn.Sigmoid()
        )

        # 特征融合模块
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        # 计算差异特征
        diff = torch.abs(x1 - x2)

        # 空间注意力
        avg_out = torch.mean(diff, dim=1, keepdim=True)
        max_out, _ = torch.max(diff, dim=1, keepdim=True)
        spatial_map = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))

        # 通道注意力
        channel_map = self.channel_attention(diff)

        # 应用注意力
        attended_diff = diff * spatial_map * channel_map

        # 融合特征
        return self.fusion(torch.cat([attended_diff, x2], dim=1))


class DynamicWeightModule(nn.Module):
    """动态权重分配模块"""

    def __init__(self, channels):
        super(DynamicWeightModule, self).__init__()

        # 权重生成网络
        self.weight_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 2, kernel_size=1),
            nn.Softmax(dim=1)
        )

    def forward(self, f1, f2):
        # 连接特征
        concat = torch.cat([f1, f2], dim=1)

        # 生成权重
        weights = self.weight_generator(concat)

        # 分离权重
        w1 = weights[:, 0:1, :, :]
        w2 = weights[:, 1:2, :, :]

        # 应用动态权重
        return w1 * f1 + w2 * f2


class ChangeDecoder(nn.Module):
    """变化图解码器"""

    def __init__(self, in_channels):
        super(ChangeDecoder, self).__init__()

        # 解码器网络
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(x)
