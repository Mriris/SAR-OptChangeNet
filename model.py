import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torchvision.models as models


class TransformerChangeDetection(nn.Module):
    def __init__(self, hidden_dim=256, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super(TransformerChangeDetection, self).__init__()

        # 特征提取器 - 分支1（时间点1的光学图像）
        self.branch1_encoder = self._build_encoder()

        # 特征提取器 - 分支2（时间点2的SAR图像）- 学生网络
        self.branch2_encoder = self._build_encoder(in_channels=1)  # SAR通常是单通道

        # 特征提取器 - 分支3（时间点2的光学图像）- 教师网络
        self.branch3_encoder = self._build_encoder()

        # Transformer编码器层
        encoder_layers1 = TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead,
                                                  dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder1 = TransformerEncoder(encoder_layers1, num_encoder_layers)

        encoder_layers2 = TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead,
                                                  dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder2 = TransformerEncoder(encoder_layers2, num_encoder_layers)

        encoder_layers3 = TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead,
                                                  dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder3 = TransformerEncoder(encoder_layers3, num_encoder_layers)

        # 特征投影层
        self.projection1 = nn.Conv2d(512, hidden_dim, kernel_size=1)
        self.projection2 = nn.Conv2d(512, hidden_dim, kernel_size=1)
        self.projection3 = nn.Conv2d(512, hidden_dim, kernel_size=1)

        # 差异特征注意力模块
        self.diff_attention = DifferenceAttentionModule(hidden_dim)

        # 解码器 - 用于生成变化图
        self.decoder = ChangeDecoder(hidden_dim)

        # 动态权重分配模块
        self.dynamic_weight = DynamicWeightModule(hidden_dim)

    def _build_encoder(self, in_channels=3):
        """构建基础特征提取器（使用预训练的ResNet作为骨干网络）"""
        encoder = models.resnet18(pretrained=True)

        # 修改第一个卷积层以适应不同的输入通道
        if in_channels != 3:
            encoder.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 移除最后的全连接层
        modules = list(encoder.children())[:-2]
        return nn.Sequential(*modules)

    def _prepare_for_transformer(self, x):
        """将特征图重塑为Transformer期望的格式"""
        batch_size, channels, height, width = x.size()
        # 将特征图重塑为序列形式 [seq_len, batch, features]
        return x.flatten(2).permute(2, 0, 1)

    def forward(self, t1_optical, t2_sar, t2_optical):
        # 分支1 - 时间点1的光学图像
        f1 = self.branch1_encoder(t1_optical)
        f1 = self.projection1(f1)

        # 分支2 - 时间点2的SAR图像（学生网络）
        f2 = self.branch2_encoder(t2_sar)
        f2 = self.projection2(f2)

        # 分支3 - 时间点2的光学图像（教师网络）
        f3 = self.branch3_encoder(t2_optical)
        f3 = self.projection3(f3)

        # 准备特征用于Transformer
        f1_seq = self._prepare_for_transformer(f1)
        f2_seq = self._prepare_for_transformer(f2)
        f3_seq = self._prepare_for_transformer(f3)

        # 通过Transformer编码器
        f1_trans = self.transformer_encoder1(f1_seq)
        f2_trans = self.transformer_encoder2(f2_seq)
        f3_trans = self.transformer_encoder3(f3_seq)

        # 重塑回特征图形式 [batch, channels, height, width]
        batch_size, channels = f1.size(0), f1.size(1)
        height, width = f1.size(2), f1.size(3)

        f1_trans = f1_trans.permute(1, 2, 0).reshape(batch_size, channels, height, width)
        f2_trans = f2_trans.permute(1, 2, 0).reshape(batch_size, channels, height, width)
        f3_trans = f3_trans.permute(1, 2, 0).reshape(batch_size, channels, height, width)

        # 应用差异图注意力机制
        diff_f1_f2 = self.diff_attention(f1_trans, f2_trans)

        # 应用知识蒸馏（教师网络指导学生网络）
        teacher_guidance = self.diff_attention(f3_trans, f2_trans)

        # 动态权重分配
        weighted_features = self.dynamic_weight(diff_f1_f2, teacher_guidance)

        # 解码生成变化图
        change_map = self.decoder(weighted_features)

        return change_map, f2_trans, f3_trans  # 返回变化图和特征用于蒸馏损失计算


class DifferenceAttentionModule(nn.Module):
    """差异图注意力迁移机制"""

    def __init__(self, channels):
        super(DifferenceAttentionModule, self).__init__()
        self.channels = channels

        # 注意力生成网络
        self.attention_generator = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        # 计算差异特征
        diff = torch.abs(x1 - x2)

        # 连接输入特征和差异特征
        concat = torch.cat([diff, x2], dim=1)

        # 生成注意力图
        attention_map = self.attention_generator(concat)

        # 应用注意力
        return x2 * attention_map + diff * (1 - attention_map)


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

