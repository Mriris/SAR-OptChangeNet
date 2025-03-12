# 基于全监督学习的遥感图像异源变化检测

基于全监督学习的遥感图像异源变化检测模型，利用知识蒸馏技术实现SAR图像与光学图像的特征一致性学习。

## 项目概述

本项目实现了一个三分支的变化检测网络框架，用于处理不同时间点的光学与SAR遥感图像，实现高精度的变化检测。模型结构具有以下特点：

1. **三分支结构**:
    - 分支1: 处理时间点1的光学图像
    - 分支2: 处理时间点2的SAR图像（学生网络）
    - 分支3: 处理时间点2的光学图像（教师网络）

2. **知识蒸馏学习**:
    - 利用时间点2的光学图像作为教师网络
    - 指导SAR图像（学生网络）学习更有效的特征表示

3. **创新组件**:
    - 差异图注意力迁移机制：突出显示变化区域
    - 动态权重分配：根据特征重要性动态调整权重
    - 边界感知损失：提高变化边界的准确性
    - BiFPN特征融合：高效融合多尺度特征

## 数据集结构

本项目使用的数据集包含以下格式的图像文件:

- `*_A.tif`: 时间点1的光学图像
- `*_B.tif`: 时间点2的SAR图像
- `*_D.tif`: 时间点2的光学图像
- `*_E.png`: 变化标签（二值图像）

所有图像经过分块裁剪填充处理后尺寸为256×256像素。

## 环境需求

```bash
pip install -r requirements.txt
```

## 项目结构

```
├── model.py          # 模型定义文件
├── dataset.py        # 数据集加载和预处理
├── train.py          # 训练和验证脚本
├── data/             # 数据集目录
│   ├── train/        # 训练数据
│   └── val/          # 验证数据
└── checkpoints/      # 模型保存目录
```

## 模型架构

### 特征提取器

- 光学图像分支: 使用ResNet50作为主干网络
- SAR图像分支: 使用修改后的ResNet50，适配单通道输入，并添加SAR特定处理模块

### 核心模块

1. **差异图注意力迁移机制**
    - 计算两个时相特征的差异
    - 结合空间注意力和通道注意力突出变化区域

2. **BiFPN特征融合**
    - 双向特征金字塔，高效融合多尺度特征
    - 包含自适应权重机制

3. **动态权重分配**
    - 动态计算不同特征的重要性权重
    - 自适应调整特征融合过程

4. **边界感知损失**
    - 通过拉普拉斯算子提取边界信息
    - 增强对变化区域边界的学习

### 损失函数

组合多种损失函数以提高性能:

- Focal Loss: 解决类别不平衡问题
- Dice Loss: 优化区域重叠
- 知识蒸馏损失: 指导学生网络学习教师网络的特征表示
- 边界感知损失: 提高边界检测精度

## 使用方法

### 数据准备

1. 将256×256数据集放置在 `data/train` 和 `data/val` 目录下
2. 确保图像按照 `*_A.tif`, `*_B.tif`, `*_D.tif`, `*_E.png` 的格式命名

### 训练模型

基本训练命令:

```bash
python train.py
```

自定义训练参数:

```bash
python train.py --train_dir ./data/train --val_dir ./data/val --save_dir ./checkpoints --batch_size 32 --epochs 100 --lr 5e-5 --hidden_dim 256 --nhead 8 --num_encoder_layers 6 --alpha 0.2 --temperature 2.0
```

参数说明:

- `--train_dir`: 训练数据目录路径
- `--val_dir`: 验证数据目录路径
- `--save_dir`: 模型保存路径
- `--batch_size`: 批次大小
- `--epochs`: 训练轮次
- `--lr`: 学习率
- `--hidden_dim`: Transformer隐藏维度
- `--nhead`: 注意力头数量
- `--num_encoder_layers`: Transformer编码器层数
- `--alpha`: 蒸馏损失权重
- `--temperature`: 蒸馏温度
- `--num_workers`: 数据加载工作线程数

### 评估指标

| 类别     | 损失     | BCE    | 蒸馏     | 精确率    | 召回率    | F1     | IoU    |
|--------|--------|--------|--------|--------|--------|--------|--------|
| **训练** | 0.3293 | 0.4114 | 0.0006 | 0.2841 | 0.3852 | 0.3180 | 0.1944 |
| **验证** | 0.3560 | 0.4448 | 0.0007 | 0.2607 | 0.3244 | 0.2329 | 0.1402 |

## 创新点总结

1. **三分支架构结合知识蒸馏**:
    - 利用同时期光学图像作为教师，指导SAR图像特征学习
    - 解决了异质数据源间的特征差异问题

2. **差异图注意力迁移机制**:
    - 融合空间和通道注意力，突出变化区域
    - 有效提取变化特征，减少背景干扰

3. **动态权重分配策略**:
    - 根据输入特征自适应调整权重
    - 增强模型对不同类型变化的适应能力

4. **复合损失函数**:
    - 结合Focal Loss、Dice Loss、边界感知损失和知识蒸馏损失
    - 全面优化模型从像素级到语义级的学习

5. **BiFPN特征融合**:
    - 高效融合多尺度特征
    - 提高模型对不同尺度变化的敏感度
