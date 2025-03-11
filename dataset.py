import os
import numpy as np
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class ChangeDetectionDataset(Dataset):
    def __init__(self, root_dir, transform=None, label_transform=None):
        """
        异源变化检测数据集加载类

        参数:
            root_dir (string): 数据集根目录
            transform (callable, optional): 图像变换操作
            label_transform (callable, optional): 标签变换操作
        """
        self.root_dir = root_dir
        self.transform = transform
        self.label_transform = label_transform

        # 获取所有图像的基础名称（不含后缀和分支标识符）
        # 假设图像命名格式如: "185863_127514_A.tif", "185863_127514_B.tif" 等
        all_files = os.listdir(root_dir)
        self.base_names = []

        for file in all_files:
            if file.endswith('_A.tif'):  # 时间点1的光学图像
                base_name = file.rsplit('_', 1)[0]  # 获取基础名称，如 "185863_127514"

                # 确保所有必需文件都存在
                a_file = f"{base_name}_A.tif"
                b_file = f"{base_name}_B.tif"
                d_file = f"{base_name}_D.tif"
                e_file = f"{base_name}_E.png"

                if (a_file in all_files and b_file in all_files and
                        d_file in all_files and e_file in all_files):
                    self.base_names.append(base_name)

    def __len__(self):
        return len(self.base_names)

    def __getitem__(self, idx):
        base_name = self.base_names[idx]

        # 构建文件路径
        t1_optical_path = os.path.join(self.root_dir, f"{base_name}_A.tif")
        t2_sar_path = os.path.join(self.root_dir, f"{base_name}_B.tif")
        t2_optical_path = os.path.join(self.root_dir, f"{base_name}_D.tif")
        label_path = os.path.join(self.root_dir, f"{base_name}_E.png")

        # 读取图像
        t1_optical = Image.open(t1_optical_path).convert('RGB')
        t2_sar = Image.open(t2_sar_path).convert('L')  # SAR图像转为单通道
        t2_optical = Image.open(t2_optical_path).convert('RGB')
        label = Image.open(label_path).convert('L')  # 确保标签是灰度图

        # 应用变换
        if self.transform:
            t1_optical = self.transform(t1_optical)

            # SAR图像特殊处理
            t2_sar = transforms.ToTensor()(t2_sar)
            t2_sar = transforms.Normalize(mean=[0.5], std=[0.5])(t2_sar)

            t2_optical = self.transform(t2_optical)
        else:
            t1_optical = transforms.ToTensor()(t1_optical)
            t2_sar = transforms.ToTensor()(t2_sar)
            t2_optical = transforms.ToTensor()(t2_optical)

        # 标签变换
        if self.label_transform:
            label = self.label_transform(label)
        else:
            label = transforms.ToTensor()(label)
            label = (label > 0.5).float()  # 二值化确保是0或1

        return {
            't1_optical': t1_optical,
            't2_sar': t2_sar,
            't2_optical': t2_optical,
            'label': label
        }


# 定义数据转换
def get_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def binarize_label(x):
    return (x > 0.5).float()


# 定义标签转换
def get_label_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        binarize_label
    ])


def create_dataloaders(args):
    """
    创建训练和验证数据加载器

    参数:
        args: 命令行参数

    返回:
        train_loader, val_loader: 训练和验证数据加载器
    """
    # 默认路径设置
    default_train_dir = "./data/train"
    default_val_dir = "./data/val"

    # 优先使用命令行参数，如果没有则使用默认设置
    train_dir = args.train_dir if hasattr(args, 'train_dir') and args.train_dir else default_train_dir
    val_dir = args.val_dir if hasattr(args, 'val_dir') and args.val_dir else default_val_dir
    batch_size = args.batch_size if hasattr(args, 'batch_size') else 4
    num_workers = args.num_workers if hasattr(args, 'num_workers') else 4

    print(f"使用训练数据目录: {train_dir}")
    print(f"使用验证数据目录: {val_dir}")

    transform = get_transforms()
    label_transform = get_label_transform()

    train_dataset = ChangeDetectionDataset(
        root_dir=train_dir,
        transform=transform,
        label_transform=label_transform
    )

    val_dataset = ChangeDetectionDataset(
        root_dir=val_dir,
        transform=transform,
        label_transform=label_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


# 测试代码
if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='Change Detection Dataset Test')
    parser.add_argument('--train_dir', type=str, help='Path to training data directory')
    parser.add_argument('--val_dir', type=str, help='Path to validation data directory')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')

    args = parser.parse_args()

    # 创建数据加载器
    train_loader, val_loader = create_dataloaders(args)

    # 打印数据信息
    print(f"训练数据集大小: {len(train_loader.dataset)}")
    print(f"验证数据集大小: {len(val_loader.dataset)}")

    # 测试数据加载
    for batch in train_loader:
        print("训练数据批次形状:")
        print(f"t1_optical: {batch['t1_optical'].shape}")  # [B, 3, H, W]
        print(f"t2_sar: {batch['t2_sar'].shape}")  # [B, 1, H, W]
        print(f"t2_optical: {batch['t2_optical'].shape}")  # [B, 3, H, W]
        print(f"label: {batch['label'].shape}")  # [B, 1, H, W]
        break
