import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# 导入自定义模块
from model import ImprovedTransformerChangeDetection
from dataset import create_dataloaders


# 损失函数
class EnhancedLoss(nn.Module):
    def __init__(self, alpha=0.5, T=2.0, focal_gamma=2.0, dice_weight=0.5):
        super(EnhancedLoss, self).__init__()
        self.alpha = alpha
        self.T = T
        self.focal_gamma = focal_gamma
        self.dice_weight = dice_weight
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def focal_loss(self, pred, target):
        """Focal Loss - 更好地处理类别不平衡"""
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = target * pred + (1 - target) * (1 - pred)
        focal_weight = (1 - pt) ** self.focal_gamma
        return (focal_weight * bce).mean()

    def dice_loss(self, pred, target, smooth=1.0):
        """Dice Loss - 更好地关注区域重叠"""
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

    def forward(self, outputs, targets, student_features, teacher_features):
        # 组合Focal和Dice损失
        seg_loss = self.focal_loss(outputs, targets) * (1 - self.dice_weight) + \
                   self.dice_loss(outputs, targets) * self.dice_weight

        # 知识蒸馏损失
        bs, c, h, w = student_features.size()
        student_flat = student_features.view(bs, c, -1)
        teacher_flat = teacher_features.view(bs, c, -1)

        student_log_softmax = F.log_softmax(student_flat / self.T, dim=2)
        teacher_softmax = F.softmax(teacher_flat / self.T, dim=2)

        distillation = self.kl_loss(student_log_softmax, teacher_softmax) * (self.T ** 2)

        # 总损失
        loss = (1 - self.alpha) * seg_loss + self.alpha * distillation

        return loss, seg_loss, distillation


# 评估指标
def calculate_metrics(pred, target):
    pred_binary = (pred > 0.5).float()

    # 计算混淆矩阵元素
    tp = torch.sum(pred_binary * target).item()
    fp = torch.sum(pred_binary * (1 - target)).item()
    fn = torch.sum((1 - pred_binary) * target).item()
    tn = torch.sum((1 - pred_binary) * (1 - target)).item()

    # 计算评估指标
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou
    }


def train_one_epoch(model, dataloader, criterion, optimizer, device, grad_clip_value):
    model.train()
    epoch_loss = 0
    epoch_bce_loss = 0
    epoch_distill_loss = 0
    metrics = {'precision': 0, 'recall': 0, 'f1': 0, 'iou': 0}

    pbar = tqdm(dataloader)
    for batch in pbar:
        # 获取数据
        t1_optical = batch['t1_optical'].to(device)
        t2_sar = batch['t2_sar'].to(device)
        t2_optical = batch['t2_optical'].to(device)
        labels = batch['label'].to(device)

        # 前向传播
        outputs, student_features, teacher_features = model(t1_optical, t2_sar, t2_optical)

        # 计算损失
        loss, bce_loss, distill_loss = criterion(outputs, labels, student_features, teacher_features)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)  # 使用传入的参数
        optimizer.step()

        # 记录损失和指标
        epoch_loss += loss.item()
        epoch_bce_loss += bce_loss.item()
        epoch_distill_loss += distill_loss.item()

        # 计算评估指标
        batch_metrics = calculate_metrics(outputs.detach(), labels)
        for k in metrics:
            metrics[k] += batch_metrics[k]

        # 更新进度条
        pbar.set_description(f"Train Loss: {loss.item():.4f}")

    # 计算平均值
    epoch_loss /= len(dataloader)
    epoch_bce_loss /= len(dataloader)
    epoch_distill_loss /= len(dataloader)
    for k in metrics:
        metrics[k] /= len(dataloader)

    return epoch_loss, epoch_bce_loss, epoch_distill_loss, metrics



def validate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    epoch_bce_loss = 0
    epoch_distill_loss = 0
    metrics = {'precision': 0, 'recall': 0, 'f1': 0, 'iou': 0}

    with torch.no_grad():
        pbar = tqdm(dataloader)
        for batch in pbar:
            # 获取数据
            t1_optical = batch['t1_optical'].to(device)
            t2_sar = batch['t2_sar'].to(device)
            t2_optical = batch['t2_optical'].to(device)
            labels = batch['label'].to(device)

            # 前向传播
            outputs, student_features, teacher_features = model(t1_optical, t2_sar, t2_optical)

            # 计算损失
            loss, bce_loss, distill_loss = criterion(outputs, labels, student_features, teacher_features)

            # 记录损失和指标
            epoch_loss += loss.item()
            epoch_bce_loss += bce_loss.item()
            epoch_distill_loss += distill_loss.item()

            # 计算评估指标
            batch_metrics = calculate_metrics(outputs, labels)
            for k in metrics:
                metrics[k] += batch_metrics[k]

            # 更新进度条
            pbar.set_description(f"Val Loss: {loss.item():.4f}")

    # 计算平均值
    epoch_loss /= len(dataloader)
    epoch_bce_loss /= len(dataloader)
    epoch_distill_loss /= len(dataloader)
    for k in metrics:
        metrics[k] /= len(dataloader)

    return epoch_loss, epoch_bce_loss, epoch_distill_loss, metrics


def main(args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建数据加载器 - 使用新函数，支持命令行参数优先
    train_loader, val_loader = create_dataloaders(args)

    # 创建停止训练的参数
    patience = 10
    patience_counter = 0
    best_val_loss = float('inf')

    # 创建模型
    model = ImprovedTransformerChangeDetection(
        hidden_dim=args.hidden_dim,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers
    ).to(device)

    # 定义损失函数和优化器
    criterion = EnhancedLoss(
        alpha=args.alpha,
        T=args.temperature,
        focal_gamma=2.0,
        dice_weight=0.5
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    grad_clip_value = 1.0  # 定义梯度裁剪值

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 训练循环
    best_val_f1 = 0.0
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")

        # 训练
        train_loss, train_bce, train_distill, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, grad_clip_value
        )

        # 验证
        val_loss, val_bce, val_distill, val_metrics = validate(
            model, val_loader, criterion, device
        )

        # 更新学习率
        scheduler.step(val_loss)

        # 打印结果
        print(f"训练损失: {train_loss:.4f}, BCE: {train_bce:.4f}, 蒸馏: {train_distill:.4f}")
        print(
            f"训练指标: 精确率: {train_metrics['precision']:.4f}, 召回率: {train_metrics['recall']:.4f}, F1: {train_metrics['f1']:.4f}, IoU: {train_metrics['iou']:.4f}")
        print(f"验证损失: {val_loss:.4f}, BCE: {val_bce:.4f}, 蒸馏: {val_distill:.4f}")
        print(
            f"验证指标: 精确率: {val_metrics['precision']:.4f}, 召回率: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}, IoU: {val_metrics['iou']:.4f}")

        # 保存最佳模型
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
            print(f"保存最佳模型，F1分数: {best_val_f1:.4f}")

        # 保存最新模型
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_f1': best_val_f1
        }, os.path.join(args.save_dir, 'latest_checkpoint.pth'))

        scheduler.step()

        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
            print(f"保存最佳模型，F1分数: {val_metrics['f1']:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停：验证损失{patience}轮未改善")
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练变化检测模型')
    # 数据集路径 - 现在是可选的，会使用默认值如果未指定
    parser.add_argument('--train_dir', type=str, help='训练数据目录路径')
    parser.add_argument('--val_dir', type=str, help='验证数据目录路径')

    # 其他训练参数
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='保存模型检查点的目录')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮次')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Transformer隐藏维度')
    parser.add_argument('--nhead', type=int, default=8, help='注意力头数量')
    parser.add_argument('--num_encoder_layers', type=int, default=6, help='Transformer编码器层数')
    parser.add_argument('--alpha', type=float, default=0.2, help='蒸馏损失权重')
    parser.add_argument('--temperature', type=float, default=2.0, help='蒸馏温度')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载的工作线程数')

    args = parser.parse_args()

    main(args)
