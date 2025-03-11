import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# 导入自定义模块
from model import ImprovedTransformerChangeDetection, BoundaryAwareLoss
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
        pos_weight = torch.ones_like(target) * 15.0  # 正样本权重设为15倍
        weight = target * (pos_weight - 1.0) + 1.0
        bce = F.binary_cross_entropy(pred, target, reduction='none', weight=weight)
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


def train_one_epoch(model, dataloader, criterion, boundary_criterion, optimizer, device, grad_clip_value):
    model.train()
    epoch_loss = 0
    epoch_seg_loss = 0
    epoch_distill_loss = 0
    epoch_boundary_loss = 0  # 边界损失记录
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

        # 计算原始损失
        loss, seg_loss, distill_loss = criterion(outputs, labels, student_features, teacher_features)

        # 计算边界损失
        boundary_loss = boundary_criterion(outputs, labels)

        # 组合总损失
        total_loss = loss + boundary_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
        optimizer.step()

        # 更新损失记录
        epoch_loss += total_loss.item()
        epoch_seg_loss += seg_loss.item()
        epoch_distill_loss += distill_loss.item()
        epoch_boundary_loss += boundary_loss.item()

        # 计算评估指标
        batch_metrics = calculate_metrics(outputs.detach(), labels)
        for k in metrics:
            metrics[k] += batch_metrics[k]

        pbar.set_description(f"Train Loss: {total_loss.item():.4f}")

    # 计算平均值
    epoch_loss /= len(dataloader)
    epoch_seg_loss /= len(dataloader)
    epoch_distill_loss /= len(dataloader)
    epoch_boundary_loss /= len(dataloader)
    for k in metrics:
        metrics[k] /= len(dataloader)

    return epoch_loss, epoch_seg_loss, epoch_distill_loss, epoch_boundary_loss, metrics


def save_visualization(inputs, predictions, targets, save_dir, epoch, batch_idx):
        os.makedirs(save_dir, exist_ok=True)

        t1_optical = inputs['t1_optical'][0].cpu().numpy().transpose(1, 2, 0)
        t2_sar = inputs['t2_sar'][0].cpu().numpy().transpose(1, 2, 0)
        t2_optical = inputs['t2_optical'][0].cpu().numpy().transpose(1, 2, 0)
        pred = predictions[0].cpu().numpy().squeeze()
        target = targets[0].cpu().numpy().squeeze()

        t1_optical = (t1_optical - t1_optical.min()) / (t1_optical.max() - t1_optical.min())
        t2_sar = (t2_sar - t2_sar.min()) / (t2_sar.max() - t2_sar.min())
        t2_optical = (t2_optical - t2_optical.min()) / (t2_optical.max() - t2_optical.min())

        plt.figure(figsize=(15, 10))
        plt.subplot(2, 3, 1);
        plt.imshow(t1_optical);
        plt.title('T1 Optical')
        plt.subplot(2, 3, 2);
        plt.imshow(t2_sar.squeeze(), cmap='gray');
        plt.title('T2 SAR')
        plt.subplot(2, 3, 3);
        plt.imshow(t2_optical);
        plt.title('T2 Optical')
        plt.subplot(2, 3, 4);
        plt.imshow(target, cmap='gray');
        plt.title('Ground Truth')
        plt.subplot(2, 3, 5);
        plt.imshow(pred, cmap='gray');
        plt.title('Prediction')
        plt.subplot(2, 3, 6);
        plt.imshow((pred > 0.5).astype(np.float32), cmap='gray');
        plt.title('Thresholded Prediction')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/epoch_{epoch}_batch_{batch_idx}.png')
        plt.close()


def ensemble_predictions(model_paths, dataloader, device):
    models = []
    for path in model_paths:
        model = ImprovedTransformerChangeDetection(hidden_dim=256, nhead=8, num_encoder_layers=6).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        models.append(model)

    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in dataloader:
            t1_optical = batch['t1_optical'].to(device)
            t2_sar = batch['t2_sar'].to(device)
            t2_optical = batch['t2_optical'].to(device)
            labels = batch['label'].to(device)
            batch_preds = [model(t1_optical, t2_sar, t2_optical)[0] for model in models]
            ensemble_pred = torch.mean(torch.stack(batch_preds), dim=0)
            all_preds.append(ensemble_pred)
            all_targets.append(labels)
    return torch.cat(all_preds), torch.cat(all_targets)

def validate(model, dataloader, criterion, boundary_criterion, device, epoch):
    model.eval()
    epoch_loss = 0
    epoch_seg_loss = 0
    epoch_distill_loss = 0
    epoch_boundary_loss = 0  # 边界损失记录
    metrics = {'precision': 0, 'recall': 0, 'f1': 0, 'iou': 0}

    with torch.no_grad():
        pbar = tqdm(dataloader)
        for batch_idx, batch in enumerate(pbar):
            t1_optical = batch['t1_optical'].to(device)
            t2_sar = batch['t2_sar'].to(device)
            t2_optical = batch['t2_optical'].to(device)
            labels = batch['label'].to(device)

            # 前向传播
            outputs, student_features, teacher_features = model(t1_optical, t2_sar, t2_optical)

            loss, seg_loss, distill_loss = criterion(outputs, labels, student_features, teacher_features)
            boundary_loss = boundary_criterion(outputs, labels)
            total_loss = loss + boundary_loss

            epoch_loss += total_loss.item()
            epoch_seg_loss += seg_loss.item()
            epoch_distill_loss += distill_loss.item()
            epoch_boundary_loss += boundary_loss.item()

            # 计算评估指标
            batch_metrics = calculate_metrics(outputs, labels)
            for k in metrics:
                metrics[k] += batch_metrics[k]

            pbar.set_description(f"Val Loss: {total_loss.item():.4f}")

    # 计算平均值
    epoch_loss /= len(dataloader)
    epoch_seg_loss /= len(dataloader)
    epoch_distill_loss /= len(dataloader)
    epoch_boundary_loss /= len(dataloader)
    for k in metrics:
        metrics[k] /= len(dataloader)

    # 保存可视化
    if epoch % 10 == 0:
        save_visualization(batch, outputs, labels, os.path.join(args.save_dir, 'visualizations'), epoch, batch_idx)

    return epoch_loss, epoch_seg_loss, epoch_distill_loss, epoch_boundary_loss, metrics


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
    best_val_f1 = 0.0

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
        focal_gamma=4.0,
        dice_weight=0.7
    )
    boundary_criterion = BoundaryAwareLoss(weight=0.3).to(device)  # 新增边界损失

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)  # 修改为更稳定的调度器
    grad_clip_value = 1.0

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 训练循环
    best_val_f1 = 0.0
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")

        # 修改调用以传递 boundary_criterion
        train_loss, train_seg, train_distill, train_boundary, train_metrics = train_one_epoch(
            model, train_loader, criterion, boundary_criterion, optimizer, device, grad_clip_value
        )

        val_loss, val_seg, val_distill, val_boundary, val_metrics = validate(
            model, val_loader, criterion, boundary_criterion, device, epoch
        )

        # 更新学习率
        scheduler.step(val_loss)

        # 更新打印信息
        print(f"训练损失: {train_loss:.4f}, Seg: {train_seg:.4f}, 蒸馏: {train_distill:.4f}, Boundary: {train_boundary:.4f}")
        print(f"训练指标: 精确率: {train_metrics['precision']:.4f}, 召回率: {train_metrics['recall']:.4f}, F1: {train_metrics['f1']:.4f}, IoU: {train_metrics['iou']:.4f}")
        print(f"验证损失: {val_loss:.4f}, Seg: {val_seg:.4f}, 蒸馏: {val_distill:.4f}, Boundary: {val_boundary:.4f}")
        print(f"验证指标: 精确率: {val_metrics['precision']:.4f}, 召回率: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}, IoU: {val_metrics['iou']:.4f}")

        # 保存最佳模型
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
            print(f"保存最佳模型，F1分数: {best_val_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停：验证损失或F1分数{patience}轮未改善")
                break

        # 保存最新模型
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_f1': best_val_f1
        }, os.path.join(args.save_dir, 'latest_checkpoint.pth'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练变化检测模型')
    # 数据集路径 - 现在是可选的，会使用默认值如果未指定
    parser.add_argument('--train_dir', type=str, help='训练数据目录路径')
    parser.add_argument('--val_dir', type=str, help='验证数据目录路径')

    # 其他训练参数
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='保存模型检查点的目录')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮次')
    parser.add_argument('--lr', type=float, default=5e-5, help='学习率')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Transformer隐藏维度')
    parser.add_argument('--nhead', type=int, default=8, help='注意力头数量')
    parser.add_argument('--num_encoder_layers', type=int, default=6, help='Transformer编码器层数')
    parser.add_argument('--alpha', type=float, default=0.2, help='蒸馏损失权重')
    parser.add_argument('--temperature', type=float, default=2.0, help='蒸馏温度')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载的工作线程数')

    args = parser.parse_args()

    main(args)
