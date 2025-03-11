import os
import argparse
import random
import shutil
from pathlib import Path
from tqdm import tqdm
import glob


def find_base_names(input_dir):
    """
    从文件夹中查找所有基础名称（不含_A.tif等后缀）

    参数:
        input_dir: 输入目录

    返回:
        base_names: 基础名称列表
    """
    base_names = set()

    # 查找所有A类型文件
    a_files = glob.glob(os.path.join(input_dir, "*_A.tif"))
    for a_file in a_files:
        # 提取基础名称（去掉_A.tif部分）
        filename = os.path.basename(a_file)
        base_name = filename[:-6] if filename.endswith("_A.tif") else filename

        # 检查是否存在完整的文件集
        if (os.path.exists(os.path.join(input_dir, f"{base_name}_B.tif")) and
                os.path.exists(os.path.join(input_dir, f"{base_name}_D.tif")) and
                os.path.exists(os.path.join(input_dir, f"{base_name}_E.png"))):
            base_names.add(base_name)

    return list(base_names)


def split_dataset(input_dir, train_dir, val_dir, val_ratio=0.2, small_batch=False, small_batch_size=100, seed=42):
    """
    将数据集分割为训练集和验证集

    参数:
        input_dir: 输入目录，包含所有裁剪后的数据
        train_dir: 训练集输出目录
        val_dir: 验证集输出目录
        val_ratio: 验证集比例，默认0.2
        small_batch: 是否创建小批次数据集，默认False
        small_batch_size: 小批次大小，默认100
        seed: 随机种子，默认42
    """
    # 设置随机种子以确保可重复性
    random.seed(seed)

    # 查找所有基础名称
    base_names = find_base_names(input_dir)

    if not base_names:
        print(f"错误: 在 {input_dir} 中未找到符合要求的数据文件")
        return

    print(f"找到 {len(base_names)} 组有效数据")

    # 随机打乱数据
    random.shuffle(base_names)

    # 计算验证集大小
    val_size = int(len(base_names) * val_ratio)
    if small_batch and small_batch_size < len(base_names) - val_size:
        train_size = small_batch_size
    else:
        train_size = len(base_names) - val_size

    # 分割数据集
    train_names = base_names[:train_size]
    val_names = base_names[-val_size:] if val_size > 0 else []

    print(f"分割为训练集 {len(train_names)} 组，验证集 {len(val_names)} 组")

    # 创建输出目录
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # 复制训练集文件
    print("复制训练集文件...")
    for base_name in tqdm(train_names):
        copy_files(base_name, input_dir, train_dir)

    # 复制验证集文件
    if val_names:
        print("复制验证集文件...")
        for base_name in tqdm(val_names):
            copy_files(base_name, input_dir, val_dir)

    print("数据集分割完成!")
    print(f"训练集: {train_dir} ({len(train_names)} 组)")
    print(f"验证集: {val_dir} ({len(val_names)} 组)")


def copy_files(base_name, src_dir, dst_dir):
    """
    复制一组相关文件

    参数:
        base_name: 基础文件名
        src_dir: 源目录
        dst_dir: 目标目录
    """
    # 定义文件后缀
    suffixes = ["_A.tif", "_B.tif", "_D.tif", "_E.png"]

    for suffix in suffixes:
        src_file = os.path.join(src_dir, f"{base_name}{suffix}")
        dst_file = os.path.join(dst_dir, f"{base_name}{suffix}")

        # 如果源文件存在，复制到目标目录
        if os.path.exists(src_file):
            shutil.copy2(src_file, dst_file)


def main():
    # 设置默认参数
    default_input_dir = "./data/processed"
    default_train_dir = "./data/train"
    default_val_dir = "./data/val"
    default_val_ratio = 0.2
    default_small_batch = True  # 默认是否创建小批次
    default_small_batch_size = 100  # 默认小批次大小
    default_seed = 42

    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='将裁剪后的数据集分割为训练集和验证集')
    parser.add_argument('--input_dir', type=str, help=f'输入目录 (默认: {default_input_dir})')
    parser.add_argument('--train_dir', type=str, help=f'训练集输出目录 (默认: {default_train_dir})')
    parser.add_argument('--val_dir', type=str, help=f'验证集输出目录 (默认: {default_val_dir})')
    parser.add_argument('--val_ratio', type=float, help=f'验证集比例 (默认: {default_val_ratio})')
    parser.add_argument('--small_batch', action='store_true', help='是否创建小批次数据集')
    parser.add_argument('--no_small_batch', action='store_true', help='不创建小批次数据集')
    parser.add_argument('--small_batch_size', type=int, help=f'小批次大小 (默认: {default_small_batch_size})')
    parser.add_argument('--seed', type=int, help=f'随机种子 (默认: {default_seed})')

    args = parser.parse_args()

    # 优先使用命令行参数，如果没有则使用默认值
    input_dir = args.input_dir if args.input_dir else default_input_dir
    train_dir = args.train_dir if args.train_dir else default_train_dir
    val_dir = args.val_dir if args.val_dir else default_val_dir
    val_ratio = args.val_ratio if args.val_ratio is not None else default_val_ratio

    # 处理小批次参数，命令行参数优先级高于默认值
    if args.small_batch and args.no_small_batch:
        print("警告: 同时指定了--small_batch和--no_small_batch，将使用--small_batch")
        small_batch = True
    elif args.small_batch:
        small_batch = True
    elif args.no_small_batch:
        small_batch = False
    else:
        small_batch = default_small_batch

    small_batch_size = args.small_batch_size if args.small_batch_size is not None else default_small_batch_size
    seed = args.seed if args.seed is not None else default_seed

    print(f"输入目录: {input_dir}")
    print(f"训练集输出目录: {train_dir}")
    print(f"验证集输出目录: {val_dir}")
    print(f"验证集比例: {val_ratio}")
    print(f"是否创建小批次: {'是' if small_batch else '否'}")
    if small_batch:
        print(f"小批次大小: {small_batch_size}")
    print(f"随机种子: {seed}")

    # 执行数据集分割
    split_dataset(
        input_dir=input_dir,
        train_dir=train_dir,
        val_dir=val_dir,
        val_ratio=val_ratio,
        small_batch=small_batch,
        small_batch_size=small_batch_size,
        seed=seed
    )


if __name__ == "__main__":
    main()
