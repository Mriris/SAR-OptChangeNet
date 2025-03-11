import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import glob
from pathlib import Path
import math


def tile_image(img, tile_size, pad_value=0):
    """
    将图像切分为固定大小的小块，不足的地方进行填充

    参数:
        img: PIL图像对象
        tile_size: 小块大小 (width, height)
        pad_value: 填充值

    返回:
        tiles: 切分后的小块列表
        positions: 每个小块在原图中的位置 (x, y)
    """
    width, height = img.size
    tile_width, tile_height = tile_size

    # 计算行列数
    num_cols = math.ceil(width / tile_width)
    num_rows = math.ceil(height / tile_height)

    tiles = []
    positions = []

    # 创建填充后的图像
    padded_width = num_cols * tile_width
    padded_height = num_rows * tile_height

    if img.mode == 'L':
        padded_img = Image.new('L', (padded_width, padded_height), pad_value)
    else:  # RGB或其他模式
        if isinstance(pad_value, int):
            pad_value = (pad_value,) * len(img.getbands())
        padded_img = Image.new(img.mode, (padded_width, padded_height), pad_value)

    # 粘贴原图
    padded_img.paste(img, (0, 0))

    # 切分图像
    for row in range(num_rows):
        for col in range(num_cols):
            x = col * tile_width
            y = row * tile_height

            # 提取小块
            box = (x, y, x + tile_width, y + tile_height)
            tile = padded_img.crop(box)

            tiles.append(tile)
            positions.append((x, y))

    return tiles, positions


def is_acceptable_size_difference(sizes, tolerance=2):
    """
    检查尺寸差异是否在可接受范围内

    参数:
        sizes: 尺寸列表
        tolerance: 允许的像素差异阈值

    返回:
        是否可接受
    """
    max_width = max(w for w, h in sizes)
    min_width = min(w for w, h in sizes)
    max_height = max(h for h, h in sizes)
    min_height = min(h for w, h in sizes)

    width_diff = max_width - min_width
    height_diff = max_height - min_height

    return width_diff <= tolerance and height_diff <= tolerance


def process_image_set(base_name, input_dir, output_dir, tile_size=(256, 256), pad_value=0, size_tolerance=2):
    """
    处理一组相关的图像(A、B、D、E)，切分为小块

    参数:
        base_name: 图像基础名称
        input_dir: 输入目录
        output_dir: 输出目录
        tile_size: 小块大小 (width, height)
        pad_value: 填充值
        size_tolerance: 允许的尺寸差异像素数

    返回:
        成功处理的小块数量
    """
    # 构建文件路径
    path_A = os.path.join(input_dir, f"{base_name}_A.tif")
    path_B = os.path.join(input_dir, f"{base_name}_B.tif")
    path_D = os.path.join(input_dir, f"{base_name}_D.tif")
    path_E = os.path.join(input_dir, f"{base_name}_E.png")

    # 检查文件是否存在
    if not all(os.path.exists(p) for p in [path_A, path_B, path_D, path_E]):
        print(f"警告: 文件集 {base_name} 不完整，跳过")
        return 0

    # 读取图像
    try:
        img_A = Image.open(path_A)
        img_B = Image.open(path_B)
        img_D = Image.open(path_D)
        img_E = Image.open(path_E).convert('L')  # 确保标签是灰度图
    except Exception as e:
        print(f"警告: 打开文件集 {base_name} 时出错: {e}")
        return 0

    # 检查尺寸一致性
    sizes = [img_A.size, img_B.size, img_D.size, img_E.size]

    # 如果尺寸不完全一致，但差异在可接受范围内，则调整尺寸
    if len(set(sizes)) > 1:
        if is_acceptable_size_difference(sizes, size_tolerance):
            # 找出最小的共同尺寸
            min_width = min(w for w, h in sizes)
            min_height = min(h for w, h in sizes)

            # 调整所有图像到相同尺寸
            if img_A.size != (min_width, min_height):
                img_A = img_A.crop((0, 0, min_width, min_height))
            if img_B.size != (min_width, min_height):
                img_B = img_B.crop((0, 0, min_width, min_height))
            if img_D.size != (min_width, min_height):
                img_D = img_D.crop((0, 0, min_width, min_height))
            if img_E.size != (min_width, min_height):
                img_E = img_E.crop((0, 0, min_width, min_height))

            print(f"信息: 文件集 {base_name} 尺寸已调整为 {min_width}x{min_height}")
        else:
            print(f"警告: 文件集 {base_name} 尺寸差异过大 {sizes}，跳过")
            return 0

    # 切分图像为小块
    tiles_A, positions = tile_image(img_A, tile_size)
    tiles_B, _ = tile_image(img_B, tile_size)
    tiles_D, _ = tile_image(img_D, tile_size)
    tiles_E, _ = tile_image(img_E, tile_size, pad_value=0)  # 标签用0填充

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 保存切分后的小块
    for i, ((x, y), tile_A, tile_B, tile_D, tile_E) in enumerate(zip(positions, tiles_A, tiles_B, tiles_D, tiles_E)):
        # 构建新的基础名称，包含原始坐标信息
        new_base_name = f"{base_name}_x{x}_y{y}"

        # 保存路径
        save_path_A = os.path.join(output_dir, f"{new_base_name}_A.tif")
        save_path_B = os.path.join(output_dir, f"{new_base_name}_B.tif")
        save_path_D = os.path.join(output_dir, f"{new_base_name}_D.tif")
        save_path_E = os.path.join(output_dir, f"{new_base_name}_E.png")

        # 保存小块
        tile_A.save(save_path_A)
        tile_B.save(save_path_B)
        tile_D.save(save_path_D)
        tile_E.save(save_path_E)

    return len(tiles_A)


def find_base_names_from_folder(input_dir):
    """
    从文件夹中找出所有基础名称

    参数:
        input_dir: 输入目录

    返回:
        base_names: 基础名称列表
    """
    base_names = set()

    # 查找所有A类型文件
    a_files = glob.glob(os.path.join(input_dir, "*_A.tif"))
    for a_file in a_files:
        # 去掉路径和后缀
        filename = os.path.basename(a_file)
        # 去掉_A.tif部分
        base_name = filename[:-6] if filename.endswith("_A.tif") else filename
        base_names.add(base_name)

    return list(base_names)


def process_dataset(input_dir, output_dir, tile_size=(256, 256), size_tolerance=2):
    """
    处理整个数据集的图像

    参数:
        input_dir: 输入目录
        output_dir: 输出目录
        tile_size: 小块大小 (width, height)
        size_tolerance: 允许的尺寸差异像素数
    """
    # 获取所有基础名称
    base_names = find_base_names_from_folder(input_dir)

    if not base_names:
        print(f"在 {input_dir} 中未找到符合格式的图像")
        return

    print(f"找到 {len(base_names)} 组原始图像")

    # 处理每组图像
    total_tiles = 0
    processed_groups = 0

    for base_name in tqdm(base_names, desc="处理图像"):
        tiles_count = process_image_set(base_name, input_dir, output_dir, tile_size, size_tolerance=size_tolerance)
        if tiles_count > 0:
            total_tiles += tiles_count
            processed_groups += 1

    print(f"成功处理 {processed_groups}/{len(base_names)} 组图像")
    print(f"总共生成 {total_tiles} 个小块")


def main():
    # 设置默认参数
    default_input_dir = r"C:\0Program\Datasets\241120\Compare\Datas\Final"
    default_output_dir = "./data/processed"
    default_tile_size = 256
    default_size_tolerance = 2

    parser = argparse.ArgumentParser(description='将变化检测数据集的图像裁剪为小块')
    parser.add_argument('--input_dir', type=str, help=f'输入目录 (默认: {default_input_dir})')
    parser.add_argument('--output_dir', type=str, help=f'输出目录 (默认: {default_output_dir})')
    parser.add_argument('--tile_size', type=int, default=default_tile_size,
                        help=f'小块大小 (默认: {default_tile_size})')
    parser.add_argument('--size_tolerance', type=int, default=default_size_tolerance,
                        help=f'允许的图像尺寸差异像素数 (默认: {default_size_tolerance})')

    args = parser.parse_args()

    # 优先使用命令行参数，如果没有则使用默认值
    input_dir = args.input_dir if args.input_dir else default_input_dir
    output_dir = args.output_dir if args.output_dir else default_output_dir
    tile_size = (args.tile_size, args.tile_size)
    size_tolerance = args.size_tolerance

    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"小块大小: {tile_size[0]}x{tile_size[1]}")
    print(f"允许的尺寸差异: {size_tolerance}像素")

    # 处理数据集
    process_dataset(input_dir, output_dir, tile_size, size_tolerance)

    print("处理完成！")


if __name__ == "__main__":
    main()
