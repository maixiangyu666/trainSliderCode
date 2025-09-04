#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
滑块验证码数据生成脚本
用于生成YOLOv5训练所需的滑块验证码数据集
"""

import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import json
from pathlib import Path
import argparse

class SliderCaptchaGenerator:
    def __init__(self, background_dir="images", output_dir="dataset"):
        """
        初始化滑块验证码生成器
        
        Args:
            background_dir: 背景图片文件夹
            output_dir: 输出数据集文件夹
        """
        self.background_dir = Path(background_dir)
        self.output_dir = Path(output_dir)
        
        # 创建输出目录结构
        self.setup_directories()
        
        # 滑块参数配置
        self.slider_sizes = [(60, 60), (80, 80), (100, 100)]  # 滑块大小变化
        self.image_size = (350, 200)  # 验证码图片标准尺寸
        
    def setup_directories(self):
        """创建YOLOv5数据集目录结构"""
        dirs = [
            self.output_dir / "images" / "train",
            self.output_dir / "images" / "val", 
            self.output_dir / "labels" / "train",
            self.output_dir / "labels" / "val"
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        print(f"✅ 创建数据集目录结构: {self.output_dir}")

    def create_slider_shape(self, size):
        """
        创建滑块形状 (真实拼图形状)
        
        Args:
            size: (width, height) 滑块尺寸
            
        Returns:
            PIL.Image: 滑块形状mask (白色为滑块区域)
        """
        width, height = size
        mask = Image.new('L', size, 0)  # 黑色背景
        draw = ImageDraw.Draw(mask)
        
        # 基础矩形 (稍微缩小边距)
        margin = 3
        base_rect = [margin, margin, width-margin, height-margin]
        draw.rectangle(base_rect, fill=255)
        
        # 计算凸起/凹陷的尺寸
        bump_size = min(width, height) // 5  # 稍微小一点
        half_bump = bump_size // 2
        
        # 四个方向随机添加凸起或凹陷
        directions = ['top', 'right', 'bottom', 'left']
        random.shuffle(directions)  # 随机顺序
        
        # 每个方向有70%概率添加凸起/凹陷
        for direction in directions[:random.randint(1, 3)]:  # 1-3个方向有凸起/凹陷
            if random.random() < 0.7:  # 70%概率
                is_bump = random.choice([True, False])  # 随机凸起或凹陷
                
                if direction == 'top':
                    # 上方凸起/凹陷
                    center_x = width // 2 + random.randint(-width//6, width//6)
                    center_y = margin
                    ellipse_coords = [
                        center_x - half_bump, center_y - half_bump,
                        center_x + half_bump, center_y + half_bump
                    ]
                    
                elif direction == 'right':
                    # 右侧凸起/凹陷
                    center_x = width - margin
                    center_y = height // 2 + random.randint(-height//6, height//6)
                    ellipse_coords = [
                        center_x - half_bump, center_y - half_bump,
                        center_x + half_bump, center_y + half_bump
                    ]
                    
                elif direction == 'bottom':
                    # 下方凸起/凹陷
                    center_x = width // 2 + random.randint(-width//6, width//6)
                    center_y = height - margin
                    ellipse_coords = [
                        center_x - half_bump, center_y - half_bump,
                        center_x + half_bump, center_y + half_bump
                    ]
                    
                else:  # left
                    # 左侧凸起/凹陷
                    center_x = margin
                    center_y = height // 2 + random.randint(-height//6, height//6)
                    ellipse_coords = [
                        center_x - half_bump, center_y - half_bump,
                        center_x + half_bump, center_y + half_bump
                    ]
                
                # 绘制凸起或凹陷
                if is_bump:
                    # 凸起：在基础矩形外扩展
                    draw.ellipse(ellipse_coords, fill=255)
                else:
                    # 凹陷：在基础矩形内挖空
                    draw.ellipse(ellipse_coords, fill=0)
        
        # 添加一些随机的小凸起/凹陷增加复杂度
        for _ in range(random.randint(0, 2)):  # 0-2个额外的小特征
            small_bump = bump_size // 3
            x = random.randint(margin + small_bump, width - margin - small_bump)
            y = random.randint(margin + small_bump, height - margin - small_bump)
            
            # 确保小特征在边缘附近
            if (x < margin + small_bump * 2 or x > width - margin - small_bump * 2 or
                y < margin + small_bump * 2 or y > height - margin - small_bump * 2):
                
                coords = [x - small_bump//2, y - small_bump//2,
                         x + small_bump//2, y + small_bump//2]
                
                if random.choice([True, False]):
                    draw.ellipse(coords, fill=255)  # 小凸起
                else:
                    draw.ellipse(coords, fill=0)    # 小凹陷
        
        return mask

    def generate_slider_captcha(self, background_image, slider_size):
        """
        生成单个滑块验证码
        
        Args:
            background_image: PIL.Image 背景图片
            slider_size: (width, height) 滑块尺寸
            
        Returns:
            tuple: (captcha_image, slider_x, slider_y, slider_width, slider_height)
        """
        # 调整背景图片到标准尺寸
        bg = background_image.resize(self.image_size, Image.Resampling.LANCZOS)
        
        # 随机选择滑块位置 (确保滑块完全在图片内)
        max_x = self.image_size[0] - slider_size[0] - 10
        max_y = self.image_size[1] - slider_size[1] - 10
        slider_x = random.randint(50, max_x)  # 左边留出空间给滑动轨道
        slider_y = random.randint(10, max_y)
        
        # 创建滑块形状mask
        slider_mask = self.create_slider_shape(slider_size)
        
        # 在背景图上创建缺口
        captcha_img = bg.copy()
        
        # 获取滑块区域
        slider_area = captcha_img.crop((slider_x, slider_y, 
                                       slider_x + slider_size[0], 
                                       slider_y + slider_size[1]))
        
        # 使用mask创建真实的拼图形状缺口
        # 将mask区域变暗并添加阴影效果
        slider_area_array = np.array(slider_area)
        mask_array = np.array(slider_mask)
        
        # 对mask区域应用缺口效果
        for i in range(3):  # RGB三个通道
            channel = slider_area_array[:, :, i]
            # 在mask区域(白色=255)创建缺口效果
            mask_effect = (mask_array == 255)
            channel[mask_effect] = (channel[mask_effect] * 0.2).astype(np.uint8)  # 变暗
            slider_area_array[:, :, i] = channel
        
        # 转换回PIL图像
        processed_area = Image.fromarray(slider_area_array)
        
        # 添加模糊效果
        processed_area = processed_area.filter(ImageFilter.GaussianBlur(0.8))
        
        # 将处理后的区域粘贴回去
        captcha_img.paste(processed_area, (slider_x, slider_y))
        
        # 使用mask绘制精确的缺口边框
        draw = ImageDraw.Draw(captcha_img)
        
        # 创建边框效果 - 沿着mask的边缘
        mask_with_border = slider_mask.filter(ImageFilter.FIND_EDGES)
        
        # 将边框应用到原图
        for y in range(slider_mask.height):
            for x in range(slider_mask.width):
                if mask_with_border.getpixel((x, y)) > 100:  # 边缘像素
                    img_x, img_y = slider_x + x, slider_y + y
                    if 0 <= img_x < captcha_img.width and 0 <= img_y < captcha_img.height:
                        draw.point((img_x, img_y), fill=(80, 80, 80))
        
        return captcha_img, slider_x, slider_y, slider_size[0], slider_size[1]

    def create_yolo_annotation(self, image_width, image_height, x, y, width, height):
        """
        创建YOLOv5格式的标注
        
        Args:
            image_width, image_height: 图片尺寸
            x, y, width, height: 边界框坐标和尺寸
            
        Returns:
            str: YOLO格式标注行
        """
        # 转换为YOLO格式 (归一化的中心点坐标和宽高)
        center_x = (x + width / 2) / image_width
        center_y = (y + height / 2) / image_height
        norm_width = width / image_width
        norm_height = height / image_height
        
        # 类别ID: 0 表示滑块缺口
        return f"0 {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}"

    def generate_dataset(self, num_train=800, num_val=200):
        """
        生成完整数据集
        
        Args:
            num_train: 训练集数量
            num_val: 验证集数量
        """
        if not self.background_dir.exists():
            print(f"❌ 背景图片目录不存在: {self.background_dir}")
            print("请创建 'images' 文件夹并放入背景图片")
            return
            
        # 获取背景图片列表
        bg_images = list(self.background_dir.glob("*.jpg")) + \
                   list(self.background_dir.glob("*.png")) + \
                   list(self.background_dir.glob("*.jpeg"))
        
        if not bg_images:
            print(f"❌ 在 {self.background_dir} 中没有找到图片文件")
            return
            
        print(f"📁 找到 {len(bg_images)} 张背景图片")
        
        # 生成训练集
        print("🚀 开始生成训练集...")
        self._generate_split(bg_images, "train", num_train)
        
        # 生成验证集
        print("🚀 开始生成验证集...")
        self._generate_split(bg_images, "val", num_val)
        
        # 创建dataset.yaml配置文件
        self.create_dataset_yaml()
        
        print("✅ 数据集生成完成!")
        print(f"📊 训练集: {num_train} 张")
        print(f"📊 验证集: {num_val} 张")
        print(f"📁 输出目录: {self.output_dir}")

    def _generate_split(self, bg_images, split, count):
        """生成训练集或验证集"""
        images_dir = self.output_dir / "images" / split
        labels_dir = self.output_dir / "labels" / split
        
        for i in range(count):
            # 随机选择背景图片和滑块尺寸
            bg_path = random.choice(bg_images)
            slider_size = random.choice(self.slider_sizes)
            
            try:
                # 加载背景图片
                bg_img = Image.open(bg_path).convert('RGB')
                
                # 生成滑块验证码
                captcha_img, x, y, w, h = self.generate_slider_captcha(bg_img, slider_size)
                
                # 保存图片
                img_filename = f"{split}_{i:06d}.jpg"
                img_path = images_dir / img_filename
                captcha_img.save(img_path, quality=95)
                
                # 创建标注
                annotation = self.create_yolo_annotation(
                    self.image_size[0], self.image_size[1], x, y, w, h
                )
                
                # 保存标注文件
                label_filename = f"{split}_{i:06d}.txt"
                label_path = labels_dir / label_filename
                with open(label_path, 'w') as f:
                    f.write(annotation)
                    
                if (i + 1) % 100 == 0:
                    print(f"  ✓ {split} 已生成 {i + 1}/{count}")
                    
            except Exception as e:
                print(f"⚠️  生成第 {i} 张图片时出错: {e}")
                continue

    def create_dataset_yaml(self):
        """创建YOLOv5数据集配置文件 - 符合官方文档格式"""
        # 使用相对路径，符合官方文档要求
        yaml_content = f"""# 滑块验证码数据集配置
# 按照 https://docs.ultralytics.com/zh/yolov5/tutorials/train_custom_data/ 格式

path: {self.output_dir.absolute().as_posix()}  # 数据集根目录 (绝对路径)
train: images/train  # 训练集图片路径 (相对于path)
val: images/val      # 验证集图片路径 (相对于path)

# 类别数量和名称
nc: 1  # 类别数量
names: ['slider_gap']  # 类别名称: 0-slider_gap

# 数据集信息
download: |
  # 滑块验证码数据集
  # 用于训练YOLOv5识别滑块验证码中的缺口位置
  # 数据格式: YOLO格式 (归一化坐标)
  # 标注格式: class_id center_x center_y width height
"""
        
        yaml_path = self.output_dir / "dataset.yaml"
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
            
        print(f"✅ 创建配置文件: {yaml_path}")
        print("📋 数据集配置符合YOLOv5官方格式")

    def preview_slider_shapes(self, count=5):
        """预览生成的滑块形状"""
        print("🎨 生成滑块形状预览...")
        
        preview_dir = Path("preview_shapes")
        preview_dir.mkdir(exist_ok=True)
        
        for i in range(count):
            # 随机选择尺寸
            size = random.choice(self.slider_sizes)
            
            # 生成滑块形状
            mask = self.create_slider_shape(size)
            
            # 创建预览图 - 白色背景上显示滑块形状
            preview = Image.new('RGB', size, (240, 240, 240))
            
            # 将mask应用到预览图
            mask_array = np.array(mask)
            preview_array = np.array(preview)
            
            # 滑块区域显示为深色
            slider_pixels = (mask_array == 255)
            preview_array[slider_pixels] = [100, 150, 200]  # 蓝色滑块
            
            # 边框
            border_mask = mask.filter(ImageFilter.FIND_EDGES)
            border_array = np.array(border_mask)
            border_pixels = (border_array > 100)
            preview_array[border_pixels] = [50, 50, 50]  # 深色边框
            
            # 保存预览
            preview_result = Image.fromarray(preview_array)
            preview_path = preview_dir / f"slider_shape_{i+1}_{size[0]}x{size[1]}.png"
            preview_result.save(preview_path)
            
        print(f"✅ 滑块形状预览保存到: {preview_dir}/")
        print("💡 查看生成的滑块形状是否符合预期")


def main():
    parser = argparse.ArgumentParser(description='滑块验证码数据集生成器')
    parser.add_argument('--background-dir', default='images', 
                       help='背景图片文件夹 (默认: images)')
    parser.add_argument('--output-dir', default='dataset', 
                       help='输出数据集文件夹 (默认: dataset)')
    parser.add_argument('--train-count', type=int, default=800,
                       help='训练集数量 (默认: 800)')
    parser.add_argument('--val-count', type=int, default=200,
                       help='验证集数量 (默认: 200)')
    parser.add_argument('--preview-only', action='store_true',
                       help='只生成滑块形状预览，不生成数据集')
    
    args = parser.parse_args()
    
    print("🎯 滑块验证码数据集生成器")
    print("=" * 50)
    
    generator = SliderCaptchaGenerator(
        background_dir=args.background_dir,
        output_dir=args.output_dir
    )
    
    if args.preview_only:
        # 只生成预览
        generator.preview_slider_shapes(count=10)
        print("🎨 预览生成完成！请查看 preview_shapes/ 文件夹")
        print("💡 如果形状满意，可以运行完整数据生成:")
        print(f"   python {__file__} --train-count {args.train_count} --val-count {args.val_count}")
    else:
        # 先生成预览
        print("🎨 生成滑块形状预览...")
        generator.preview_slider_shapes(count=5)
        
        # 生成完整数据集
        generator.generate_dataset(
            num_train=args.train_count,
            num_val=args.val_count
        )


if __name__ == "__main__":
    main()
