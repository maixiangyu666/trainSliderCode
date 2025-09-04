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
        创建滑块形状 (拼图形状)
        
        Args:
            size: (width, height) 滑块尺寸
            
        Returns:
            PIL.Image: 滑块形状mask (白色为滑块区域)
        """
        width, height = size
        mask = Image.new('L', size, 0)  # 黑色背景
        draw = ImageDraw.Draw(mask)
        
        # 基础矩形
        base_rect = [5, 5, width-5, height-5]
        draw.rectangle(base_rect, fill=255)
        
        # 添加拼图凸起/凹陷
        bump_size = min(width, height) // 4
        
        # 右侧凸起
        if random.choice([True, False]):
            bump_x = width - 5
            bump_y = height // 2 - bump_size // 2
            draw.ellipse([bump_x-bump_size//2, bump_y, 
                         bump_x+bump_size//2, bump_y+bump_size], fill=255)
        
        # 上方凸起/凹陷
        if random.choice([True, False]):
            bump_x = width // 2 - bump_size // 2
            bump_y = 5
            if random.choice([True, False]):  # 凸起
                draw.ellipse([bump_x, bump_y-bump_size//2,
                             bump_x+bump_size, bump_y+bump_size//2], fill=255)
            else:  # 凹陷
                draw.ellipse([bump_x, bump_y-bump_size//2,
                             bump_x+bump_size, bump_y+bump_size//2], fill=0)
        
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
        
        # 创建滑块形状
        slider_mask = self.create_slider_shape(slider_size)
        
        # 在背景图上创建缺口
        captcha_img = bg.copy()
        
        # 将滑块区域变暗或模糊来模拟缺口
        slider_area = captcha_img.crop((slider_x, slider_y, 
                                       slider_x + slider_size[0], 
                                       slider_y + slider_size[1]))
        
        # 创建缺口效果 - 降低亮度和增加边框
        slider_area = slider_area.point(lambda p: p * 0.3)  # 变暗
        slider_area = slider_area.filter(ImageFilter.GaussianBlur(1))  # 轻微模糊
        
        # 将处理后的区域粘贴回去
        captcha_img.paste(slider_area, (slider_x, slider_y))
        
        # 绘制缺口边框
        draw = ImageDraw.Draw(captcha_img)
        draw.rectangle([slider_x-1, slider_y-1, 
                       slider_x + slider_size[0], slider_y + slider_size[1]], 
                      outline=(100, 100, 100), width=2)
        
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
    
    args = parser.parse_args()
    
    print("🎯 滑块验证码数据集生成器")
    print("=" * 50)
    
    generator = SliderCaptchaGenerator(
        background_dir=args.background_dir,
        output_dir=args.output_dir
    )
    
    generator.generate_dataset(
        num_train=args.train_count,
        num_val=args.val_count
    )


if __name__ == "__main__":
    main()
