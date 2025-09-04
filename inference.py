#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
滑块验证码推理脚本
使用训练好的YOLOv5模型识别滑块验证码中的缺口位置
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class SliderCaptchaDetector:
    def __init__(self, model_path, device='cpu'):
        """
        初始化滑块验证码检测器
        
        Args:
            model_path: 训练好的模型文件路径 (.pt)
            device: 推理设备 ('cpu' 或 'cuda')
        """
        self.device = device
        self.model = self.load_model(model_path)
        
    def load_model(self, model_path):
        """加载YOLOv5模型"""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
        print(f"📦 加载模型: {model_path}")
        
        # 使用ultralytics YOLO类加载自定义模型
        from ultralytics import YOLO
        model = YOLO(model_path)
        
        print(f"✅ 模型加载成功，设备: {self.device}")
        return model
    
    def detect(self, image_path, conf_threshold=0.5):
        """
        检测滑块验证码中的缺口位置
        
        Args:
            image_path: 验证码图片路径
            conf_threshold: 置信度阈值
            
        Returns:
            dict: 检测结果 {x, y, width, height, confidence}
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"图片文件不存在: {image_path}")
            
        # 加载图片
        image = Image.open(image_path)
        original_size = image.size
        
        print(f"🔍 检测图片: {image_path}")
        print(f"📏 图片尺寸: {original_size}")
        
        # YOLOv5推理
        results = self.model(image)
        
        # 解析结果
        detections = results.pandas().xyxy[0]  # 获取pandas格式结果
        
        if len(detections) == 0:
            print("❌ 未检测到滑块缺口")
            return None
            
        # 获取置信度最高的检测结果
        best_detection = detections.loc[detections['confidence'].idxmax()]
        
        if best_detection['confidence'] < conf_threshold:
            print(f"⚠️  检测置信度过低: {best_detection['confidence']:.3f}")
            return None
            
        # 提取边界框信息
        result = {
            'x': int(best_detection['xmin']),
            'y': int(best_detection['ymin']),
            'width': int(best_detection['xmax'] - best_detection['xmin']),
            'height': int(best_detection['ymax'] - best_detection['ymin']),
            'confidence': float(best_detection['confidence']),
            'center_x': int((best_detection['xmin'] + best_detection['xmax']) / 2),
            'center_y': int((best_detection['ymin'] + best_detection['ymax']) / 2)
        }
        
        print(f"✅ 检测成功!")
        print(f"📍 缺口位置: ({result['x']}, {result['y']})")
        print(f"📏 缺口尺寸: {result['width']} x {result['height']}")
        print(f"🎯 置信度: {result['confidence']:.3f}")
        print(f"🎯 中心点: ({result['center_x']}, {result['center_y']})")
        
        return result
    
    def visualize_detection(self, image_path, detection_result, save_path=None):
        """
        可视化检测结果
        
        Args:
            image_path: 原图路径
            detection_result: 检测结果
            save_path: 保存路径 (可选)
        """
        if detection_result is None:
            print("❌ 无检测结果可视化")
            return
            
        # 加载图片
        image = Image.open(image_path)
        
        # 创建matplotlib图形
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)
        
        # 绘制边界框
        rect = patches.Rectangle(
            (detection_result['x'], detection_result['y']),
            detection_result['width'], detection_result['height'],
            linewidth=3, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        
        # 绘制中心点
        ax.plot(detection_result['center_x'], detection_result['center_y'], 
               'ro', markersize=8)
        
        # 添加标签
        label = f"滑块缺口\n置信度: {detection_result['confidence']:.3f}\n" \
                f"位置: ({detection_result['x']}, {detection_result['y']})\n" \
                f"中心: ({detection_result['center_x']}, {detection_result['center_y']})"
        
        ax.text(detection_result['x'], detection_result['y'] - 10, label,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
               fontsize=10, color='black')
        
        ax.set_title('滑块验证码缺口检测结果', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 可视化结果保存到: {save_path}")
        
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='滑块验证码推理')
    parser.add_argument('--model', required=True,
                       help='训练好的模型文件路径 (.pt)')
    parser.add_argument('--image', required=True,
                       help='待检测的验证码图片路径')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='置信度阈值 (默认: 0.5)')
    parser.add_argument('--device', default='cpu',
                       help='推理设备 (cpu/cuda)')
    parser.add_argument('--visualize', action='store_true',
                       help='显示可视化结果')
    parser.add_argument('--save-result', 
                       help='保存可视化结果的路径')
    
    args = parser.parse_args()
    
    print("🎯 滑块验证码推理")
    print("=" * 50)
    
    try:
        # 创建检测器
        detector = SliderCaptchaDetector(
            model_path=args.model,
            device=args.device
        )
        
        # 执行检测
        result = detector.detect(
            image_path=args.image,
            conf_threshold=args.conf
        )
        
        # 输出结果
        if result:
            print("\n📊 检测结果:")
            print(f"  缺口位置: ({result['x']}, {result['y']})")
            print(f"  缺口尺寸: {result['width']} x {result['height']}")
            print(f"  中心坐标: ({result['center_x']}, {result['center_y']})")
            print(f"  置信度: {result['confidence']:.3f}")
            
            # 可视化
            if args.visualize or args.save_result:
                detector.visualize_detection(
                    args.image, result, args.save_result
                )
        else:
            print("❌ 检测失败")
            
    except Exception as e:
        print(f"❌ 推理过程出错: {e}")

if __name__ == "__main__":
    main()
