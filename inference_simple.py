#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版滑块验证码推理脚本
使用ultralytics YOLO类
"""

import torch
from ultralytics import YOLO
from pathlib import Path
import argparse
from PIL import Image
import cv2
import numpy as np

class SimpleSliderDetector:
    def __init__(self, model_path="runs/train/slider_captcha/weights/best.pt"):
        """
        初始化检测器
        
        Args:
            model_path: 训练好的模型文件路径
        """
        self.model_path = model_path
        self.model = self.load_model()
        
    def load_model(self):
        """加载训练好的模型"""
        if not Path(self.model_path).exists():
            print(f"❌ 模型文件不存在: {self.model_path}")
            print("请先训练模型: python train_yolov5.py")
            return None
            
        print(f"📦 加载模型: {self.model_path}")
        
        try:
            # 使用ultralytics YOLO加载模型
            model = YOLO(self.model_path)
            print("✅ 模型加载成功")
            return model
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return None
    
    def detect(self, image_path, conf_threshold=0.4):
        """
        检测滑块缺口位置
        
        Args:
            image_path: 图片路径
            conf_threshold: 置信度阈值
            
        Returns:
            dict: 检测结果或None
        """
        if self.model is None:
            print("❌ 模型未加载")
            return None
            
        print(f"🔍 检测图片: {image_path}")
        
        try:
            # 执行推理
            results = self.model(image_path, conf=conf_threshold, verbose=False)
            
            # 检查检测结果
            if len(results) == 0:
                print("❌ 未检测到任何目标")
                return None
            
            result = results[0]
            if result.boxes is None or len(result.boxes) == 0:
                print("❌ 未检测到滑块缺口")
                return None
            
            # 获取最佳检测结果 (置信度最高)
            boxes = result.boxes
            best_idx = torch.argmax(boxes.conf)
            best_box = boxes[best_idx]
            
            # 提取坐标和置信度
            xyxy = best_box.xyxy[0].cpu().numpy()
            confidence = float(best_box.conf[0].cpu().numpy())
            
            x1, y1, x2, y2 = xyxy
            detection_result = {
                'x': int(x1),
                'y': int(y1),
                'width': int(x2 - x1),
                'height': int(y2 - y1),
                'confidence': confidence,
                'center_x': int((x1 + x2) / 2),
                'center_y': int((y1 + y2) / 2)
            }
            
            print(f"✅ 检测成功!")
            print(f"   缺口位置: ({detection_result['center_x']}, {detection_result['center_y']})")
            print(f"   置信度: {detection_result['confidence']:.3f}")
            
            return detection_result
            
        except Exception as e:
            print(f"❌ 检测过程出错: {e}")
            return None
    
    def detect_and_save(self, image_path, output_path=None, conf_threshold=0.4):
        """检测并保存可视化结果"""
        if self.model is None:
            return None
            
        try:
            # 执行推理并保存结果
            results = self.model(image_path, conf=conf_threshold, save=True)
            
            if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                print(f"✅ 检测完成，结果已保存到: runs/detect/predict/")
                return self.detect(image_path, conf_threshold)
            else:
                print("❌ 未检测到目标")
                return None
                
        except Exception as e:
            print(f"❌ 检测保存失败: {e}")
            return None

def main():
    parser = argparse.ArgumentParser(description='滑块验证码推理 (ultralytics)')
    parser.add_argument('--image', required=True, help='输入图片路径')
    parser.add_argument('--model', default='runs/train/slider_captcha/weights/best.pt',
                       help='模型文件路径')
    parser.add_argument('--conf', type=float, default=0.4,
                       help='置信度阈值')
    parser.add_argument('--save', action='store_true',
                       help='保存可视化结果')
    
    args = parser.parse_args()
    
    print("🎯 滑块验证码推理测试")
    print("=" * 50)
    
    # 检查输入图片
    if not Path(args.image).exists():
        print(f"❌ 图片文件不存在: {args.image}")
        return
    
    # 初始化检测器
    detector = SimpleSliderDetector(args.model)
    
    if detector.model is None:
        return
    
    # 执行检测
    if args.save:
        result = detector.detect_and_save(args.image, conf_threshold=args.conf)
    else:
        result = detector.detect(args.image, conf_threshold=args.conf)
    
    if result:
        print(f"\n📊 检测结果:")
        print(f"   缺口坐标: ({result['x']}, {result['y']})")
        print(f"   缺口尺寸: {result['width']} x {result['height']}")
        print(f"   中心位置: ({result['center_x']}, {result['center_y']})")
        print(f"   置信度: {result['confidence']:.3f}")

if __name__ == "__main__":
    main()
