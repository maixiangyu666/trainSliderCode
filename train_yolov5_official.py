#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv5滑块验证码训练脚本
使用官方方式训练，但简化流程
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse
import torch

def train_model(dataset_yaml, epochs=100, batch_size=16, img_size=640, model_size='s'):
    """
    使用虚拟环境中的yolo命令训练模型
    """
    print("🚀 使用虚拟环境中的YOLO训练...")
    
    # 检查数据集配置文件
    dataset_path = Path(dataset_yaml).absolute()
    if not dataset_path.exists():
        print(f"❌ 数据集配置文件不存在: {dataset_path}")
        return False
    
    # 使用yolo命令训练 (虚拟环境中已安装)
    weights = f"yolov5{model_size}.pt"
    device = "0" if torch.cuda.is_available() else "cpu"
    
    train_cmd = [
        "yolo", "train",
        "data=" + str(dataset_path),
        "model=" + weights,
        f"epochs={epochs}",
        f"batch={batch_size}",
        f"imgsz={img_size}",
        f"device={device}",
        "project=runs/train",
        "name=slider_captcha"
    ]
    
    print("🚀 使用虚拟环境YOLO命令训练...")
    print(f"📋 训练命令: {' '.join(train_cmd)}")
    print("=" * 60)
    
    try:
        # 直接在当前目录执行训练
        result = subprocess.run(train_cmd, check=True)
        
        print("✅ 训练完成!")
        print("📁 模型保存位置: runs/train/slider_captcha/weights/")
        print("🏆 最佳模型: runs/train/slider_captcha/weights/best.pt")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ YOLO命令训练失败: {e}")
        print("💡 可能原因: yolo命令不可用，请确保ultralytics包已正确安装")
        return False
    except FileNotFoundError:
        print("❌ yolo命令未找到")
        print("💡 请安装: pip install ultralytics")
        return False
    except KeyboardInterrupt:
        print("⏹️ 训练被用户中断")
        return False

def main():
    parser = argparse.ArgumentParser(description='YOLOv5滑块验证码训练 (官方方式)')
    parser.add_argument('--data', default='dataset/dataset.yaml',
                       help='数据集配置文件路径')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='批次大小')
    parser.add_argument('--img-size', type=int, default=640,
                       help='输入图片尺寸')
    parser.add_argument('--model-size', default='s', choices=['n', 's', 'm', 'l', 'x'],
                       help='模型大小 (n=nano, s=small, m=medium, l=large, x=xlarge)')
    
    args = parser.parse_args()
    
    print("🎯 YOLOv5滑块验证码训练 (官方train.py方式)")
    print("=" * 50)
    
    # 检查数据集文件
    if not Path(args.data).exists():
        print(f"❌ 数据集配置文件不存在: {args.data}")
        print("请先运行 generate_slider_dataset.py 生成数据集")
        return
    
    # 检查环境
    print(f"🔥 PyTorch版本: {torch.__version__}")
    print(f"🖥️ CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🎮 GPU设备: {torch.cuda.get_device_name()}")
    
    # 开始训练
    success = train_model(
        dataset_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        model_size=args.model_size
    )
    
    if success:
        print("\n🎉 训练完成! 接下来可以:")
        print("   1. 测试推理: python inference_simple.py --image test.jpg")
        print("   2. 启动API: python api_server.py")
        print("   3. 查看训练结果: runs/train/slider_captcha/")
    else:
        print("\n❌ 训练失败，请检查上述错误信息")

if __name__ == "__main__":
    main()
