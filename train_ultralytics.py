#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用ultralytics包训练YOLOv5滑块验证码模型
这是最简单直接的方式
"""

import torch
from ultralytics import YOLO
from pathlib import Path
import argparse

def train_model(dataset_yaml, epochs=100, batch_size=16, img_size=640, model_size='s'):
    """
    使用ultralytics YOLO训练模型
    
    Args:
        dataset_yaml: 数据集配置文件路径
        epochs: 训练轮数
        batch_size: 批次大小
        img_size: 输入图片尺寸  
        model_size: 模型大小 (n, s, m, l, x)
    """
    print("🚀 使用ultralytics YOLO训练模型...")
    
    # 检查数据集配置文件
    if not Path(dataset_yaml).exists():
        print(f"❌ 数据集配置文件不存在: {dataset_yaml}")
        return False
    
    try:
        # 加载预训练模型
        model_name = f"yolov5{model_size}.pt"
        print(f"📥 加载预训练模型: {model_name}")
        
        model = YOLO(model_name)  # 自动下载预训练权重
        print("✅ 模型加载成功")
        
        # 检查设备
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🖥️ 使用设备: {device}")
        
        print(f"🎯 开始训练...")
        print(f"   模型: {model_name}")
        print(f"   数据集: {dataset_yaml}")
        print(f"   轮数: {epochs}")
        print(f"   批次大小: {batch_size}")
        print(f"   图片尺寸: {img_size}")
        print("=" * 60)
        
        # 执行训练
        results = model.train(
            data=dataset_yaml,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            device=device,
            project='runs/train',
            name='slider_captcha',
            save_period=10,
            verbose=True
        )
        
        print("\n✅ 训练完成!")
        print("📁 模型保存位置: runs/train/slider_captcha/weights/")
        print("🏆 最佳模型: runs/train/slider_captcha/weights/best.pt")
        print("📈 最后模型: runs/train/slider_captcha/weights/last.pt")
        
        # 显示训练结果
        results_dir = Path("runs/train/slider_captcha")
        if results_dir.exists():
            print(f"\n📊 训练结果文件:")
            result_files = [
                ("results.png", "📈 训练曲线"),
                ("confusion_matrix.png", "🎯 混淆矩阵"),
                ("F1_curve.png", "📊 F1曲线"),
                ("P_curve.png", "📊 精确率曲线"),
                ("R_curve.png", "📊 召回率曲线"),
                ("PR_curve.png", "📊 PR曲线")
            ]
            
            for filename, description in result_files:
                file_path = results_dir / filename
                if file_path.exists():
                    print(f"   {description}: {file_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ 训练过程出错: {e}")
        print("💡 可能的解决方案:")
        print("   1. 检查ultralytics包是否正确安装: pip install ultralytics")
        print("   2. 检查数据集路径和格式")
        print("   3. 尝试减小batch_size")
        print("   4. 确保有足够的内存/显存")
        return False

def main():
    parser = argparse.ArgumentParser(description='YOLOv5滑块验证码训练 (ultralytics包)')
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
    
    print("🎯 YOLOv5滑块验证码训练 (ultralytics包)")
    print("=" * 50)
    
    # 检查数据集文件
    if not Path(args.data).exists():
        print(f"❌ 数据集配置文件不存在: {args.data}")
        print("请先运行 generate_slider_dataset.py 生成数据集")
        return
    
    # 检查ultralytics包
    try:
        from ultralytics import YOLO
        print("✅ ultralytics包可用")
    except ImportError:
        print("❌ ultralytics包未安装")
        print("请安装: pip install ultralytics")
        return
    
    # 检查torch环境
    print(f"🔥 PyTorch版本: {torch.__version__}")
    print(f"🖥️ CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🎮 GPU设备: {torch.cuda.get_device_name()}")
        print(f"💾 GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
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
        print("   3. 查看训练结果图表: runs/train/slider_captcha/results.png")
        
        # 提供测试命令
        print(f"\n🧪 快速测试模型:")
        print(f"   python -c \"from ultralytics import YOLO; model=YOLO('runs/train/slider_captcha/weights/best.pt'); print('模型加载成功!')\"")
    else:
        print("\n❌ 训练失败")
        print("建议:")
        print("   1. 检查数据集是否正确生成")
        print("   2. 尝试减小batch_size (如 --batch-size 8)")
        print("   3. 确保网络连接正常")

if __name__ == "__main__":
    main()
