#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键运行脚本 - 完整流程自动化
从数据生成到模型训练到API部署
"""

import subprocess
import sys
import os
from pathlib import Path
import argparse
import time

def run_command(cmd, description, check=True):
    """运行命令并显示进度"""
    print(f"\n🚀 {description}")
    print(f"💻 执行命令: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    print("-" * 60)
    
    try:
        if isinstance(cmd, str):
            result = subprocess.run(cmd, shell=True, check=check)
        else:
            result = subprocess.run(cmd, check=check)
        print(f"✅ {description} - 完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - 失败: {e}")
        return False

def check_prerequisites():
    """检查前置条件"""
    print("🔍 检查前置条件...")
    
    # 检查Python版本
    if sys.version_info < (3, 7):
        print("❌ 需要Python 3.7+")
        return False
    
    # 检查images文件夹
    if not Path("images").exists():
        print("⚠️ images文件夹不存在，请创建并放入背景图片")
        Path("images").mkdir(exist_ok=True)
        print("📁 已创建images文件夹，请放入背景图片后重新运行")
        return False
    
    # 检查背景图片
    images = list(Path("images").glob("*.jpg")) + \
             list(Path("images").glob("*.png")) + \
             list(Path("images").glob("*.jpeg"))
    
    if not images:
        print("❌ images文件夹中没有图片，请添加背景图片")
        return False
    
    print(f"✅ 找到 {len(images)} 张背景图片")
    return True

def main():
    parser = argparse.ArgumentParser(description='滑块验证码项目一键运行')
    parser.add_argument('--skip-data', action='store_true',
                       help='跳过数据生成 (如果已有数据集)')
    parser.add_argument('--skip-train', action='store_true',
                       help='跳过训练 (如果已有模型)')
    parser.add_argument('--train-count', type=int, default=800,
                       help='训练集数量')
    parser.add_argument('--val-count', type=int, default=200,
                       help='验证集数量')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--api-only', action='store_true',
                       help='只启动API服务')
    
    args = parser.parse_args()
    
    print("🎯 YOLOv5滑块验证码识别 - 一键运行")
    print("=" * 60)
    
    if args.api_only:
        print("🚀 只启动API服务...")
        run_command([sys.executable, "api_server.py"], "启动API服务", check=False)
        return
    
    # 检查前置条件
    if not check_prerequisites():
        return
    
    # 检查torch环境
    try:
        import torch
        print(f"✅ PyTorch已安装: {torch.__version__}")
        print(f"🖥️ CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"🎮 GPU设备: {torch.cuda.get_device_name()}")
        print("🚀 使用现有torch环境，无需额外安装")
    except ImportError:
        print("❌ PyTorch未安装，请先在conda环境中安装torch")
        return
    
    # 2. 生成数据集
    if not args.skip_data:
        if not run_command([
            sys.executable, "generate_slider_dataset.py",
            "--train-count", str(args.train_count),
            "--val-count", str(args.val_count)
        ], "生成训练数据集"):
            return
    else:
        print("⏭️ 跳过数据生成")
    
    # 3. 测试训练环境
    print("🧪 检查训练环境...")
    if not run_command([sys.executable, "test_training.py"], "检查训练环境", check=False):
        print("⚠️ 环境检查有警告，但继续进行...")
    
    # 4. 训练模型
    if not args.skip_train:
        # 优先使用ultralytics方式训练
        print("🚀 尝试使用ultralytics包训练...")
        success = run_command([
            sys.executable, "train_ultralytics.py",
            "--epochs", str(args.epochs),
            "--batch-size", "16" if args.epochs > 50 else "8"  # 长训练用大batch
        ], "使用ultralytics训练YOLOv5模型", check=False)
        
        if not success:
            print("🔄 ultralytics训练失败，尝试官方脚本...")
            success = run_command([
                sys.executable, "train_yolov5_official.py",
                "--epochs", str(args.epochs),
                "--batch-size", "8"  # 官方脚本用较小batch避免内存问题
            ], "使用官方脚本训练YOLOv5模型", check=False)
            
        if not success:
            print("❌ 所有训练方式都失败了")
            print("💡 建议:")
            print("   1. 检查数据集是否正确生成")
            print("   2. 手动运行: python train_ultralytics.py --epochs 10 --batch-size 4")
            return
    else:
        print("⏭️ 跳过模型训练")
    
    # 5. 测试推理 (如果有测试图片)
    test_images = list(Path("test_images").glob("*.jpg")) if Path("test_images").exists() else []
    if test_images:
        print(f"\n🧪 发现测试图片，进行推理测试...")
        run_command([
            sys.executable, "inference_simple.py",
            "--image", str(test_images[0]),
            "--save"
        ], "测试模型推理", check=False)
    else:
        # 创建一个简单测试图片
        print("🎨 创建测试图片...")
        run_command([
            sys.executable, "example_usage.py", "--create-sample"
        ], "创建测试图片", check=False)
    
    # 6. 启动API服务
    print(f"\n🎉 所有步骤完成!")
    print(f"🚀 即将启动API服务...")
    print(f"📡 服务将在 http://localhost:8000 启动")
    print(f"📖 API文档: http://localhost:8000/docs")
    
    # 检查模型是否存在
    model_path = Path("runs/train/slider_captcha/weights/best.pt")
    if model_path.exists():
        print(f"✅ 找到训练好的模型: {model_path}")
        input("\n按Enter键启动API服务...")
        run_command([sys.executable, "api_server.py"], "启动API服务", check=False)
    else:
        print(f"⚠️ 模型文件不存在: {model_path}")
        print("请先完成训练或手动指定模型路径")

if __name__ == "__main__":
    main()
