#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速开始脚本 - 专为torch.hub环境优化
适用于已安装torch的conda虚拟环境
"""

import torch
import sys
from pathlib import Path

def check_environment():
    """检查环境配置"""
    print("🔍 检查环境配置...")
    
    # 检查Python版本
    print(f"🐍 Python版本: {sys.version}")
    
    # 检查PyTorch
    print(f"🔥 PyTorch版本: {torch.__version__}")
    print(f"🖥️ CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🎮 GPU设备: {torch.cuda.get_device_name()}")
        print(f"💾 GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 检查YOLOv5可用性
    try:
        print("📥 测试YOLOv5下载...")
        test_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        print("✅ YOLOv5可以正常使用")
        del test_model  # 释放内存
    except Exception as e:
        print(f"❌ YOLOv5测试失败: {e}")
        return False
    
    return True

def setup_project():
    """设置项目结构"""
    print("\n📁 设置项目结构...")
    
    # 创建必要目录
    dirs = ["images", "test_images", "results"]
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✅ 创建目录: {dir_name}/")
    
    # 检查背景图片
    images_dir = Path("images")
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    
    if not image_files:
        print("\n⚠️ 重要提醒:")
        print("   请在 images/ 文件夹中放入背景图片 (jpg/png格式)")
        print("   建议至少50张不同风格的图片以获得更好的训练效果")
        return False
    
    print(f"✅ 找到 {len(image_files)} 张背景图片")
    return True

def main():
    print("🎯 YOLOv5滑块验证码识别 - 快速开始")
    print("=" * 60)
    
    # 检查环境
    if not check_environment():
        print("❌ 环境检查失败，请检查torch安装")
        return
    
    # 设置项目
    if not setup_project():
        print("❌ 项目设置失败，请添加背景图片")
        return
    
    print("\n🚀 环境就绪！接下来的步骤:")
    print("\n1️⃣ 生成训练数据:")
    print("   python generate_slider_dataset.py")
    
    print("\n2️⃣ 训练模型:")
    print("   python train_ultralytics.py --epochs 50")
    print("   # 推荐使用ultralytics包，简单快速")
    
    print("\n3️⃣ 测试推理:")
    print("   python inference_simple.py --image test_images/sample.jpg")
    
    print("\n4️⃣ 启动API服务:")
    print("   python api_server.py")
    
    print("\n🔄 一键完成所有步骤:")
    print("   python run_all.py")
    
    print("\n📖 详细说明请查看: README.md")
    
    # 询问是否立即开始
    choice = input("\n是否立即开始生成数据？(y/n): ").lower().strip()
    if choice == 'y':
        print("🚀 开始生成数据...")
        import subprocess
        subprocess.run([sys.executable, "generate_slider_dataset.py"])

if __name__ == "__main__":
    main()
