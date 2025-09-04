#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试训练环境和ultralytics包
"""

def test_ultralytics():
    """测试ultralytics包是否正确安装"""
    print("🧪 测试ultralytics包...")
    
    try:
        from ultralytics import YOLO
        print("✅ ultralytics包导入成功")
        
        # 测试加载预训练模型
        print("📥 测试加载YOLOv5s模型...")
        model = YOLO('yolov5s.pt')
        print("✅ YOLOv5s模型加载成功")
        
        # 检查模型信息
        print(f"📋 模型信息:")
        print(f"   设备: {model.device}")
        print(f"   模型类型: {type(model).__name__}")
        
        # 测试train方法是否存在
        if hasattr(model, 'train'):
            print("✅ train方法存在")
        else:
            print("❌ train方法不存在")
            
        return True
        
    except ImportError as e:
        print(f"❌ ultralytics包导入失败: {e}")
        print("请安装: pip install ultralytics")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_torch_hub():
    """测试torch.hub方式"""
    print("\n🧪 测试torch.hub方式...")
    
    try:
        import torch
        print(f"🔥 PyTorch版本: {torch.__version__}")
        
        # 测试torch.hub加载
        print("📥 测试torch.hub加载YOLOv5...")
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        print("✅ torch.hub加载成功")
        
        # 检查可用方法
        methods = [method for method in dir(model) if not method.startswith('_')]
        print(f"📋 模型可用方法: {', '.join(methods[:10])}...")
        
        # 检查是否有train方法
        if hasattr(model, 'train'):
            print("✅ train方法存在")
        else:
            print("❌ train方法不存在 (这是正常的，torch.hub加载的模型主要用于推理)")
            
        return True
        
    except Exception as e:
        print(f"❌ torch.hub测试失败: {e}")
        return False

def test_dataset():
    """测试数据集是否存在"""
    print("\n🧪 检查数据集...")
    
    dataset_yaml = Path("dataset/dataset.yaml")
    if dataset_yaml.exists():
        print("✅ 数据集配置文件存在")
        
        # 检查数据集目录
        dataset_dir = Path("dataset")
        train_images = list((dataset_dir / "images" / "train").glob("*.jpg"))
        val_images = list((dataset_dir / "images" / "val").glob("*.jpg"))
        train_labels = list((dataset_dir / "labels" / "train").glob("*.txt"))
        val_labels = list((dataset_dir / "labels" / "val").glob("*.txt"))
        
        print(f"📊 训练图片: {len(train_images)} 张")
        print(f"📊 验证图片: {len(val_images)} 张")
        print(f"📝 训练标注: {len(train_labels)} 个")
        print(f"📝 验证标注: {len(val_labels)} 个")
        
        if len(train_images) > 0 and len(train_images) == len(train_labels):
            print("✅ 数据集格式正确")
            return True
        else:
            print("❌ 数据集格式不正确")
            return False
    else:
        print("❌ 数据集配置文件不存在")
        print("请先运行: python generate_slider_dataset.py")
        return False

def main():
    print("🎯 YOLOv5训练环境测试")
    print("=" * 50)
    
    # 测试各个组件
    ultralytics_ok = test_ultralytics()
    torch_hub_ok = test_torch_hub() 
    dataset_ok = test_dataset()
    
    print("\n📋 测试总结:")
    print(f"   ultralytics包: {'✅' if ultralytics_ok else '❌'}")
    print(f"   torch.hub: {'✅' if torch_hub_ok else '❌'}")
    print(f"   数据集: {'✅' if dataset_ok else '❌'}")
    
    if ultralytics_ok and dataset_ok:
        print("\n🚀 环境就绪! 可以开始训练:")
        print("   python train_ultralytics.py")
    elif torch_hub_ok and dataset_ok:
        print("\n🚀 可以使用官方脚本训练:")
        print("   python train_yolov5_official.py")
    else:
        print("\n❌ 环境未就绪，请解决上述问题")
        
        if not dataset_ok:
            print("   首先生成数据集: python generate_slider_dataset.py")
        if not ultralytics_ok:
            print("   安装ultralytics: pip install ultralytics")

if __name__ == "__main__":
    main()
