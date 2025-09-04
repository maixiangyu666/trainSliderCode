#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用示例脚本
演示如何使用训练好的模型进行滑块验证码识别
"""

import requests
import json
from pathlib import Path
from PIL import Image, ImageDraw
import io

def test_local_inference():
    """测试本地推理"""
    print("🔍 本地推理测试")
    print("-" * 30)
    
    try:
        from inference import SliderCaptchaDetector
        
        # 初始化检测器
        detector = SliderCaptchaDetector()
        
        # 测试图片路径
        test_images = list(Path("test_images").glob("*.jpg")) if Path("test_images").exists() else []
        
        if not test_images:
            print("⚠️ 没有找到测试图片，请在test_images文件夹中放入验证码图片")
            return
        
        for img_path in test_images[:3]:  # 测试前3张
            print(f"\n📷 测试图片: {img_path.name}")
            result = detector.detect(str(img_path))
            
            if result:
                print(f"✅ 检测成功!")
                print(f"   缺口位置: ({result['center_x']}, {result['center_y']})")
                print(f"   置信度: {result['confidence']:.3f}")
            else:
                print("❌ 检测失败")
                
    except ImportError:
        print("⚠️ 推理模块未找到，请确保inference.py存在")
    except Exception as e:
        print(f"❌ 本地推理测试失败: {e}")

def test_api_service(base_url="http://localhost:8000"):
    """测试API服务"""
    print("🌐 API服务测试")
    print("-" * 30)
    
    # 1. 健康检查
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        health = response.json()
        print(f"✅ 服务状态: {health}")
        
        if not health.get('model_loaded'):
            print("⚠️ 模型未加载，API功能可能不可用")
            return
            
    except requests.exceptions.RequestException:
        print(f"❌ 无法连接到API服务: {base_url}")
        print("请确保API服务已启动: python api_server.py")
        return
    
    # 2. 测试图片检测
    test_images = list(Path("test_images").glob("*.jpg")) if Path("test_images").exists() else []
    
    if not test_images:
        print("⚠️ 没有找到测试图片")
        return
    
    for img_path in test_images[:2]:  # 测试前2张
        print(f"\n📷 API测试图片: {img_path.name}")
        
        try:
            with open(img_path, 'rb') as f:
                files = {'file': f}
                response = requests.post(f"{base_url}/detect", files=files)
                result = response.json()
            
            print(f"📊 API响应: {json.dumps(result, indent=2, ensure_ascii=False)}")
            
            if result.get('success'):
                gap = result['gap_position']
                print(f"✅ API检测成功!")
                print(f"   缺口位置: ({gap['center_x']}, {gap['center_y']})")
                
        except Exception as e:
            print(f"❌ API测试失败: {e}")

def create_sample_captcha():
    """创建示例验证码图片用于测试"""
    print("🎨 创建示例验证码...")
    
    # 创建test_images文件夹
    test_dir = Path("test_images")
    test_dir.mkdir(exist_ok=True)
    
    # 创建简单的示例图片
    img = Image.new('RGB', (350, 200), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    
    # 绘制背景纹理
    for i in range(0, 350, 20):
        for j in range(0, 200, 20):
            color = (200 + (i+j) % 50, 220 + (i*j) % 30, 210 + i % 40)
            draw.rectangle([i, j, i+15, j+15], fill=color)
    
    # 绘制模拟缺口
    gap_x, gap_y = 150, 70
    gap_size = 60
    
    # 缺口区域变暗
    draw.rectangle([gap_x, gap_y, gap_x+gap_size, gap_y+gap_size], 
                  fill=(100, 100, 100), outline=(50, 50, 50), width=2)
    
    # 保存示例图片
    sample_path = test_dir / "sample_captcha.jpg"
    img.save(sample_path)
    print(f"✅ 创建示例图片: {sample_path}")

def main():
    parser = argparse.ArgumentParser(description='使用示例和测试')
    parser.add_argument('--create-sample', action='store_true',
                       help='创建示例验证码图片')
    parser.add_argument('--test-local', action='store_true',
                       help='测试本地推理')
    parser.add_argument('--test-api', action='store_true',
                       help='测试API服务')
    parser.add_argument('--api-url', default='http://localhost:8000',
                       help='API服务地址')
    
    args = parser.parse_args()
    
    print("🎯 滑块验证码识别 - 使用示例")
    print("=" * 50)
    
    if args.create_sample:
        create_sample_captcha()
    
    if args.test_local:
        test_local_inference()
    
    if args.test_api:
        test_api_service(args.api_url)
    
    if not any([args.create_sample, args.test_local, args.test_api]):
        print("💡 使用说明:")
        print("   --create-sample  创建示例验证码图片")
        print("   --test-local     测试本地推理")
        print("   --test-api       测试API服务")
        print("\n📖 完整使用流程请参考 README.md")

if __name__ == "__main__":
    main()
