#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API测试脚本
用于测试滑块验证码识别API服务
"""

import requests
import json
from pathlib import Path
import argparse

def test_health_check(base_url):
    """测试健康检查接口"""
    try:
        response = requests.get(f"{base_url}/health")
        print(f"✅ 健康检查: {response.json()}")
        return True
    except Exception as e:
        print(f"❌ 健康检查失败: {e}")
        return False

def test_model_info(base_url):
    """测试模型信息接口"""
    try:
        response = requests.get(f"{base_url}/model/info")
        info = response.json()
        print(f"🤖 模型信息: {json.dumps(info, indent=2, ensure_ascii=False)}")
        return info.get('model_loaded', False)
    except Exception as e:
        print(f"❌ 获取模型信息失败: {e}")
        return False

def test_detection(base_url, image_path):
    """测试单张图片检测"""
    if not Path(image_path).exists():
        print(f"❌ 测试图片不存在: {image_path}")
        return
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{base_url}/detect", files=files)
            result = response.json()
            
        print(f"🔍 检测结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
        
        if result.get('success'):
            gap = result['gap_position']
            print(f"✅ 检测成功!")
            print(f"   缺口中心: ({gap['center_x']}, {gap['center_y']})")
            print(f"   置信度: {gap['confidence']:.3f}")
        else:
            print(f"⚠️ 检测失败: {result.get('message', '未知错误')}")
            
    except Exception as e:
        print(f"❌ 检测请求失败: {e}")

def main():
    parser = argparse.ArgumentParser(description='API测试脚本')
    parser.add_argument('--url', default='http://localhost:8000',
                       help='API服务地址')
    parser.add_argument('--image', 
                       help='测试图片路径')
    
    args = parser.parse_args()
    
    print("🧪 滑块验证码API测试")
    print("=" * 50)
    
    # 测试健康检查
    if not test_health_check(args.url):
        print("❌ 服务不可用，请检查API服务是否启动")
        return
    
    # 测试模型信息
    model_loaded = test_model_info(args.url)
    if not model_loaded:
        print("⚠️ 模型未加载，检测功能可能不可用")
    
    # 测试图片检测
    if args.image:
        print(f"\n🔍 测试图片检测: {args.image}")
        test_detection(args.url, args.image)
    else:
        print("\n💡 提示: 使用 --image 参数指定测试图片")

if __name__ == "__main__":
    main()
