#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目配置文件
集中管理所有配置参数
"""

from pathlib import Path

class Config:
    """项目配置类"""
    
    # 路径配置
    PROJECT_ROOT = Path(__file__).parent
    BACKGROUND_DIR = PROJECT_ROOT / "images"
    DATASET_DIR = PROJECT_ROOT / "dataset"
    YOLO_DIR = PROJECT_ROOT / "yolov5"
    TEST_DIR = PROJECT_ROOT / "test_images"
    
    # 数据生成配置
    SLIDER_SIZES = [(60, 60), (80, 80), (100, 100)]  # 滑块尺寸变化
    IMAGE_SIZE = (350, 200)  # 验证码标准尺寸
    DEFAULT_TRAIN_COUNT = 800
    DEFAULT_VAL_COUNT = 200
    
    # 训练配置
    DEFAULT_EPOCHS = 100
    DEFAULT_BATCH_SIZE = 16
    DEFAULT_IMG_SIZE = 640
    MODEL_CONFIG = "yolov5s.yaml"  # 模型配置
    PRETRAINED_WEIGHTS = "yolov5s.pt"  # 预训练权重
    
    # 推理配置
    CONFIDENCE_THRESHOLD = 0.4
    IOU_THRESHOLD = 0.5
    
    # API配置
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    MODEL_PATH = YOLO_DIR / "runs/train/slider_captcha/weights/best.pt"
    
    # 日志配置
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @classmethod
    def get_model_path(cls, custom_path=None):
        """获取模型路径"""
        if custom_path:
            return Path(custom_path)
        return cls.MODEL_PATH
    
    @classmethod
    def ensure_directories(cls):
        """确保必要目录存在"""
        dirs = [
            cls.BACKGROUND_DIR,
            cls.DATASET_DIR,
            cls.TEST_DIR
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def print_config(cls):
        """打印当前配置"""
        print("⚙️ 当前配置:")
        print(f"   项目根目录: {cls.PROJECT_ROOT}")
        print(f"   背景图片目录: {cls.BACKGROUND_DIR}")
        print(f"   数据集目录: {cls.DATASET_DIR}")
        print(f"   模型路径: {cls.MODEL_PATH}")
        print(f"   API地址: http://{cls.API_HOST}:{cls.API_PORT}")


# 实例化配置
config = Config()

if __name__ == "__main__":
    config.print_config()
