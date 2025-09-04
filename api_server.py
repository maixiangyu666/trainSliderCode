#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
滑块验证码识别API服务
基于FastAPI提供HTTP接口
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch
from PIL import Image
import io
import numpy as np
import cv2
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="滑块验证码识别API",
    description="基于YOLOv5的滑块验证码缺口检测服务",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量存储模型
model = None

def load_model(model_path="runs/train/slider_captcha/weights/best.pt"):
    """加载训练好的YOLOv5模型"""
    global model
    
    if not Path(model_path).exists():
        logger.error(f"模型文件不存在: {model_path}")
        return False
    
    try:
        # 使用ultralytics YOLO加载自定义模型
        from ultralytics import YOLO
        model = YOLO(model_path)
        
        # 设置默认参数
        model.overrides['conf'] = 0.4  # 置信度阈值
        model.overrides['iou'] = 0.5   # NMS IoU阈值
        
        logger.info(f"✅ 模型加载成功: {model_path}")
        logger.info(f"🖥️ 设备: {model.device}")
        return True
    except Exception as e:
        logger.error(f"❌ 模型加载失败: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """服务启动时加载模型"""
    success = load_model()
    if not success:
        logger.warning("⚠️ 模型未加载，请确保训练完成并检查模型路径")

@app.get("/")
async def root():
    """根路径，返回API信息"""
    return {
        "message": "滑块验证码识别API服务",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/detect")
async def detect_slider_gap(file: UploadFile = File(...)):
    """
    检测滑块验证码中的缺口位置
    
    Args:
        file: 上传的验证码图片文件
        
    Returns:
        JSON: 检测结果，包含缺口的坐标信息
    """
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="模型未加载，请检查模型文件是否存在"
        )
    
    # 验证文件类型
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="请上传图片文件 (JPG, PNG等格式)"
        )
    
    try:
        # 读取上传的图片
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # 使用ultralytics YOLO推理
        results = model(image, conf=0.4, verbose=False)
        
        # 检查检测结果
        if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
            return JSONResponse({
                "success": False,
                "message": "未检测到滑块缺口",
                "gap_position": None
            })
        
        # 获取最佳检测结果
        boxes = results[0].boxes
        best_box = boxes[0]  # 第一个结果通常是置信度最高的
        
        # 提取坐标和置信度
        xyxy = best_box.xyxy[0].cpu().numpy()
        confidence = float(best_box.conf[0].cpu().numpy())
        
        x1, y1, x2, y2 = xyxy
        gap_info = {
            "x": int(x1),
            "y": int(y1),
            "width": int(x2 - x1),
            "height": int(y2 - y1),
            "confidence": confidence,
            "center_x": int((x1 + x2) / 2),
            "center_y": int((y1 + y2) / 2)
        }
        
        return JSONResponse({
            "success": True,
            "message": "成功检测到滑块缺口",
            "gap_position": gap_info,
            "image_size": {
                "width": image.width,
                "height": image.height
            }
        })
        
    except Exception as e:
        logger.error(f"检测过程出错: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"检测过程出错: {str(e)}"
        )

@app.post("/detect_batch")
async def detect_batch(files: list[UploadFile] = File(...)):
    """
    批量检测滑块验证码缺口
    
    Args:
        files: 多个验证码图片文件
        
    Returns:
        JSON: 批量检测结果
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="模型未加载，请检查模型文件是否存在"
        )
    
    results = []
    
    for i, file in enumerate(files):
        try:
            # 读取图片
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            
            # 推理
            detection_results = model(image)
            detections = detection_results.pandas().xyxy[0]
            
            if len(detections) > 0:
                best_detection = detections.iloc[0]
                gap_info = {
                    "x": int(best_detection['xmin']),
                    "y": int(best_detection['ymin']),
                    "width": int(best_detection['xmax'] - best_detection['xmin']),
                    "height": int(best_detection['ymax'] - best_detection['ymin']),
                    "confidence": float(best_detection['confidence']),
                    "center_x": int((best_detection['xmin'] + best_detection['xmax']) / 2),
                    "center_y": int((best_detection['ymin'] + best_detection['ymax']) / 2)
                }
                
                results.append({
                    "file_index": i,
                    "filename": file.filename,
                    "success": True,
                    "gap_position": gap_info
                })
            else:
                results.append({
                    "file_index": i,
                    "filename": file.filename,
                    "success": False,
                    "message": "未检测到滑块缺口"
                })
                
        except Exception as e:
            results.append({
                "file_index": i,
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return JSONResponse({
        "total_files": len(files),
        "results": results
    })

@app.get("/model/info")
async def get_model_info():
    """获取模型信息"""
    if model is None:
        return {"model_loaded": False}
    
    return {
        "model_loaded": True,
        "model_type": "YOLOv5",
        "classes": ["slider_gap"],
        "confidence_threshold": float(model.conf),
        "iou_threshold": float(model.iou)
    }

@app.post("/model/reload")
async def reload_model(model_path: str = "runs/train/slider_captcha/weights/best.pt"):
    """重新加载模型"""
    success = load_model(model_path)
    if success:
        return {"message": "模型重新加载成功", "model_path": model_path}
    else:
        raise HTTPException(status_code=500, detail="模型加载失败")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='滑块验证码识别API服务')
    parser.add_argument('--host', default='0.0.0.0', help='服务器地址')
    parser.add_argument('--port', type=int, default=8000, help='服务器端口')
    parser.add_argument('--model-path', 
                       default='runs/train/slider_captcha/weights/best.pt',
                       help='模型文件路径')
    
    args = parser.parse_args()
    
    print("🚀 启动滑块验证码识别API服务")
    print(f"📡 服务地址: http://{args.host}:{args.port}")
    print(f"🤖 模型路径: {args.model_path}")
    
    # 预加载模型
    load_model(args.model_path)
    
    # 启动服务
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )
