# 🎯 YOLOv5滑块验证码识别项目

基于YOLOv5的滑块验证码缺口检测系统，包含数据生成、模型训练、推理和API服务的完整解决方案。

## 📋 项目结构

```
trainSliderCode/
├── generate_slider_dataset.py  # 数据生成脚本
├── train_yolov5.py            # 训练脚本
├── inference.py               # 推理脚本
├── api_server.py              # API服务
├── requirements.txt           # 项目依赖
├── main.py                   # 原始文件
└── README.md                 # 说明文档

生成的文件结构:
├── images/                   # 背景图片文件夹 (用户创建)
├── dataset/                  # 生成的训练数据集
│   ├── images/
│   │   ├── train/           # 训练图片
│   │   └── val/             # 验证图片
│   ├── labels/
│   │   ├── train/           # 训练标注
│   │   └── val/             # 验证标注
│   └── dataset.yaml         # 数据集配置
└── yolov5/                  # YOLOv5仓库 (自动下载)
    └── runs/train/slider_captcha/  # 训练结果
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 创建背景图片文件夹
mkdir images
```

### 2. 准备训练数据

将背景图片放入 `images/` 文件夹中，支持jpg、png、jpeg格式。

```bash
# 生成训练数据集 (默认800训练+200验证)
python generate_slider_dataset.py

# 自定义参数生成
python generate_slider_dataset.py --train-count 1000 --val-count 300 --background-dir images --output-dir dataset
```

### 3. 训练模型

```bash
# 推荐方式：使用ultralytics包训练
python train_ultralytics.py --epochs 100 --batch-size 16

# 备用方式：使用官方脚本训练  
python train_yolov5_official.py --epochs 100 --batch-size 16

# 快速测试训练
python train_ultralytics.py --epochs 10 --batch-size 4
```

训练完成后，最佳模型保存在：`yolov5/runs/train/slider_captcha/weights/best.pt`

### 4. 测试推理

```bash
# 单张图片推理
python inference_simple.py --image test_image.jpg

# 保存可视化结果
python inference_simple.py --image test.jpg --save

# 批量推理 (使用原inference.py)
python inference.py --image-dir test_images/ --output results/
```

### 5. 启动API服务

```bash
# 启动API服务 (默认端口8000)
python api_server.py

# 自定义端口和模型
python api_server.py --port 9000 --model-path your_model.pt
```

## 🔧 API接口说明

### 基础信息
- **服务地址**: `http://localhost:8000`
- **文档地址**: `http://localhost:8000/docs` (Swagger UI)

### 主要接口

#### 1. 单张图片检测
```http
POST /detect
Content-Type: multipart/form-data

参数:
- file: 验证码图片文件

返回:
{
  "success": true,
  "message": "成功检测到滑块缺口", 
  "gap_position": {
    "x": 150,
    "y": 80,
    "width": 60,
    "height": 60,
    "confidence": 0.95,
    "center_x": 180,
    "center_y": 110
  },
  "image_size": {
    "width": 350,
    "height": 200
  }
}
```

#### 2. 批量检测
```http
POST /detect_batch
Content-Type: multipart/form-data

参数:
- files: 多个验证码图片文件
```

#### 3. 健康检查
```http
GET /health

返回:
{
  "status": "healthy",
  "model_loaded": true
}
```

## 📊 参数配置

### 数据生成参数
- `--background-dir`: 背景图片文件夹 (默认: images)
- `--output-dir`: 输出数据集文件夹 (默认: dataset)
- `--train-count`: 训练集数量 (默认: 800)
- `--val-count`: 验证集数量 (默认: 200)

### 训练参数
- `--epochs`: 训练轮数 (默认: 100)
- `--batch-size`: 批次大小 (默认: 16)
- `--img-size`: 输入图片尺寸 (默认: 640)

### 推理参数
- `--confidence`: 置信度阈值 (默认: 0.4)
- `--iou-threshold`: NMS IoU阈值 (默认: 0.5)

## 🔍 使用示例

### Python客户端调用API

```python
import requests

# 单张图片检测
with open('captcha.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/detect', files=files)
    result = response.json()
    
if result['success']:
    gap = result['gap_position']
    print(f"缺口位置: ({gap['center_x']}, {gap['center_y']})")
    print(f"置信度: {gap['confidence']:.2f}")
```

### JavaScript客户端调用

```javascript
// 使用fetch API上传图片
const formData = new FormData();
formData.append('file', imageFile);

fetch('http://localhost:8000/detect', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    if (data.success) {
        console.log('缺口位置:', data.gap_position);
    }
});
```

## 🛠️ 故障排除

### 常见问题

1. **模型加载失败**
   - 检查模型文件路径是否正确
   - 确保训练已完成并生成了 `best.pt` 文件

2. **GPU训练问题**
   - 确保安装了正确版本的PyTorch和CUDA
   - 检查显卡内存是否足够，可以减小batch_size

3. **检测精度不高**
   - 增加训练数据量
   - 调整训练轮数和学习率
   - 检查数据质量和标注准确性

4. **API服务启动失败**
   - 检查端口是否被占用
   - 确认所有依赖已正确安装

## 📈 性能优化建议

1. **数据质量**
   - 使用多样化的背景图片
   - 确保滑块位置分布均匀
   - 适当增加数据增强

2. **训练优化**
   - 使用更大的模型 (yolov5m, yolov5l)
   - 调整超参数 (学习率、权重衰减等)
   - 使用预训练权重加速收敛

3. **部署优化**
   - 使用TensorRT或ONNX优化推理速度
   - 配置负载均衡处理高并发
   - 添加缓存机制减少重复计算

## 📝 开发计划

- [ ] 支持更多验证码类型 (旋转、点选等)
- [ ] 添加模型量化支持
- [ ] 集成Docker部署
- [ ] 添加监控和日志系统
- [ ] 支持模型热更新

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目！

## 📄 许可证

MIT License
