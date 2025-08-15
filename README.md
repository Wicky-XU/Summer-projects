# COVID-19 肺部CT图像分类项目

*基于VGG16深度学习的COVID-19肺部CT图像智能分类系统*

</div>

## 📋 项目简介

本项目使用深度学习技术对肺部CT图像进行COVID-19诊断分类，基于预训练的VGG16模型构建，支持10轮和50轮两种训练配置，提供完整的数据处理、模型训练、预测分析和结果可视化功能。

### 🎯 主要特性

- **双版本训练**：支持10轮快速训练和50轮完整训练
- **智能预处理**：自动数据分割、增强和验证
- **高级架构**：基于VGG16的增强分类网络
- **全面监控**：实时训练监控和性能分析
- **预测分析**：详细的置信度和不确定性评估
- **可视化**：丰富的图表和统计分析
- **模块化设计**：清晰的代码结构便于维护

## 📁 项目结构

```
lung-cnn/
├── 📄 README.md                    # 项目文档（本文件）
├── 📄 requirements.txt             # Python依赖包列表
├── 📄 .gitignore                  # Git忽略文件配置
│
├── 📂 src/                        # 模块化源代码
│   ├── 📄 __init__.py             # Python包初始化
│   ├── 📄 config.py               # 配置管理模块
│   ├── 📄 data_utils.py           # 数据处理工具
│   ├── 📄 model_utils.py          # 模型构建和训练
│   ├── 📄 prediction_utils.py     # 预测和评估
│   ├── 📄 visualization.py        # 可视化工具
│   └── 📄 main.py                # 主程序入口
│
├── 📂 notebooks/                  # Jupyter笔记本
│   ├── 📓 train_10_epochs.ipynb  # 10轮训练版本
│   ├── 📓 train_50_epochs.ipynb  # 50轮训练版本
│   └── 📓 project_template.ipynb # 项目模板
│
├── 📂 data/                       # 数据文件夹
│   ├── 📂 train_covid19/         # 训练数据
│   │   ├── 📂 yes/               # COVID-19阳性CT图像
│   │   └── 📂 no/                # COVID-19阴性CT图像
│   ├── 📂 test_healthcare/       # 测试数据
│   │   └── 📂 test/              # 待预测CT图像
│   └── 📂 processed/             # 处理后数据
│       ├── 📂 Train_covid/       # 训练集
│       └── 📂 Val_covid/         # 验证集
│
├── 📂 models/                     # 训练好的模型
│   ├── 🤖 covid_classifier_vgg16_10epochs.h5     # 10轮训练模型
│   ├── 🤖 covid_classifier_vgg16_50epochs.h5     # 50轮训练模型
│   ├── 🤖 best_model_10epochs.h5                 # 10轮最佳检查点
│   ├── 🤖 best_model_50epochs.h5                 # 50轮最佳检查点
│   ├── 📊 training_history_10epochs.pkl          # 10轮训练历史
│   └── 📊 training_history_50epochs_enhanced.pkl # 50轮训练历史
│
├── 📂 results/                    # 结果文件夹
│   ├── 📂 plots/                 # 训练和预测图表
│   │   ├── 📈 training_history_10epochs.png     # 10轮训练曲线
│   │   ├── 📈 training_history_50epochs.png     # 50轮训练曲线
│   │   ├── 📊 prediction_results_10epochs.png   # 10轮预测结果
│   │   ├── 📊 prediction_results_50epochs.png   # 50轮预测结果
│   │   └── 📊 model_comparison.png               # 模型对比图
│   ├── 📂 predictions/           # 预测结果
│   │   ├── 📄 predictions_10epochs.json         # 10轮预测结果
│   │   └── 📄 detailed_predictions_50epochs.json # 50轮详细预测
│   └── 📂 logs/                  # 训练日志
│       ├── 📄 training_10epochs.log             # 10轮训练日志
│       └── 📄 training_50epochs.log             # 50轮训练日志
│
└── 📂 docs/                      # 文档文件夹
    ├── 📄 python_tutorial.md     # Python编程教学文档
    ├── 📄 model_summary_10epochs.txt  # 10轮模型摘要
    ├── 📄 model_summary_50epochs.txt  # 50轮模型摘要
    └── 📄 api_documentation.md   # API文档（可选）
```

## 🚀 快速开始

### 环境要求

- **Python**: 3.7+ (推荐3.8或3.9)
- **操作系统**: Windows 10+, macOS 10.14+, Ubuntu 18.04+
- **内存**: 最少8GB RAM (推荐16GB+)
- **GPU**: 可选，但强烈推荐 (6GB+ 显存)
- **存储**: 至少10GB可用空间

### 1. 环境安装

```bash
# 克隆项目（如果从Git获取）
git clone <repository-url>
cd lung-cnn

# 创建虚拟环境（推荐）
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

将您的数据按以下结构组织：

```
data/
├── train_covid19/
│   ├── yes/          # COVID-19阳性CT图像
│   └── no/           # COVID-19阴性CT图像
└── test_healthcare/
    └── test/         # 待预测CT图像
```

### 3. 运行方式

#### 方式A: 使用Jupyter笔记本（推荐初学者）

```bash
# 启动Jupyter Notebook
jupyter notebook

# 在浏览器中打开以下文件之一：
# - notebooks/train_10_epochs.ipynb  (快速训练)
# - notebooks/train_50_epochs.ipynb  (完整训练)
```

#### 方式B: 使用模块化代码（推荐生产环境）

```bash
# 快速10轮训练
python src/main.py --mode train --version 10_epochs

# 完整50轮训练
python src/main.py --mode train --version 50_epochs

# 进行预测
python src/main.py --mode predict --version 10_epochs

# 比较模型性能
python src/main.py --mode compare

# 运行完整流程（训练+预测+比较）
python src/main.py --all
```

## 📚 文件功能详解

### 🔧 核心模块 (`src/`)

| 文件 | 功能描述 | 主要类/函数 |
|------|----------|-------------|
| `config.py` | 统一配置管理 | `Config` - 管理所有参数和路径 |
| `data_utils.py` | 数据处理工具 | `DataProcessor` - 数据加载、预处理、分割 |
| `model_utils.py` | 模型相关功能 | `ModelBuilder`, `ModelTrainer` - 构建和训练模型 |
| `prediction_utils.py` | 预测和评估 | `Predictor` - 模型预测和结果分析 |
| `visualization.py` | 可视化工具 | `TrainingVisualizer` - 绘制训练曲线和结果 |
| `main.py` | 主程序入口 | 整合所有功能的命令行界面 |

### 📓 Jupyter笔记本 (`notebooks/`)

#### `train_10_epochs.ipynb` - 快速训练版本
- **用途**: 快速验证和原型开发
- **训练轮数**: 10轮
- **特点**: 
  - 基础数据增强
  - 标准VGG16架构
  - 快速训练（约30-60分钟）
  - 适合初次尝试和概念验证

#### `train_50_epochs.ipynb` - 完整训练版本
- **用途**: 生产级模型训练
- **训练轮数**: 50轮
- **特点**:
  - 增强的数据增强策略
  - 改进的模型架构（4层分类器）
  - 高级回调和监控
  - 不确定性分析
  - 全面的性能评估
  - 适合实际部署

#### `project_template.ipynb` - 项目模板
- **用途**: 项目结构参考和快速上手
- **内容**: 基础流程模板和示例

### 🤖 模型文件 (`models/`)

| 文件类型 | 命名规则 | 说明 |
|----------|----------|------|
| 最终模型 | `covid_classifier_vgg16_{version}.h5` | 完整训练后的最终模型 |
| 最佳检查点 | `best_model_{version}.h5` | 验证集上表现最佳的模型 |
| 训练历史 | `training_history_{version}.pkl` | 训练过程的所有指标数据 |
| 模型摘要 | `model_summary_{version}.txt` | 模型架构和性能总结 |

### 📊 结果文件 (`results/`)

#### 图表文件 (`plots/`)
- **训练曲线**: 准确率、损失、精确度、召回率变化
- **预测分析**: 置信度分布、类别统计、不确定性分析
- **模型对比**: 不同版本性能对比图表

#### 预测结果 (`predictions/`)
- **JSON格式**: 详细的预测结果，包含文件名、置信度、不确定性
- **统计摘要**: 类别分布、性能指标、置信度统计

### 📄 配置和文档

| 文件 | 用途 |
|------|------|
| `requirements.txt` | Python依赖包列表和版本要求 |
| `.gitignore` | Git版本控制忽略文件配置 |
| `README.md` | 项目文档（本文件） |
| `docs/python_tutorial.md` | Python编程教学文档 |

## ⚙️ 配置参数

### 主要参数配置 (`src/config.py`)

```python
# 图像参数
IMG_SIZE = (224, 224)          # VGG16要求的输入尺寸
BATCH_SIZE = 32                # 批处理大小
LEARNING_RATE = 1e-3           # 学习率

# 训练参数
NUM_EPOCHS_10 = 10             # 10轮版本
NUM_EPOCHS_50 = 50             # 50轮版本
TRAIN_VAL_SPLIT = 0.6          # 训练/验证分割比例

# 数据增强参数
ROTATION_RANGE = 20            # 旋转角度范围
WIDTH_SHIFT_RANGE = 0.2        # 宽度偏移范围
HEIGHT_SHIFT_RANGE = 0.2       # 高度偏移范围
ZOOM_RANGE = 0.2               # 缩放范围
```

### 版本差异对比

| 特性 | 10轮版本 | 50轮版本 |
|------|----------|----------|
| 训练时间 | 30-60分钟 | 2-4小时 |
| 数据增强 | 基础增强 | 增强版增强 |
| 模型架构 | 标准分类器 | 4层深度分类器 |
| 早停耐心 | 5轮 | 10轮 |
| 学习率调度 | 无 | 自适应调整 |
| 监控指标 | 准确率 | 准确率+精确度+召回率 |
| 不确定性分析 | 基础 | 详细分析 |
| 适用场景 | 快速验证 | 生产部署 |

## 📈 性能指标

### 评估指标

- **准确率 (Accuracy)**: 正确分类的样本比例
- **精确度 (Precision)**: 预测为阳性中真正为阳性的比例
- **召回率 (Recall)**: 实际阳性中被正确预测的比例
- **F1分数**: 精确度和召回率的调和平均
- **AUC-ROC**: 受试者工作特征曲线下面积
- **置信度**: 模型对预测结果的确信程度
- **不确定性**: 预测的不确定程度

### 预期性能

| 版本 | 训练准确率 | 验证准确率 | 推荐用途 |
|------|------------|------------|----------|
| 10轮 | 85-95% | 80-90% | 概念验证、快速测试 |
| 50轮 | 90-98% | 85-95% | 生产部署、临床研究 |

## 🔧 高级功能

### 1. 模型集成

```python
# 加载多个模型进行集成预测
from src.prediction_utils import EnsemblePredictor

models = ['models/best_model_10epochs.h5', 'models/best_model_50epochs.h5']
ensemble = EnsemblePredictor(models)
predictions = ensemble.predict(test_images)
```

### 2. 梯度类激活映射 (Grad-CAM)

```python
# 生成热力图解释模型决策
from src.visualization import GradCAMVisualizer

visualizer = GradCAMVisualizer(model)
heatmap = visualizer.generate_heatmap(image, class_index=1)
```

### 3. 超参数优化

```python
# 使用网格搜索优化超参数
from src.model_utils import HyperparameterOptimizer

optimizer = HyperparameterOptimizer()
best_params = optimizer.grid_search(param_grid)
```

## 🚨 重要说明

### ⚠️ 医学AI使用声明

1. **研究目的**: 本项目仅用于研究和教育目的
2. **临床验证**: 医学应用前需要严格的临床验证
3. **专业咨询**: 诊断决策应咨询专业医生
4. **数据隐私**: 确保患者数据的隐私和安全
5. **监管合规**: 遵守当地医疗AI相关法规

### 🔒 数据安全

- 使用前请确保数据已脱敏
- 遵守HIPAA等医疗数据保护法规
- 建议在安全的隔离环境中运行
- 定期备份重要结果和模型

## 🛠️ 故障排除

### 常见问题

1. **GPU内存不足**
   ```python
   # 在config.py中减少批处理大小
   BATCH_SIZE = 16  # 或更小
   ```

2. **依赖包冲突**
   ```bash
   # 使用虚拟环境
   pip install --upgrade pip
   pip install -r requirements.txt --force-reinstall
   ```

3. **数据加载错误**
   - 检查数据路径配置
   - 确认图像格式支持 (jpg, png, bmp, tiff)
   - 验证文件夹结构

4. **训练中断恢复**
   ```python
   # 加载最佳检查点继续训练
   model = load_model('models/best_model_50epochs.h5')
   ```

### 性能优化建议

1. **GPU加速**
   - 安装CUDA和cuDNN
   - 监控GPU使用率
   - 调整批处理大小

2. **内存优化**
   - 使用数据生成器避免内存溢出
   - 定期清理临时文件
   - 监控内存使用

3. **训练加速**
   - 使用混合精度训练
   - 预训练模型微调
   - 数据预处理优化

## 🤝 贡献指南

### 开发环境设置

```bash
# 克隆开发分支
git clone -b develop <repository-url>

# 安装开发依赖
pip install -r requirements-dev.txt

# 运行测试
pytest tests/

# 代码格式化
black src/
flake8 src/
```

### 提交规范

- 使用清晰的commit信息
- 遵循代码风格指南
- 添加必要的测试用例
- 更新相关文档


## 🙏 致谢

- 感谢VGG团队提供的预训练模型
- 感谢TensorFlow和Keras开发团队
- 感谢医学影像数据提供者
- 感谢开源社区的支持

---

<div align="center">

**🎯 项目目标**: 推动AI在医学影像诊断中的应用发展

**⚡ 技术栈**: TensorFlow + VGG16 + Python + Jupyter

**🔬 应用领域**: 医学影像分析 + 深度学习 + 计算机视觉

</div>
