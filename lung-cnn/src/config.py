"""
COVID-19 肺部CT图像分类项目 - 配置管理模块
================================================

统一管理项目的所有配置参数，包括路径、模型参数、训练参数等
基于notebook中的配置重构而成

"""

import os
import random
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict, List, Tuple, Any


class ProjectConfig:
    """项目配置管理类"""
    
    def __init__(self):
        """初始化配置参数"""
        self._setup_base_paths()
        self._setup_model_params()
        self._setup_training_params()
        self._setup_data_augmentation()
        self._setup_file_names()
        self._setup_thresholds()
        
    def _setup_base_paths(self):
        """设置基础路径配置"""
        # 项目根目录 - 从src目录向上一级
        self.PROJECT_ROOT = Path(__file__).parent.parent.absolute()
        
        # 数据路径
        self.DATA_ROOT = self.PROJECT_ROOT / "data"
        self.TRAIN_PATH = self.DATA_ROOT / "train_covid19"
        self.TEST_PATH = self.DATA_ROOT / "test_healthcare"
        self.PROCESSED_PATH = self.DATA_ROOT / "processed"
        
        # 模型和结果路径
        self.MODELS_PATH = self.PROJECT_ROOT / "models"
        self.RESULTS_PATH = self.PROJECT_ROOT / "results"
        self.PLOTS_PATH = self.RESULTS_PATH / "plots"
        self.PREDICTIONS_PATH = self.RESULTS_PATH / "predictions"
        self.LOGS_PATH = self.RESULTS_PATH / "logs"
        
        # 其他路径
        self.NOTEBOOKS_PATH = self.PROJECT_ROOT / "notebooks"
        self.DOCS_PATH = self.PROJECT_ROOT / "docs"
        
    def _setup_model_params(self):
        """设置模型相关参数"""
        # 图像参数
        self.IMG_SIZE = (224, 224)  # VGG16要求的输入尺寸
        self.IMG_CHANNELS = 3
        self.INPUT_SHAPE = self.IMG_SIZE + (self.IMG_CHANNELS,)
        
        # 模型架构参数
        self.DROPOUT_RATE = 0.5
        self.DENSE_DROPOUT_RATE = 0.3
        self.DENSE_UNITS = 128
        
        # 50轮版本的增强架构参数
        self.ENHANCED_DROPOUT_RATE = 0.6
        self.ENHANCED_DENSE_UNITS = [256, 128, 64]  # 多层Dense架构
        self.ENHANCED_DROPOUT_RATES = [0.4, 0.3, 0.2]
        
    def _setup_training_params(self):
        """设置训练相关参数"""
        # 基础训练参数
        self.BATCH_SIZE = 32
        self.LEARNING_RATE = 1e-3
        self.RANDOM_SEED = 100
        self.TRAIN_VAL_SPLIT = 0.6  # 60%训练，40%验证
        
        # 版本控制
        self.CURRENT_VERSION = "10_epochs"  # 默认版本
        self.AVAILABLE_VERSIONS = ["10_epochs", "50_epochs"]
        
        # 版本特定参数
        self.VERSION_CONFIGS = {
            "10_epochs": {
                "num_epochs": 10,
                "early_stopping_patience": 5,
                "reduce_lr_patience": 3,
                "min_lr": 1e-6,
                "architecture": "standard"
            },
            "50_epochs": {
                "num_epochs": 50,
                "early_stopping_patience": 10,
                "reduce_lr_patience": 5,
                "min_lr": 1e-7,
                "architecture": "enhanced"
            }
        }
        
        # 当前版本的参数（默认10轮）
        self._update_version_params("10_epochs")
        
    def _setup_data_augmentation(self):
        """设置数据增强参数"""
        # 10轮版本的基础数据增强
        self.BASIC_AUGMENTATION = {
            "rotation_range": 20,
            "width_shift_range": 0.2,
            "height_shift_range": 0.2,
            "shear_range": 0.2,
            "zoom_range": 0.2,
            "horizontal_flip": True,
            "vertical_flip": False,  # 医学图像通常不垂直翻转
            "fill_mode": "nearest"
        }
        
        # 50轮版本的增强数据增强
        self.ENHANCED_AUGMENTATION = {
            "rotation_range": 30,
            "width_shift_range": 0.25,
            "height_shift_range": 0.25,
            "shear_range": 0.25,
            "zoom_range": 0.25,
            "horizontal_flip": True,
            "vertical_flip": False,
            "brightness_range": [0.8, 1.2],  # 新增亮度变化
            "fill_mode": "nearest"
        }
        
        # 当前使用的增强参数
        self.CURRENT_AUGMENTATION = self.BASIC_AUGMENTATION.copy()
        
    def _setup_file_names(self):
        """设置文件命名规则"""
        # 基础文件名模板
        self.MODEL_FILENAME_TEMPLATE = "covid_classifier_vgg16_{version}.h5"
        self.BEST_MODEL_TEMPLATE = "best_model_{version}.h5"
        self.HISTORY_FILENAME_TEMPLATE = "training_history_{version}.pkl"
        self.SUMMARY_FILENAME_TEMPLATE = "model_summary_{version}.txt"
        
        # 处理后数据目录名
        self.TRAIN_DIR_TEMPLATE = "Train_covid_{version}"
        self.VAL_DIR_TEMPLATE = "Val_covid_{version}"
        
        # 类别名称
        self.CLASS_NAMES = ["no", "yes"]  # no=阴性, yes=阳性
        self.CLASS_MAPPING = {0: "no", 1: "yes"}
        
        # 支持的图像格式
        self.SUPPORTED_IMAGE_FORMATS = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        
    def _setup_thresholds(self):
        """设置预测阈值和置信度参数"""
        self.PREDICTION_THRESHOLD = 0.5
        self.HIGH_CONFIDENCE_THRESHOLD = 0.9
        self.MEDIUM_CONFIDENCE_THRESHOLD = 0.7
        self.LOW_CONFIDENCE_THRESHOLD = 0.5
        
        # 置信度等级定义
        self.CONFIDENCE_LEVELS = {
            "very_high": 0.95,
            "high": 0.9,
            "medium": 0.7,
            "low": 0.5
        }
        
    def switch_version(self, version: str) -> None:
        """
        切换训练版本配置
        
        Args:
            version: 版本名称 ("10_epochs" 或 "50_epochs")
        """
        if version not in self.AVAILABLE_VERSIONS:
            raise ValueError(f"版本必须是 {self.AVAILABLE_VERSIONS} 之一")
        
        self.CURRENT_VERSION = version
        self._update_version_params(version)
        
        print(f"✅ 已切换到 {version} 版本配置")
        
    def _update_version_params(self, version: str) -> None:
        """更新当前版本的参数"""
        config = self.VERSION_CONFIGS[version]
        
        # 更新训练参数
        self.NUM_EPOCHS = config["num_epochs"]
        self.EARLY_STOPPING_PATIENCE = config["early_stopping_patience"]
        self.REDUCE_LR_PATIENCE = config["reduce_lr_patience"]
        self.MIN_LR = config["min_lr"]
        self.ARCHITECTURE_TYPE = config["architecture"]
        
        # 更新数据增强
        if version == "50_epochs":
            self.CURRENT_AUGMENTATION = self.ENHANCED_AUGMENTATION.copy()
        else:
            self.CURRENT_AUGMENTATION = self.BASIC_AUGMENTATION.copy()
        
    def get_file_paths(self, version: str = None) -> Dict[str, Path]:
        """
        获取指定版本的文件路径
        
        Args:
            version: 版本名称，如果为None则使用当前版本
            
        Returns:
            包含所有文件路径的字典
        """
        if version is None:
            version = self.CURRENT_VERSION
            
        return {
            "model": self.MODELS_PATH / self.MODEL_FILENAME_TEMPLATE.format(version=version),
            "best_model": self.MODELS_PATH / self.BEST_MODEL_TEMPLATE.format(version=version),
            "history": self.MODELS_PATH / self.HISTORY_FILENAME_TEMPLATE.format(version=version),
            "summary": self.DOCS_PATH / self.SUMMARY_FILENAME_TEMPLATE.format(version=version),
            "train_dir": self.PROCESSED_PATH / self.TRAIN_DIR_TEMPLATE.format(version=version),
            "val_dir": self.PROCESSED_PATH / self.VAL_DIR_TEMPLATE.format(version=version),
            "predictions": self.PREDICTIONS_PATH / f"predictions_{version}.json",
            "plots": self.PLOTS_PATH / f"training_results_{version}.png"
        }
    
    def create_directories(self) -> None:
        """创建必要的目录结构"""
        directories = [
            self.DATA_ROOT,
            self.PROCESSED_PATH,
            self.MODELS_PATH,
            self.RESULTS_PATH,
            self.PLOTS_PATH,
            self.PREDICTIONS_PATH,
            self.LOGS_PATH,
            self.DOCS_PATH
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        print("✅ 项目目录结构创建完成")
    
    def validate_data_paths(self) -> bool:
        """
        验证数据路径是否存在
        
        Returns:
            bool: 如果所有必要路径都存在则返回True
        """
        required_paths = [
            (self.TRAIN_PATH, "训练数据路径"),
            (self.TEST_PATH, "测试数据路径")
        ]
        
        all_valid = True
        for path, description in required_paths:
            if not path.exists():
                print(f"❌ {description}不存在: {path}")
                all_valid = False
            else:
                print(f"✅ {description}验证通过: {path}")
                
        return all_valid
    
    def setup_reproducibility(self) -> None:
        """设置随机种子以确保结果可重现"""
        random.seed(self.RANDOM_SEED)
        np.random.seed(self.RANDOM_SEED)
        tf.random.set_seed(self.RANDOM_SEED)
        
        # 设置TensorFlow的确定性操作
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        
        print(f"✅ 随机种子设置为 {self.RANDOM_SEED}，确保结果可重现")
    
    def setup_gpu(self) -> bool:
        """
        配置GPU设置
        
        Returns:
            bool: 如果GPU可用并配置成功则返回True
        """
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ GPU配置完成，发现 {len(gpus)} 个GPU")
                return True
            except RuntimeError as e:
                print(f"❌ GPU配置错误: {e}")
                return False
        else:
            print("⚠️ 未发现GPU，将使用CPU训练")
            return False
    
    def get_model_architecture_config(self) -> Dict[str, Any]:
        """获取当前版本的模型架构配置"""
        if self.ARCHITECTURE_TYPE == "enhanced":
            return {
                "dropout_rate": self.ENHANCED_DROPOUT_RATE,
                "dense_units": self.ENHANCED_DENSE_UNITS,
                "dropout_rates": self.ENHANCED_DROPOUT_RATES,
                "metrics": ['accuracy', 'precision', 'recall']
            }
        else:
            return {
                "dropout_rate": self.DROPOUT_RATE,
                "dense_units": [self.DENSE_UNITS],
                "dropout_rates": [self.DENSE_DROPOUT_RATE],
                "metrics": ['accuracy']
            }
    
    def print_current_config(self) -> None:
        """打印当前配置信息"""
        print("=" * 60)
        print("COVID-19 肺部CT分类 - 当前配置")
        print("=" * 60)
        print(f"项目根目录: {self.PROJECT_ROOT}")
        print(f"当前版本: {self.CURRENT_VERSION}")
        print(f"训练轮数: {self.NUM_EPOCHS}")
        print(f"批次大小: {self.BATCH_SIZE}")
        print(f"学习率: {self.LEARNING_RATE}")
        print(f"图像尺寸: {self.IMG_SIZE}")
        print(f"架构类型: {self.ARCHITECTURE_TYPE}")
        print(f"训练数据: {self.TRAIN_PATH}")
        print(f"测试数据: {self.TEST_PATH}")
        print(f"早停耐心: {self.EARLY_STOPPING_PATIENCE}")
        print("=" * 60)
    
    def get_callbacks_config(self) -> Dict[str, Any]:
        """获取回调函数配置"""
        return {
            "early_stopping": {
                "monitor": "val_accuracy",
                "patience": self.EARLY_STOPPING_PATIENCE,
                "restore_best_weights": True,
                "mode": "max",
                "verbose": 1
            },
            "reduce_lr": {
                "monitor": "val_loss",
                "factor": 0.2,
                "patience": self.REDUCE_LR_PATIENCE,
                "min_lr": self.MIN_LR,
                "verbose": 1
            },
            "model_checkpoint": {
                "monitor": "val_accuracy",
                "save_best_only": True,
                "mode": "max",
                "verbose": 1
            }
        }


# 全局配置实例
config = ProjectConfig()

# 便捷的配置函数
def get_config() -> ProjectConfig:
    """获取全局配置实例"""
    return config

def switch_version(version: str) -> None:
    """切换版本的便捷函数"""
    config.switch_version(version)

def setup_environment() -> bool:
    """设置完整的运行环境"""
    print("🚀 设置COVID-19肺部CT分类运行环境")
    print("=" * 50)
    
    # 创建目录
    config.create_directories()
    
    # 验证数据路径
    if not config.validate_data_paths():
        print("❌ 数据路径验证失败")
        return False
    
    # 设置随机种子
    config.setup_reproducibility()
    
    # 配置GPU
    gpu_available = config.setup_gpu()
    
    # 打印配置
    config.print_current_config()
    
    print("✅ 环境设置完成")
    return True


if __name__ == "__main__":
    # 测试配置模块
    print("测试配置模块...")
    
    # 设置环境
    setup_environment()
    
    # 测试版本切换
    print("\n测试版本切换...")
    switch_version("50_epochs")
    config.print_current_config()
    
    # 测试文件路径获取
    print("\n测试文件路径获取...")
    paths = config.get_file_paths()
    for key, path in paths.items():
        print(f"{key}: {path}")