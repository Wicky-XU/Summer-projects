"""
COVID-19 肺部CT图像分类项目 - 模型构建和训练模块
===============================================

负责模型架构构建、训练配置、回调设置和训练执行等功能
基于notebook中的模型相关代码重构而成

"""

import os
import time
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, 
    Callback, CSVLogger
)

from config import get_config


class GPUManager:
    """GPU管理器"""
    
    @staticmethod
    def setup_gpu() -> bool:
        """
        配置GPU设置以获得最佳性能
        
        Returns:
            bool: GPU是否可用并配置成功
        """
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # 启用内存增长以避免占用所有GPU内存
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ GPU配置完成，发现 {len(gpus)} 个GPU")
                
                # 显示GPU信息
                for i, gpu in enumerate(gpus):
                    print(f"  GPU {i}: {gpu.name}")
                
                return True
            except RuntimeError as e:
                print(f"❌ GPU配置错误: {e}")
                return False
        else:
            print("⚠️ 未发现GPU，将使用CPU训练")
            return False
    
    @staticmethod
    def get_gpu_memory_info() -> Dict[str, Any]:
        """获取GPU内存信息"""
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                return {
                    "gpu_count": len(gpus),
                    "gpu_available": True,
                    "gpu_names": [gpu.name for gpu in gpus]
                }
            else:
                return {"gpu_available": False, "gpu_count": 0}
        except Exception as e:
            return {"gpu_available": False, "error": str(e)}


class ModelBuilder:
    """模型构建器"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
    
    def create_model(self, version: str = None) -> tf.keras.Model:
        """
        根据版本创建模型
        
        Args:
            version: 模型版本 ("10_epochs" 或 "50_epochs")
            
        Returns:
            编译好的Keras模型
        """
        if version is None:
            version = self.config.CURRENT_VERSION
        
        # 获取架构配置
        arch_config = self.config.get_model_architecture_config()
        
        if version == "50_epochs" or arch_config.get("metrics") == ['accuracy', 'precision', 'recall']:
            return self._create_enhanced_model(arch_config)
        else:
            return self._create_standard_model(arch_config)
    
    def _create_standard_model(self, arch_config: Dict[str, Any]) -> tf.keras.Model:
        """
        创建标准VGG16模型（10轮版本）
        
        Args:
            arch_config: 架构配置
            
        Returns:
            编译好的模型
        """
        print("创建标准VGG16模型架构...")
        
        # 加载预训练VGG16模型
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=self.config.INPUT_SHAPE
        )
        
        # 创建自定义分类器
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dropout(arch_config["dropout_rate"]),
            Dense(arch_config["dense_units"][0], activation='relu'),
            BatchNormalization(),
            Dropout(self.config.DENSE_DROPOUT_RATE),
            Dense(1, activation='sigmoid')  # 二分类
        ])
        
        # 冻结预训练层
        base_model.trainable = False
        
        # 编译模型
        model.compile(
            optimizer=Adam(learning_rate=self.config.LEARNING_RATE),
            loss='binary_crossentropy',
            metrics=arch_config["metrics"]
        )
        
        print("✅ 标准VGG16模型创建完成")
        return model
    
    def _create_enhanced_model(self, arch_config: Dict[str, Any]) -> tf.keras.Model:
        """
        创建增强VGG16模型（50轮版本）
        
        Args:
            arch_config: 架构配置
            
        Returns:
            编译好的模型
        """
        print("创建增强VGG16模型架构...")
        
        # 加载预训练VGG16模型
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=self.config.INPUT_SHAPE
        )
        
        # 创建增强的自定义分类器
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(arch_config["dropout_rate"])(x)
        
        # 多层Dense网络
        dense_units = arch_config["dense_units"]
        dropout_rates = arch_config["dropout_rates"]
        
        for i, (units, dropout) in enumerate(zip(dense_units, dropout_rates)):
            x = Dense(units, activation='relu', name=f'dense_{i+1}')(x)
            x = BatchNormalization(name=f'bn_{i+1}')(x)
            x = Dropout(dropout, name=f'dropout_{i+1}')(x)
        
        # 输出层
        predictions = Dense(1, activation='sigmoid', name='predictions')(x)
        
        # 创建模型
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # 冻结预训练层
        base_model.trainable = False
        
        # 编译模型
        model.compile(
            optimizer=Adam(learning_rate=self.config.LEARNING_RATE),
            loss='binary_crossentropy',
            metrics=arch_config["metrics"]
        )
        
        print("✅ 增强VGG16模型创建完成")
        return model
    
    def create_fine_tuned_model(self, base_model_path: str, 
                              unfreeze_layers: int = 4) -> tf.keras.Model:
        """
        创建微调模型
        
        Args:
            base_model_path: 基础模型路径
            unfreeze_layers: 解冻的顶层数量
            
        Returns:
            微调后的模型
        """
        print(f"从 {base_model_path} 加载模型进行微调...")
        
        # 加载基础模型
        model = load_model(base_model_path)
        
        # 获取VGG16基础模型
        base_model = model.layers[0]  # VGG16基础模型
        base_model.trainable = True
        
        # 冻结除顶层外的所有层
        for layer in base_model.layers[:-unfreeze_layers]:
            layer.trainable = False
        
        # 使用更低的学习率进行微调
        model.compile(
            optimizer=Adam(learning_rate=self.config.LEARNING_RATE / 10),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print(f"✅ 微调模型创建完成，解冻了顶部 {unfreeze_layers} 层")
        return model
    
    @staticmethod
    def print_model_summary(model: tf.keras.Model, version: str = "current") -> None:
        """
        打印详细的模型摘要
        
        Args:
            model: Keras模型
            version: 版本标识
        """
        print("\n" + "="*70)
        print(f"模型架构摘要 - {version.upper()} 版本")
        print("="*70)
        
        # 显示模型结构
        model.summary()
        
        # 计算参数统计
        trainable_params = sum([np.prod(v.get_shape().as_list()) 
                              for v in model.trainable_variables])
        non_trainable_params = sum([np.prod(v.get_shape().as_list()) 
                                  for v in model.non_trainable_variables])
        total_params = trainable_params + non_trainable_params
        
        print(f"\n参数统计:")
        print(f"  可训练参数: {trainable_params:,}")
        print(f"  不可训练参数: {non_trainable_params:,}")
        print(f"  总参数: {total_params:,}")
        
        # 显示内存估计
        memory_mb = (total_params * 4) / (1024 * 1024)  # 假设float32
        print(f"  估计内存使用: {memory_mb:.1f} MB")
        
        # 显示模型配置
        print(f"\n模型配置:")
        print(f"  输入形状: {model.input_shape}")
        print(f"  输出形状: {model.output_shape}")
        print(f"  层数: {len(model.layers)}")
        
        print("="*70)


class TrainingCallbacks:
    """训练回调管理器"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
    
    def create_callbacks(self, version: str = None, 
                        save_path: str = None) -> List[Callback]:
        """
        创建训练回调列表
        
        Args:
            version: 版本名称
            save_path: 模型保存路径
            
        Returns:
            回调函数列表
        """
        if version is None:
            version = self.config.CURRENT_VERSION
        
        if save_path is None:
            paths = self.config.get_file_paths(version)
            save_path = str(paths["best_model"])
        
        # 获取回调配置
        callback_config = self.config.get_callbacks_config()
        
        callbacks = []
        
        # 早停回调
        early_stopping = EarlyStopping(
            **callback_config["early_stopping"]
        )
        callbacks.append(early_stopping)
        
        # 模型检查点
        model_checkpoint = ModelCheckpoint(
            filepath=save_path,
            save_weights_only=False,
            **callback_config["model_checkpoint"]
        )
        callbacks.append(model_checkpoint)
        
        # 学习率调度
        reduce_lr = ReduceLROnPlateau(
            **callback_config["reduce_lr"]
        )
        callbacks.append(reduce_lr)
        
        # CSV日志记录
        log_path = self.config.LOGS_PATH / f"training_{version}.csv"
        csv_logger = CSVLogger(str(log_path))
        callbacks.append(csv_logger)
        
        # 详细的训练监控回调
        verbose_callback = self._create_verbose_callback(version)
        callbacks.append(verbose_callback)
        
        print(f"创建了 {len(callbacks)} 个训练回调:")
        print("- 早停 (EarlyStopping)")
        print("- 模型检查点 (ModelCheckpoint)")
        print("- 学习率调度 (ReduceLROnPlateau)")
        print("- CSV日志记录 (CSVLogger)")
        print("- 详细监控 (VerboseCallback)")
        
        return callbacks
    
    def _create_verbose_callback(self, version: str) -> Callback:
        """创建详细的训练监控回调"""
        
        class EnhancedVerboseCallback(Callback):
            def __init__(self, version, num_epochs):
                super().__init__()
                self.version = version
                self.num_epochs = num_epochs
                self.epoch_start_time = None
                self.training_start_time = time.time()
            
            def on_train_begin(self, logs=None):
                print(f"\n🚀 开始 {self.version} 训练会话")
                print(f"目标轮数: {self.num_epochs}")
                print("="*50)
            
            def on_epoch_begin(self, epoch, logs=None):
                self.epoch_start_time = time.time()
                print(f"\n--- 轮次 {epoch + 1}/{self.num_epochs} ---")
            
            def on_epoch_end(self, epoch, logs=None):
                epoch_time = time.time() - self.epoch_start_time
                total_time = time.time() - self.training_start_time
                
                print(f"轮次 {epoch + 1}/{self.num_epochs} 完成 (用时 {epoch_time:.1f}s)")
                print(f"  训练   - 损失: {logs['loss']:.4f}, 准确率: {logs['accuracy']:.4f}")
                print(f"  验证   - 损失: {logs['val_loss']:.4f}, 准确率: {logs['val_accuracy']:.4f}")
                
                # 显示额外指标（如果有）
                if 'precision' in logs:
                    print(f"  精确度: {logs.get('precision', 0):.4f}, 召回率: {logs.get('recall', 0):.4f}")
                
                # 进度指示
                progress = (epoch + 1) / self.num_epochs * 100
                eta = (total_time / (epoch + 1)) * (self.num_epochs - epoch - 1)
                print(f"  进度: {progress:.1f}% | 剩余时间: {eta/60:.1f}分钟")
                
                # 过拟合检测
                if epoch > 2:  # 至少训练3轮后才检测
                    acc_gap = logs['accuracy'] - logs['val_accuracy']
                    if acc_gap > 0.15:
                        print(f"  ⚠️ 可能过拟合 (准确率差异: {acc_gap:.3f})")
                    elif acc_gap < -0.05:
                        print(f"  📈 验证表现更好 (差异: {acc_gap:.3f})")
            
            def on_train_end(self, logs=None):
                total_time = time.time() - self.training_start_time
                print(f"\n✅ {self.version} 训练完成!")
                print(f"总训练时间: {total_time/60:.1f}分钟")
        
        return EnhancedVerboseCallback(version, self.config.NUM_EPOCHS)


class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.model = None
        self.history = None
        self.callbacks_manager = TrainingCallbacks(self.config)
    
    def train_model(self, model: tf.keras.Model, 
                   train_generator, validation_generator,
                   version: str = None) -> tf.keras.callbacks.History:
        """
        训练模型
        
        Args:
            model: 要训练的模型
            train_generator: 训练数据生成器
            validation_generator: 验证数据生成器
            version: 版本名称
            
        Returns:
            训练历史
        """
        if version is None:
            version = self.config.CURRENT_VERSION
        
        self.model = model
        
        print(f"\n🚀 开始 {version} 模型训练")
        print("="*60)
        
        # 创建回调
        callbacks = self.callbacks_manager.create_callbacks(version)
        
        # 计算训练步数
        steps_per_epoch = train_generator.samples // self.config.BATCH_SIZE
        validation_steps = validation_generator.samples // self.config.BATCH_SIZE
        
        print(f"\n训练配置:")
        print(f"- 批次大小: {self.config.BATCH_SIZE}")
        print(f"- 训练轮数: {self.config.NUM_EPOCHS}")
        print(f"- 每轮步数: {steps_per_epoch}")
        print(f"- 验证步数: {validation_steps}")
        print(f"- 学习率: {self.config.LEARNING_RATE}")
        print(f"- 优化器: Adam")
        
        # 开始训练
        try:
            start_time = time.time()
            
            self.history = model.fit(
                train_generator,
                steps_per_epoch=steps_per_epoch,
                epochs=self.config.NUM_EPOCHS,
                validation_data=validation_generator,
                validation_steps=validation_steps,
                callbacks=callbacks,
                verbose=1  # 显示进度条
            )
            
            training_time = time.time() - start_time
            print(f"\n✅ 训练完成! 总用时: {training_time/60:.1f}分钟")
            
        except KeyboardInterrupt:
            print("\n⏹️ 训练被用户中断")
            return None
        except Exception as e:
            print(f"\n❌ 训练过程中出错: {str(e)}")
            raise e
        
        return self.history
    
    def save_model_and_history(self, version: str = None) -> Dict[str, str]:
        """
        保存模型和训练历史
        
        Args:
            version: 版本名称
            
        Returns:
            保存的文件路径字典
        """
        if version is None:
            version = self.config.CURRENT_VERSION
        
        if self.model is None or self.history is None:
            raise ValueError("没有可保存的模型或历史记录")
        
        paths = self.config.get_file_paths(version)
        saved_files = {}
        
        # 保存最终模型
        model_path = paths["model"]
        self.model.save(str(model_path))
        saved_files["model"] = str(model_path)
        print(f"✅ 最终模型已保存: {model_path}")
        
        # 保存训练历史
        history_data = {
            'history': self.history.history,
            'config': {
                'version': version,
                'epochs': self.config.NUM_EPOCHS,
                'batch_size': self.config.BATCH_SIZE,
                'learning_rate': self.config.LEARNING_RATE,
                'architecture': self.config.ARCHITECTURE_TYPE,
                'augmentation': self.config.CURRENT_AUGMENTATION
            },
            'training_info': {
                'total_epochs': len(self.history.history['accuracy']),
                'final_train_acc': self.history.history['accuracy'][-1],
                'final_val_acc': self.history.history['val_accuracy'][-1],
                'best_val_acc': max(self.history.history['val_accuracy']),
                'best_val_acc_epoch': np.argmax(self.history.history['val_accuracy']) + 1
            }
        }
        
        history_path = paths["history"]
        with open(history_path, 'wb') as f:
            pickle.dump(history_data, f)
        saved_files["history"] = str(history_path)
        print(f"✅ 训练历史已保存: {history_path}")
        
        # 保存模型摘要
        summary_path = paths["summary"]
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"COVID-19分类模型摘要 - {version} 版本\n")
            f.write("="*50 + "\n\n")
            f.write(f"训练完成轮数: {len(self.history.history['accuracy'])}/{self.config.NUM_EPOCHS}\n")
            f.write(f"最终训练准确率: {self.history.history['accuracy'][-1]:.4f}\n")
            f.write(f"最终验证准确率: {self.history.history['val_accuracy'][-1]:.4f}\n")
            f.write(f"最佳验证准确率: {max(self.history.history['val_accuracy']):.4f}\n")
            f.write(f"最佳准确率轮次: {np.argmax(self.history.history['val_accuracy']) + 1}\n")
            f.write(f"模型架构: {self.config.ARCHITECTURE_TYPE}\n")
            
            # 计算参数
            trainable_params = sum([np.prod(v.get_shape().as_list()) 
                                  for v in self.model.trainable_variables])
            total_params = sum([np.prod(v.get_shape().as_list()) 
                              for v in self.model.variables])
            f.write(f"可训练参数: {trainable_params:,}\n")
            f.write(f"总参数: {total_params:,}\n")
        
        saved_files["summary"] = str(summary_path)
        print(f"✅ 模型摘要已保存: {summary_path}")
        
        return saved_files
    
    def get_training_summary(self) -> Dict[str, Any]:
        """获取训练摘要信息"""
        if self.history is None:
            return {}
        
        history = self.history.history
        
        return {
            "epochs_completed": len(history['accuracy']),
            "final_train_accuracy": history['accuracy'][-1],
            "final_val_accuracy": history['val_accuracy'][-1],
            "final_train_loss": history['loss'][-1],
            "final_val_loss": history['val_loss'][-1],
            "best_val_accuracy": max(history['val_accuracy']),
            "best_val_accuracy_epoch": np.argmax(history['val_accuracy']) + 1,
            "min_val_loss": min(history['val_loss']),
            "min_val_loss_epoch": np.argmin(history['val_loss']) + 1,
            "overfitting_score": history['accuracy'][-1] - history['val_accuracy'][-1]
        }


# 便捷函数
def create_and_train_model(train_generator, validation_generator, 
                          version: str = None) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """
    创建并训练模型的完整流程
    
    Args:
        train_generator: 训练数据生成器
        validation_generator: 验证数据生成器
        version: 版本名称
        
    Returns:
        (训练好的模型, 训练历史)
    """
    config = get_config()
    if version:
        config.switch_version(version)
    
    # 设置GPU
    GPUManager.setup_gpu()
    
    # 创建模型
    builder = ModelBuilder(config)
    model = builder.create_model(version)
    builder.print_model_summary(model, version or config.CURRENT_VERSION)
    
    # 训练模型
    trainer = ModelTrainer(config)
    history = trainer.train_model(model, train_generator, validation_generator, version)
    
    if history:
        # 保存模型和历史
        saved_files = trainer.save_model_and_history(version)
        print(f"\n📁 保存的文件: {list(saved_files.values())}")
        
        # 打印训练摘要
        summary = trainer.get_training_summary()
        print(f"\n📊 训练摘要:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
    
    return model, history


if __name__ == "__main__":
    # 测试模型构建模块
    print("测试模型构建模块...")
    
    # 设置GPU
    print("\n1. 设置GPU...")
    gpu_info = GPUManager.get_gpu_memory_info()
    print(f"GPU信息: {gpu_info}")
    
    # 创建模型构建器
    print("\n2. 创建模型构建器...")
    builder = ModelBuilder()
    
    # 测试标准模型创建
    print("\n3. 创建标准模型...")
    model_10 = builder.create_model("10_epochs")
    builder.print_model_summary(model_10, "10_epochs")
    
    # 测试增强模型创建
    print("\n4. 创建增强模型...")
    model_50 = builder.create_model("50_epochs")
    builder.print_model_summary(model_50, "50_epochs")
    
    print("\n✅ 模型构建模块测试完成!")