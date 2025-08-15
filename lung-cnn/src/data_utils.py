"""
COVID-19 肺部CT图像分类项目 - 数据处理工具模块
=============================================

负责数据加载、预处理、分割、增强和验证等功能
基于notebook中的数据处理流程重构而成

"""

import os
import shutil
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input

from config import get_config


class DataStructureValidator:
    """数据结构验证器"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
    
    def check_data_structure(self) -> bool:
        """
        分析和验证数据结构
        
        Returns:
            bool: 如果数据结构有效则返回True
        """
        print("=" * 60)
        print("数据结构分析")
        print("=" * 60)
        
        train_valid = self._check_training_data()
        test_valid = self._check_testing_data()
        
        print("=" * 60)
        return train_valid and test_valid
    
    def _check_training_data(self) -> bool:
        """检查训练数据结构"""
        train_path = self.config.TRAIN_PATH
        
        if not train_path.exists():
            print(f"❌ 训练数据路径不存在: {train_path}")
            return False
        
        print(f"✅ 训练数据路径存在: {train_path}")
        
        # 获取子目录
        subdirs = [d for d in os.listdir(train_path) 
                  if os.path.isdir(train_path / d)]
        print(f"训练数据子目录: {subdirs}")
        
        if len(subdirs) != 2:
            print(f"⚠️ 期望2个类别目录，实际发现 {len(subdirs)} 个")
            if len(subdirs) == 0:
                return False
        
        # 分析每个类别的图像数量
        total_images = 0
        class_distribution = {}
        
        for subdir in subdirs:
            subdir_path = train_path / subdir
            img_files = self._get_image_files(subdir_path)
            img_count = len(img_files)
            total_images += img_count
            class_distribution[subdir] = img_count
            print(f"  {subdir}: {img_count} 张图像")
        
        print(f"总训练图像数: {total_images}")
        
        # 检查类别平衡
        if len(class_distribution) >= 2:
            counts = list(class_distribution.values())
            balance_ratio = min(counts) / max(counts) if max(counts) > 0 else 0
            print(f"类别平衡比例: {balance_ratio:.3f}")
            
            if balance_ratio < 0.3:
                print("⚠️ 检测到严重的类别不平衡，建议增加数据增强")
            elif balance_ratio < 0.7:
                print("⚠️ 检测到轻微的类别不平衡")
            else:
                print("✅ 类别分布相对平衡")
        
        return total_images > 0
    
    def _check_testing_data(self) -> bool:
        """检查测试数据结构"""
        test_path = self.config.TEST_PATH
        
        if not test_path.exists():
            print(f"❌ 测试数据路径不存在: {test_path}")
            return False
        
        print(f"✅ 测试数据路径存在: {test_path}")
        
        # 检查子目录和文件
        items = os.listdir(test_path)
        subdirs = [d for d in items if os.path.isdir(test_path / d)]
        root_files = self._get_image_files(test_path)
        
        print(f"测试数据子目录: {subdirs}")
        print(f"根目录图像文件: {len(root_files)} 张")
        
        total_test_images = len(root_files)
        
        # 检查子目录中的图像
        for subdir in subdirs:
            subdir_path = test_path / subdir
            if subdir_path.is_dir():
                img_files = self._get_image_files(subdir_path)
                img_count = len(img_files)
                total_test_images += img_count
                print(f"  {subdir}: {img_count} 张图像")
        
        print(f"总测试图像数: {total_test_images}")
        return total_test_images > 0
    
    def _get_image_files(self, directory: Union[str, Path]) -> List[str]:
        """获取目录中的图像文件列表"""
        directory = Path(directory)
        if not directory.exists():
            return []
        
        image_files = []
        for ext in self.config.SUPPORTED_IMAGE_FORMATS:
            image_files.extend(directory.glob(f"*{ext}"))
            image_files.extend(directory.glob(f"*{ext.upper()}"))
        
        return [f.name for f in image_files]


class DataProcessor:
    """主要的数据处理类"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.validator = DataStructureValidator(self.config)
        
        # 设置随机种子
        random.seed(self.config.RANDOM_SEED)
        np.random.seed(self.config.RANDOM_SEED)
    
    def validate_data_structure(self) -> bool:
        """验证数据结构"""
        return self.validator.check_data_structure()
    
    def setup_directories(self, version: str = None) -> None:
        """
        创建和清理训练验证目录
        
        Args:
            version: 版本名称，如果为None则使用当前版本
        """
        if version is None:
            version = self.config.CURRENT_VERSION
        
        print(f"为 {version} 版本设置目录...")
        
        # 获取目录路径
        paths = self.config.get_file_paths(version)
        train_dir = paths["train_dir"]
        val_dir = paths["val_dir"]
        
        # 删除现有目录
        for dir_path in [train_dir, val_dir]:
            if dir_path.exists():
                shutil.rmtree(dir_path)
                print(f"删除现有目录: {dir_path}")
        
        # 创建新目录
        for dir_path in [train_dir, val_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            for class_name in self.config.CLASS_NAMES:
                (dir_path / class_name).mkdir(exist_ok=True)
            print(f"创建目录: {dir_path}")
        
        print("✅ 目录设置完成")
    
    def split_data(self, version: str = None) -> bool:
        """
        将训练数据分割为训练集和验证集
        
        Args:
            version: 版本名称，如果为None则使用当前版本
            
        Returns:
            bool: 分割成功返回True
        """
        if version is None:
            version = self.config.CURRENT_VERSION
        
        print(f"\n为 {version} 版本分割数据...")
        
        # 获取类别文件夹
        train_path = self.config.TRAIN_PATH
        class_folders = [d for d in os.listdir(train_path) 
                        if os.path.isdir(train_path / d)]
        
        if len(class_folders) == 0:
            print("❌ 训练数据目录中未找到类别文件夹")
            return False
        
        # 创建类别映射
        class_mapping = self._create_class_mapping(class_folders)
        if not class_mapping:
            print("❌ 无法创建有效的类别映射")
            return False
        
        # 获取目标目录
        paths = self.config.get_file_paths(version)
        train_dir = paths["train_dir"]
        val_dir = paths["val_dir"]
        
        # 执行数据分割
        split_summary = self._perform_data_split(
            class_mapping, train_dir, val_dir
        )
        
        # 打印分割摘要
        self._print_split_summary(split_summary, version)
        
        return len(split_summary) > 0
    
    def _create_class_mapping(self, class_folders: List[str]) -> Dict[str, str]:
        """创建类别映射"""
        if len(class_folders) != 2:
            print(f"⚠️ 发现 {len(class_folders)} 个类别文件夹，期望2个")
            print(f"类别文件夹: {class_folders}")
            
            if len(class_folders) == 2:
                # 自动映射：假设有 'yes' 和 'no' 或其他命名
                if 'no' in class_folders and 'yes' in class_folders:
                    class_mapping = {'no': 'no', 'yes': 'yes'}
                else:
                    # 按字母顺序映射
                    sorted_folders = sorted(class_folders)
                    class_mapping = {sorted_folders[0]: 'no', sorted_folders[1]: 'yes'}
                print(f"类别映射: {class_mapping}")
                return class_mapping
            else:
                print("❌ 请确保训练数据恰好有2个类别文件夹")
                return {}
        else:
            # 标准情况
            return {folder: folder for folder in class_folders}
    
    def _perform_data_split(self, class_mapping: Dict[str, str], 
                          train_dir: Path, val_dir: Path) -> Dict[str, Dict]:
        """执行数据分割"""
        split_summary = {}
        total_train = 0
        total_val = 0
        
        for original_class, target_class in class_mapping.items():
            class_path = self.config.TRAIN_PATH / original_class
            
            # 获取图像文件
            files = []
            for ext in self.config.SUPPORTED_IMAGE_FORMATS:
                files.extend(list(class_path.glob(f"*{ext}")))
                files.extend(list(class_path.glob(f"*{ext.upper()}")))
            
            files = [f.name for f in files]  # 只保留文件名
            
            if len(files) == 0:
                print(f"⚠️ {original_class} 文件夹中未找到图像文件")
                continue
            
            # 随机打乱并分割
            random.shuffle(files)
            split_point = int(self.config.TRAIN_VAL_SPLIT * len(files))
            
            train_files = files[:split_point]
            val_files = files[split_point:]
            
            # 复制训练数据
            for file_name in train_files:
                src = class_path / file_name
                dst = train_dir / target_class / file_name
                shutil.copy2(src, dst)
            
            # 复制验证数据
            for file_name in val_files:
                src = class_path / file_name
                dst = val_dir / target_class / file_name
                shutil.copy2(src, dst)
            
            # 记录统计信息
            total_train += len(train_files)
            total_val += len(val_files)
            
            split_summary[original_class] = {
                'target_class': target_class,
                'total': len(files),
                'train': len(train_files),
                'val': len(val_files),
                'train_ratio': len(train_files) / len(files),
                'val_ratio': len(val_files) / len(files)
            }
            
            print(f"{original_class} -> {target_class}: "
                  f"{len(train_files)} 训练, {len(val_files)} 验证")
        
        return split_summary
    
    def _print_split_summary(self, split_summary: Dict[str, Dict], version: str) -> None:
        """打印数据分割摘要"""
        total_train = sum(stats['train'] for stats in split_summary.values())
        total_val = sum(stats['val'] for stats in split_summary.values())
        
        print(f"\n{version} 版本详细分割摘要:")
        for class_name, stats in split_summary.items():
            print(f"  {class_name}: {stats['total']} 总计 | "
                  f"{stats['train']} 训练 ({stats['train_ratio']:.1%}) | "
                  f"{stats['val']} 验证 ({stats['val_ratio']:.1%})")
        
        print(f"\n总计: {total_train} 训练图像, {total_val} 验证图像")
        print("✅ 数据分割完成")
    
    def create_data_generators(self, version: str = None) -> Tuple[ImageDataGenerator, ImageDataGenerator]:
        """
        创建训练和验证数据生成器
        
        Args:
            version: 版本名称，如果为None则使用当前版本
            
        Returns:
            训练数据生成器和验证数据生成器的元组
        """
        if version is None:
            version = self.config.CURRENT_VERSION
        
        # 获取增强参数
        aug_params = self.config.CURRENT_AUGMENTATION
        
        # 创建训练数据生成器（带增强）
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            **aug_params
        )
        
        # 创建验证数据生成器（无增强）
        val_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input
        )
        
        print("数据生成器创建完成，使用以下增强策略:")
        for key, value in aug_params.items():
            if key != 'fill_mode':
                print(f"- {key}: {value}")
        
        # 获取目录路径
        paths = self.config.get_file_paths(version)
        train_dir = paths["train_dir"]
        val_dir = paths["val_dir"]
        
        # 创建数据流
        train_generator = train_datagen.flow_from_directory(
            str(train_dir),
            target_size=self.config.IMG_SIZE,
            batch_size=self.config.BATCH_SIZE,
            class_mode='binary',
            seed=self.config.RANDOM_SEED,
            shuffle=True
        )
        
        validation_generator = val_datagen.flow_from_directory(
            str(val_dir),
            target_size=self.config.IMG_SIZE,
            batch_size=self.config.BATCH_SIZE,
            class_mode='binary',
            seed=self.config.RANDOM_SEED,
            shuffle=False
        )
        
        print(f"\n✅ 数据生成器创建成功 ({version} 版本)")
        print(f"训练样本: {train_generator.samples}")
        print(f"验证样本: {validation_generator.samples}")
        print(f"类别数: {len(train_generator.class_indices)}")
        print(f"类别索引: {train_generator.class_indices}")
        
        return train_generator, validation_generator
    
    def plot_sample_images(self, source_path: str = None, n: int = 20, 
                          title: str = "样本图像", version: str = None) -> None:
        """
        显示数据集的样本图像
        
        Args:
            source_path: 图像路径，如果为None则使用处理后的训练目录
            n: 显示的图像数量
            title: 图表标题
            version: 版本名称
        """
        if source_path is None:
            if version is None:
                version = self.config.CURRENT_VERSION
            paths = self.config.get_file_paths(version)
            source_path = str(paths["train_dir"])
        
        files_list = []
        labels_list = []
        
        # 收集所有图像文件
        for root, dirs, files in os.walk(source_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in self.config.SUPPORTED_IMAGE_FORMATS):
                    files_list.append(os.path.join(root, file))
                    label = os.path.basename(root)
                    labels_list.append(label)
        
        if not files_list:
            print(f"在 {source_path} 中未找到图像文件")
            return
        
        # 随机选择图像
        combined = list(zip(files_list, labels_list))
        random.shuffle(combined)
        files_list, labels_list = zip(*combined)
        
        # 限制显示数量
        n = min(n, len(files_list))
        cols = 5
        rows = (n + cols - 1) // cols
        
        plt.figure(figsize=(20, 4 * rows))
        plt.suptitle(f"{title} - {version} 版本", fontsize=16, fontweight='bold')
        
        for i in range(n):
            file_path, label = files_list[i], labels_list[i]
            
            # 读取并显示图像
            img = cv2.imread(file_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.subplot(rows, cols, i + 1)
                plt.imshow(img)
                plt.title(f'类别: {label}', fontsize=12, fontweight='bold')
                plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def plot_class_distribution(self, version: str = None) -> Tuple[Dict[str, int], Dict[str, int]]:
        """
        绘制训练集和验证集的类别分布
        
        Args:
            version: 版本名称
            
        Returns:
            训练集和验证集的类别分布字典
        """
        if version is None:
            version = self.config.CURRENT_VERSION
        
        paths = self.config.get_file_paths(version)
        train_dir = paths["train_dir"]
        val_dir = paths["val_dir"]
        
        # 统计类别分布
        train_classes = {'yes': 0, 'no': 0}
        val_classes = {'yes': 0, 'no': 0}
        
        for class_name in ['yes', 'no']:
            train_path = train_dir / class_name
            val_path = val_dir / class_name
            
            if train_path.exists():
                train_files = []
                for ext in self.config.SUPPORTED_IMAGE_FORMATS:
                    train_files.extend(list(train_path.glob(f"*{ext}")))
                    train_files.extend(list(train_path.glob(f"*{ext.upper()}")))
                train_classes[class_name] = len(train_files)
            
            if val_path.exists():
                val_files = []
                for ext in self.config.SUPPORTED_IMAGE_FORMATS:
                    val_files.extend(list(val_path.glob(f"*{ext}")))
                    val_files.extend(list(val_path.glob(f"*{ext.upper()}")))
                val_classes[class_name] = len(val_files)
        
        # 创建可视化
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 训练集分布
        colors = ['lightcoral', 'lightblue']
        bars1 = ax1.bar(train_classes.keys(), train_classes.values(), color=colors)
        ax1.set_title(f'训练集类别分布 - {version}', fontweight='bold', fontsize=14)
        ax1.set_ylabel('图像数量')
        
        for i, (k, v) in enumerate(train_classes.items()):
            ax1.text(i, v + max(train_classes.values()) * 0.01, str(v), 
                    ha='center', fontweight='bold', fontsize=12)
        
        # 验证集分布
        bars2 = ax2.bar(val_classes.keys(), val_classes.values(), color=colors)
        ax2.set_title(f'验证集类别分布 - {version}', fontweight='bold', fontsize=14)
        ax2.set_ylabel('图像数量')
        
        for i, (k, v) in enumerate(val_classes.items()):
            ax2.text(i, v + max(val_classes.values()) * 0.01, str(v), 
                    ha='center', fontweight='bold', fontsize=12)
        
        plt.tight_layout()
        plt.show()
        
        return train_classes, val_classes


class TestDataLoader:
    """测试数据加载器"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
    
    def load_and_preprocess_test_images(self, test_path: str = None) -> Tuple[np.ndarray, List[str]]:
        """
        加载和预处理测试图像
        
        Args:
            test_path: 测试图像路径，如果为None则使用配置中的路径
            
        Returns:
            (预处理后的图像数组, 有效图像路径列表)
        """
        if test_path is None:
            test_path = str(self.config.TEST_PATH)
        
        print(f"从 {test_path} 加载测试图像...")
        
        # 收集所有测试图像路径
        test_image_paths = []
        
        # 搜索子目录
        for root, dirs, files in os.walk(test_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in self.config.SUPPORTED_IMAGE_FORMATS):
                    test_image_paths.append(os.path.join(root, file))
        
        if not test_image_paths:
            print("❌ 未找到测试图像!")
            return np.array([]), []
        
        print(f"✅ 发现 {len(test_image_paths)} 张测试图像")
        
        # 预处理图像
        test_images = []
        valid_paths = []
        failed_images = []
        
        print("预处理测试图像...")
        for img_path in tqdm(test_image_paths, desc="加载图像"):
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    # 调整大小并预处理
                    img_resized = cv2.resize(img, self.config.IMG_SIZE)
                    img_preprocessed = preprocess_input(img_resized)
                    
                    # 基本质量检查
                    if not np.allclose(img_preprocessed, 0):  # 确保图像不是全黑
                        test_images.append(img_preprocessed)
                        valid_paths.append(img_path)
                    else:
                        failed_images.append((img_path, "图像似乎已损坏"))
                else:
                    failed_images.append((img_path, "无法读取图像"))
            except Exception as e:
                failed_images.append((img_path, f"错误: {str(e)}"))
        
        # 报告失败的图像
        if failed_images:
            print(f"⚠️ 处理失败的图像: {len(failed_images)} 张")
            for path, reason in failed_images[:3]:  # 显示前3个失败案例
                print(f"  {os.path.basename(path)}: {reason}")
            if len(failed_images) > 3:
                print(f"  ... 还有 {len(failed_images) - 3} 个")
        
        if not test_images:
            print("❌ 没有有效的测试图像加载成功!")
            return np.array([]), []
        
        test_images = np.array(test_images)
        print(f"✅ 成功加载 {len(test_images)} 张测试图像")
        print(f"图像数组形状: {test_images.shape}")
        
        return test_images, valid_paths


# 便捷函数
def process_data(version: str = None) -> Tuple[ImageDataGenerator, ImageDataGenerator]:
    """
    完整的数据处理流程
    
    Args:
        version: 版本名称
        
    Returns:
        (训练数据生成器, 验证数据生成器)
    """
    processor = DataProcessor()
    
    # 验证数据结构
    if not processor.validate_data_structure():
        raise RuntimeError("数据结构验证失败")
    
    # 设置目录并分割数据
    processor.setup_directories(version)
    if not processor.split_data(version):
        raise RuntimeError("数据分割失败")
    
    # 创建并返回数据生成器
    return processor.create_data_generators(version)


def load_test_data(test_path: str = None) -> Tuple[np.ndarray, List[str]]:
    """
    加载测试数据的便捷函数
    
    Args:
        test_path: 测试数据路径
        
    Returns:
        (预处理后的图像数组, 图像路径列表)
    """
    loader = TestDataLoader()
    return loader.load_and_preprocess_test_images(test_path)


if __name__ == "__main__":
    # 测试数据处理模块
    print("测试数据处理模块...")
    
    # 创建数据处理器
    processor = DataProcessor()
    
    # 验证数据结构
    print("\n1. 验证数据结构...")
    if processor.validate_data_structure():
        print("✅ 数据结构验证通过")
    else:
        print("❌ 数据结构验证失败")
        exit(1)
    
    # 测试10轮版本的数据处理
    print("\n2. 测试10轮版本数据处理...")
    processor.config.switch_version("10_epochs")
    processor.setup_directories()
    processor.split_data()
    
    # 创建数据生成器
    print("\n3. 创建数据生成器...")
    train_gen, val_gen = processor.create_data_generators()
    
    # 显示样本图像
    print("\n4. 显示样本图像...")
    processor.plot_sample_images(n=10)
    
    # 显示类别分布
    print("\n5. 显示类别分布...")
    train_dist, val_dist = processor.plot_class_distribution()
    
    print("\n数据处理模块测试完成!")