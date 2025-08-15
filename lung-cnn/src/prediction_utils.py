"""
COVID-19 肺部CT图像分类项目 - 预测和评估模块
==========================================

负责模型预测、结果分析、评估指标计算和结果保存等功能
基于notebook中的预测相关代码重构而成

"""

import os
import json
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix

from config import get_config


class ModelPredictor:
    """模型预测器"""
    
    def __init__(self, model_path: str = None, config=None):
        """
        初始化预测器
        
        Args:
            model_path: 模型文件路径
            config: 配置对象
        """
        self.config = config or get_config()
        self.model = None
        self.model_path = model_path
        self.class_names = {0: "no", 1: "yes"}  # 默认类别映射
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> bool:
        """
        加载训练好的模型
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            bool: 加载成功返回True
        """
        try:
            print(f"🔄 加载模型: {model_path}")
            self.model = load_model(model_path)
            self.model_path = model_path
            print("✅ 模型加载成功")
            
            # 显示模型信息
            print(f"📊 模型输入形状: {self.model.input_shape}")
            print(f"📊 模型输出形状: {self.model.output_shape}")
            
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {str(e)}")
            return False
    
    def predict_single_image(self, image_path: str) -> Dict[str, Any]:
        """
        预测单张图像
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            预测结果字典
        """
        if self.model is None:
            raise ValueError("模型未加载，请先调用 load_model()")
        
        try:
            # 加载和预处理图像
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"无法读取图像: {image_path}")
            
            # 预处理
            img_resized = cv2.resize(img, self.config.IMG_SIZE)
            img_preprocessed = preprocess_input(img_resized)
            img_batch = np.expand_dims(img_preprocessed, axis=0)
            
            # 预测
            prediction = self.model.predict(img_batch, verbose=0)[0][0]
            predicted_class = int(prediction > self.config.PREDICTION_THRESHOLD)
            confidence = prediction if predicted_class == 1 else 1 - prediction
            uncertainty = 1 - abs(prediction - 0.5) * 2
            
            return {
                'filename': os.path.basename(image_path),
                'path': image_path,
                'prediction': self.class_names[predicted_class],
                'confidence': float(confidence),
                'uncertainty': float(uncertainty),
                'raw_probability': float(prediction),
                'predicted_class': predicted_class
            }
            
        except Exception as e:
            print(f"❌ 预测图像失败 {image_path}: {str(e)}")
            return None
    
    def predict_batch_images(self, image_paths: List[str], 
                           batch_size: int = None) -> List[Dict[str, Any]]:
        """
        批量预测图像
        
        Args:
            image_paths: 图像路径列表
            batch_size: 批次大小
            
        Returns:
            预测结果列表
        """
        if self.model is None:
            raise ValueError("模型未加载，请先调用 load_model()")
        
        if batch_size is None:
            batch_size = self.config.BATCH_SIZE
        
        results = []
        valid_images = []
        valid_paths = []
        failed_images = []
        
        print(f"🔄 预处理 {len(image_paths)} 张图像...")
        
        # 预处理所有图像
        for img_path in tqdm(image_paths, desc="加载图像"):
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    img_resized = cv2.resize(img, self.config.IMG_SIZE)
                    img_preprocessed = preprocess_input(img_resized)
                    
                    # 基本质量检查
                    if not np.allclose(img_preprocessed, 0):
                        valid_images.append(img_preprocessed)
                        valid_paths.append(img_path)
                    else:
                        failed_images.append((img_path, "图像似乎已损坏"))
                else:
                    failed_images.append((img_path, "无法读取图像"))
            except Exception as e:
                failed_images.append((img_path, f"处理错误: {str(e)}"))
        
        # 报告失败的图像
        if failed_images:
            print(f"⚠️ {len(failed_images)} 张图像处理失败")
            for path, reason in failed_images[:3]:
                print(f"  {os.path.basename(path)}: {reason}")
            if len(failed_images) > 3:
                print(f"  ... 还有 {len(failed_images) - 3} 个")
        
        if not valid_images:
            print("❌ 没有有效图像用于预测")
            return []
        
        print(f"🔄 开始预测 {len(valid_images)} 张有效图像...")
        
        # 批量预测
        valid_images = np.array(valid_images)
        predictions = self.model.predict(valid_images, batch_size=batch_size, verbose=1)
        predicted_classes = (predictions > self.config.PREDICTION_THRESHOLD).astype(int).flatten()
        
        # 处理结果
        for img_path, pred_prob, pred_class in zip(valid_paths, predictions.flatten(), predicted_classes):
            confidence = pred_prob if pred_class == 1 else 1 - pred_prob
            uncertainty = 1 - abs(pred_prob - 0.5) * 2
            
            results.append({
                'filename': os.path.basename(img_path),
                'path': img_path,
                'subdirectory': self._get_subdirectory(img_path),
                'prediction': self.class_names[pred_class],
                'confidence': float(confidence),
                'uncertainty': float(uncertainty),
                'raw_probability': float(pred_prob),
                'predicted_class': pred_class
            })
        
        print(f"✅ 预测完成，处理了 {len(results)} 张图像")
        return results
    
    def predict_test_directory(self, test_path: str = None) -> List[Dict[str, Any]]:
        """
        预测测试目录中的所有图像
        
        Args:
            test_path: 测试目录路径，如果为None则使用配置中的路径
            
        Returns:
            预测结果列表
        """
        if test_path is None:
            test_path = str(self.config.TEST_PATH)
        
        print(f"📂 从 {test_path} 收集测试图像...")
        
        # 收集所有图像路径
        image_paths = []
        for root, dirs, files in os.walk(test_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in self.config.SUPPORTED_IMAGE_FORMATS):
                    image_paths.append(os.path.join(root, file))
        
        if not image_paths:
            print("❌ 测试目录中未找到图像")
            return []
        
        print(f"✅ 发现 {len(image_paths)} 张测试图像")
        
        # 按子目录分组统计
        subdir_stats = {}
        for path in image_paths:
            subdir = self._get_subdirectory(path)
            subdir_stats[subdir] = subdir_stats.get(subdir, 0) + 1
        
        print("📊 子目录分布:")
        for subdir, count in subdir_stats.items():
            print(f"  {subdir}: {count} 张图像")
        
        # 批量预测
        return self.predict_batch_images(image_paths)
    
    def _get_subdirectory(self, image_path: str) -> str:
        """获取图像所在的子目录名"""
        path_parts = Path(image_path).parts
        test_path_parts = Path(self.config.TEST_PATH).parts
        
        if len(path_parts) > len(test_path_parts):
            relative_parts = path_parts[len(test_path_parts):]
            if len(relative_parts) > 1:
                return relative_parts[0]
        
        return "root"
    
    def set_class_mapping(self, class_mapping: Dict[int, str]) -> None:
        """
        设置类别映射
        
        Args:
            class_mapping: 类别索引到名称的映射
        """
        self.class_names = class_mapping
        print(f"✅ 类别映射已更新: {self.class_names}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        if self.model is None:
            return {}
        
        return {
            'model_path': self.model_path,
            'input_shape': str(self.model.input_shape),
            'output_shape': str(self.model.output_shape),
            'total_params': self.model.count_params(),
            'trainable_params': sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
        }


class ResultsAnalyzer:
    """结果分析器"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
    
    def analyze_predictions(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析预测结果
        
        Args:
            results: 预测结果列表
            
        Returns:
            分析结果字典
        """
        if not results:
            print("❌ 没有预测结果可分析")
            return {}
        
        print(f"📊 分析 {len(results)} 个预测结果...")
        
        # 基本统计
        total_predictions = len(results)
        class_counts = {}
        confidence_stats = {'yes': [], 'no': []}
        uncertainty_stats = {'yes': [], 'no': []}
        subdir_stats = {}
        
        for result in results:
            pred = result['prediction']
            subdir = result.get('subdirectory', 'unknown')
            
            # 类别统计
            class_counts[pred] = class_counts.get(pred, 0) + 1
            confidence_stats[pred].append(result['confidence'])
            uncertainty_stats[pred].append(result['uncertainty'])
            
            # 子目录统计
            if subdir not in subdir_stats:
                subdir_stats[subdir] = {'yes': 0, 'no': 0, 'total': 0}
            subdir_stats[subdir][pred] += 1
            subdir_stats[subdir]['total'] += 1
        
        # 置信度分析
        all_confidences = [r['confidence'] for r in results]
        all_uncertainties = [r['uncertainty'] for r in results]
        
        # 置信度等级分布
        confidence_levels = {
            'very_high': sum(1 for c in all_confidences if c > self.config.CONFIDENCE_LEVELS['very_high']),
            'high': sum(1 for c in all_confidences if self.config.CONFIDENCE_LEVELS['high'] <= c <= self.config.CONFIDENCE_LEVELS['very_high']),
            'medium': sum(1 for c in all_confidences if self.config.CONFIDENCE_LEVELS['medium'] <= c < self.config.CONFIDENCE_LEVELS['high']),
            'low': sum(1 for c in all_confidences if c < self.config.CONFIDENCE_LEVELS['medium'])
        }
        
        analysis = {
            'total_predictions': total_predictions,
            'class_distribution': class_counts,
            'confidence_statistics': {
                'overall': {
                    'mean': np.mean(all_confidences),
                    'median': np.median(all_confidences),
                    'std': np.std(all_confidences),
                    'min': np.min(all_confidences),
                    'max': np.max(all_confidences)
                },
                'by_class': {}
            },
            'uncertainty_statistics': {
                'overall': {
                    'mean': np.mean(all_uncertainties),
                    'median': np.median(all_uncertainties),
                    'std': np.std(all_uncertainties)
                },
                'by_class': {}
            },
            'confidence_levels': confidence_levels,
            'subdirectory_distribution': subdir_stats,
            'raw_uncertainties': all_uncertainties  # 用于后续分析
        }
        
        # 按类别的详细统计
        for class_name in ['yes', 'no']:
            if class_name in confidence_stats and confidence_stats[class_name]:
                analysis['confidence_statistics']['by_class'][class_name] = {
                    'mean': np.mean(confidence_stats[class_name]),
                    'std': np.std(confidence_stats[class_name]),
                    'count': len(confidence_stats[class_name])
                }
                
                analysis['uncertainty_statistics']['by_class'][class_name] = {
                    'mean': np.mean(uncertainty_stats[class_name]),
                    'std': np.std(uncertainty_stats[class_name])
                }
        
        return analysis
    
    def print_analysis_report(self, analysis: Dict[str, Any]) -> None:
        """打印详细的分析报告"""
        if not analysis:
            print("❌ 无分析结果可显示")
            return
        
        print("\n" + "="*70)
        print("🔬 预测结果分析报告")
        print("="*70)
        
        # 基本统计
        total = analysis['total_predictions']
        print(f"\n📊 基本统计:")
        print(f"总预测数: {total}")
        
        # 类别分布
        print(f"\n🎯 类别分布:")
        for class_name, count in analysis['class_distribution'].items():
            percentage = (count / total) * 100
            covid_status = "COVID-19 阳性" if class_name == 'yes' else "COVID-19 阴性"
            print(f"  {covid_status}: {count:3d} 张图像 ({percentage:5.1f}%)")
        
        # 置信度统计
        conf_stats = analysis['confidence_statistics']['overall']
        print(f"\n📈 整体置信度统计:")
        print(f"  平均置信度: {conf_stats['mean']:.4f}")
        print(f"  中位数置信度: {conf_stats['median']:.4f}")
        print(f"  标准差: {conf_stats['std']:.4f}")
        print(f"  范围: {conf_stats['min']:.4f} - {conf_stats['max']:.4f}")
        
        # 按类别的置信度
        print(f"\n📊 按类别置信度统计:")
        for class_name, stats in analysis['confidence_statistics']['by_class'].items():
            covid_status = "COVID-19 阳性" if class_name == 'yes' else "COVID-19 阴性"
            print(f"  {covid_status}: 平均 {stats['mean']:.4f} (±{stats['std']:.4f}), {stats['count']} 样本")
        
        # 置信度等级分布
        print(f"\n🎯 置信度等级分布:")
        levels = analysis['confidence_levels']
        level_descriptions = {
            'very_high': '极高 (>0.95)',
            'high': '高 (0.9-0.95)',
            'medium': '中等 (0.7-0.9)',
            'low': '低 (<0.7)'
        }
        
        for level, count in levels.items():
            percentage = (count / total) * 100
            desc = level_descriptions.get(level, level)
            print(f"  {desc}: {count} 张图像 ({percentage:.1f}%)")
        
        # 不确定性分析
        unc_stats = analysis['uncertainty_statistics']['overall']
        print(f"\n🎲 不确定性分析:")
        print(f"  平均不确定性: {unc_stats['mean']:.4f}")
        print(f"  中位数不确定性: {unc_stats['median']:.4f}")
        
        high_uncertainty = sum(1 for u in analysis.get('raw_uncertainties', []) if u > 0.8)
        if high_uncertainty > 0:
            print(f"  高不确定性样本 (>0.8): {high_uncertainty} 张")
        
        # 子目录分布
        if len(analysis['subdirectory_distribution']) > 1:
            print(f"\n📁 子目录分布:")
            for subdir, stats in analysis['subdirectory_distribution'].items():
                if stats['total'] > 0:
                    yes_pct = (stats['yes'] / stats['total']) * 100
                    no_pct = (stats['no'] / stats['total']) * 100
                    print(f"  {subdir}: {stats['total']} 张图像")
                    print(f"    阳性: {stats['yes']} ({yes_pct:.1f}%), 阴性: {stats['no']} ({no_pct:.1f}%)")
        
        # 预测质量评估
        print(f"\n📈 预测质量评估:")
        high_conf_count = levels['very_high'] + levels['high']
        high_conf_percentage = (high_conf_count / total) * 100
        
        if high_conf_percentage > 80:
            print(f"  ✅ 预测质量优秀 - {high_conf_percentage:.1f}% 高置信度预测")
        elif high_conf_percentage > 60:
            print(f"  ✅ 预测质量良好 - {high_conf_percentage:.1f}% 高置信度预测")
        else:
            print(f"  ⚠️ 预测质量需关注 - 仅{high_conf_percentage:.1f}% 高置信度预测")
        
        print("="*70)
    
    def get_top_confident_predictions(self, results: List[Dict[str, Any]], 
                                    n: int = 10) -> List[Dict[str, Any]]:
        """获取置信度最高的预测结果"""
        return sorted(results, key=lambda x: x['confidence'], reverse=True)[:n]
    
    def get_uncertain_predictions(self, results: List[Dict[str, Any]], 
                                threshold: float = 0.7) -> List[Dict[str, Any]]:
        """获取不确定性高的预测结果"""
        return [r for r in results if r['uncertainty'] > threshold]
    
    def get_low_confidence_predictions(self, results: List[Dict[str, Any]], 
                                     threshold: float = 0.7) -> List[Dict[str, Any]]:
        """获取低置信度的预测结果"""
        return [r for r in results if r['confidence'] < threshold]
    
    def generate_classification_report(self, true_labels: List[str], 
                                     predicted_labels: List[str]) -> str:
        """
        生成分类报告（需要真实标签）
        
        Args:
            true_labels: 真实标签列表
            predicted_labels: 预测标签列表
            
        Returns:
            分类报告字符串
        """
        return classification_report(true_labels, predicted_labels, 
                                   target_names=['COVID-19 阴性', 'COVID-19 阳性'])


class ResultsSaver:
    """结果保存器"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
    
    def save_predictions_json(self, results: List[Dict[str, Any]], 
                            filepath: str, analysis: Dict[str, Any] = None,
                            model_info: Dict[str, Any] = None) -> None:
        """
        保存预测结果为JSON格式
        
        Args:
            results: 预测结果列表
            filepath: 保存路径
            analysis: 分析结果
            model_info: 模型信息
        """
        # 准备保存数据
        save_data = {
            'metadata': {
                'total_predictions': len(results),
                'timestamp': self._get_timestamp(),
                'config_version': self.config.CURRENT_VERSION,
                'prediction_threshold': self.config.PREDICTION_THRESHOLD,
                'image_size': self.config.IMG_SIZE,
                'class_names': self.config.CLASS_NAMES
            },
            'predictions': results
        }
        
        # 添加模型信息
        if model_info:
            save_data['metadata']['model_info'] = model_info
        
        # 添加分析结果
        if analysis:
            save_data['analysis'] = analysis
        
        # 确保目录存在
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # 保存JSON
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            print(f"✅ 预测结果已保存: {filepath}")
        except Exception as e:
            print(f"❌ 保存预测结果失败: {str(e)}")
    
    def save_analysis_report(self, analysis: Dict[str, Any], 
                           filepath: str) -> None:
        """
        保存分析报告为文本格式
        
        Args:
            analysis: 分析结果
            filepath: 保存路径
        """
        # 确保目录存在
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("COVID-19 肺部CT分类 - 预测结果分析报告\n")
                f.write("="*50 + "\n\n")
                f.write(f"生成时间: {self._get_timestamp()}\n")
                f.write(f"总预测数: {analysis['total_predictions']}\n\n")
                
                # 类别分布
                f.write("类别分布:\n")
                total = analysis['total_predictions']
                for class_name, count in analysis['class_distribution'].items():
                    percentage = (count / total) * 100
                    covid_status = "COVID-19 阳性" if class_name == 'yes' else "COVID-19 阴性"
                    f.write(f"  {covid_status}: {count} ({percentage:.1f}%)\n")
                
                # 置信度统计
                conf_stats = analysis['confidence_statistics']['overall']
                f.write(f"\n置信度统计:\n")
                f.write(f"  平均: {conf_stats['mean']:.4f}\n")
                f.write(f"  中位数: {conf_stats['median']:.4f}\n")
                f.write(f"  标准差: {conf_stats['std']:.4f}\n")
                f.write(f"  范围: {conf_stats['min']:.4f} - {conf_stats['max']:.4f}\n")
                
                # 置信度等级
                f.write(f"\n置信度等级分布:\n")
                for level, count in analysis['confidence_levels'].items():
                    percentage = (count / total) * 100
                    f.write(f"  {level}: {count} ({percentage:.1f}%)\n")
                
                # 不确定性统计
                unc_stats = analysis['uncertainty_statistics']['overall']
                f.write(f"\n不确定性统计:\n")
                f.write(f"  平均: {unc_stats['mean']:.4f}\n")
                f.write(f"  中位数: {unc_stats['median']:.4f}\n")
                f.write(f"  标准差: {unc_stats['std']:.4f}\n")
            
            print(f"✅ 分析报告已保存: {filepath}")
        except Exception as e:
            print(f"❌ 保存分析报告失败: {str(e)}")
    
    def save_csv_summary(self, results: List[Dict[str, Any]], 
                        filepath: str) -> None:
        """
        保存预测结果的CSV摘要
        
        Args:
            results: 预测结果列表
            filepath: 保存路径
        """
        try:
            import pandas as pd
            
            # 准备CSV数据
            csv_data = []
            for result in results:
                csv_data.append({
                    'filename': result['filename'],
                    'prediction': result['prediction'],
                    'confidence': result['confidence'],
                    'uncertainty': result['uncertainty'],
                    'raw_probability': result['raw_probability'],
                    'subdirectory': result.get('subdirectory', 'unknown')
                })
            
            # 创建DataFrame并保存
            df = pd.DataFrame(csv_data)
            df.to_csv(filepath, index=False, encoding='utf-8')
            print(f"✅ CSV摘要已保存: {filepath}")
            
        except ImportError:
            print("⚠️ 需要pandas库来保存CSV文件")
        except Exception as e:
            print(f"❌ 保存CSV摘要失败: {str(e)}")
    
    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class EnsemblePredictor:
    """集成预测器"""
    
    def __init__(self, model_paths: List[str], config=None):
        """
        初始化集成预测器
        
        Args:
            model_paths: 模型路径列表
            config: 配置对象
        """
        self.config = config or get_config()
        self.models = []
        self.model_paths = model_paths
        
        self._load_models()
    
    def _load_models(self):
        """加载所有模型"""
        print(f"🔄 加载 {len(self.model_paths)} 个模型进行集成...")
        
        for i, model_path in enumerate(self.model_paths):
            try:
                model = load_model(model_path)
                self.models.append(model)
                print(f"  ✅ 模型 {i+1} 加载成功: {Path(model_path).name}")
            except Exception as e:
                print(f"  ❌ 模型 {i+1} 加载失败: {str(e)}")
        
        if not self.models:
            raise RuntimeError("没有成功加载任何模型")
        
        print(f"✅ 成功加载 {len(self.models)} 个模型用于集成预测")
    
    def predict_ensemble(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        使用集成方法进行预测
        
        Args:
            image_paths: 图像路径列表
            
        Returns:
            集成预测结果列表
        """
        if not self.models:
            raise RuntimeError("没有可用的模型")
        
        print(f"🔄 使用 {len(self.models)} 个模型进行集成预测...")
        
        # 预处理图像
        valid_images = []
        valid_paths = []
        
        for img_path in tqdm(image_paths, desc="预处理图像"):
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    img_resized = cv2.resize(img, self.config.IMG_SIZE)
                    img_preprocessed = preprocess_input(img_resized)
                    valid_images.append(img_preprocessed)
                    valid_paths.append(img_path)
            except Exception as e:
                print(f"⚠️ 跳过图像 {img_path}: {str(e)}")
        
        if not valid_images:
            return []
        
        valid_images = np.array(valid_images)
        
        # 获取所有模型的预测
        all_predictions = []
        for i, model in enumerate(self.models):
            print(f"🔄 模型 {i+1} 预测中...")
            preds = model.predict(valid_images, batch_size=self.config.BATCH_SIZE, verbose=0)
            all_predictions.append(preds.flatten())
        
        # 集成预测结果
        results = []
        for j, img_path in enumerate(valid_paths):
            # 获取所有模型对这张图像的预测
            image_predictions = [pred[j] for pred in all_predictions]
            
            # 计算集成结果
            ensemble_prob = np.mean(image_predictions)
            prob_std = np.std(image_predictions)
            predicted_class = int(ensemble_prob > self.config.PREDICTION_THRESHOLD)
            confidence = ensemble_prob if predicted_class == 1 else 1 - ensemble_prob
            
            # 集成不确定性（结合模型间差异）
            model_disagreement = prob_std
            prediction_uncertainty = 1 - abs(ensemble_prob - 0.5) * 2
            ensemble_uncertainty = max(prediction_uncertainty, model_disagreement)
            
            results.append({
                'filename': os.path.basename(img_path),
                'path': img_path,
                'prediction': 'yes' if predicted_class == 1 else 'no',
                'confidence': float(confidence),
                'uncertainty': float(ensemble_uncertainty),
                'raw_probability': float(ensemble_prob),
                'predicted_class': predicted_class,
                'model_predictions': [float(p) for p in image_predictions],
                'model_std': float(prob_std),
                'ensemble_method': 'mean'
            })
        
        print(f"✅ 集成预测完成，处理了 {len(results)} 张图像")
        return results


# 便捷函数
def predict_and_analyze(model_path: str, test_path: str = None, 
                       save_results: bool = True, version: str = None) -> Tuple[List[Dict], Dict[str, Any]]:
    """
    完整的预测和分析流程
    
    Args:
        model_path: 模型路径
        test_path: 测试数据路径
        save_results: 是否保存结果
        version: 版本名称
        
    Returns:
        (预测结果, 分析结果)
    """
    config = get_config()
    if version:
        config.switch_version(version)
    
    print("🚀 开始完整的预测和分析流程")
    print("="*50)
    
    # 创建预测器
    predictor = ModelPredictor(model_path, config)
    if not predictor.model:
        raise RuntimeError("模型加载失败")
    
    # 执行预测
    print("\n📊 执行预测...")
    results = predictor.predict_test_directory(test_path)
    if not results:
        raise RuntimeError("预测失败或无测试数据")
    
    # 分析结果
    print("\n📈 分析预测结果...")
    analyzer = ResultsAnalyzer(config)
    analysis = analyzer.analyze_predictions(results)
    analyzer.print_analysis_report(analysis)
    
    # 保存结果
    if save_results:
        print("\n💾 保存结果...")
        saver = ResultsSaver(config)
        
        # 生成文件路径
        version_str = version or config.CURRENT_VERSION
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        json_path = config.PREDICTIONS_PATH / f"detailed_predictions_{version_str}_{timestamp}.json"
        report_path = config.RESULTS_PATH / f"analysis_report_{version_str}_{timestamp}.txt"
        csv_path = config.PREDICTIONS_PATH / f"predictions_summary_{version_str}_{timestamp}.csv"
        
        # 获取模型信息
        model_info = predictor.get_model_info()
        
        # 保存文件
        saver.save_predictions_json(results, str(json_path), analysis, model_info)
        saver.save_analysis_report(analysis, str(report_path))
        saver.save_csv_summary(results, str(csv_path))
        
        print(f"✅ 结果已保存:")
        print(f"  📄 JSON详情: {json_path}")
        print(f"  📄 分析报告: {report_path}")
        print(f"  📄 CSV摘要: {csv_path}")
    
    return results, analysis


def predict_with_ensemble(model_paths: List[str], test_path: str = None,
                         save_results: bool = True) -> Tuple[List[Dict], Dict[str, Any]]:
    """
    使用集成模型进行预测和分析
    
    Args:
        model_paths: 模型路径列表
        test_path: 测试数据路径
        save_results: 是否保存结果
        
    Returns:
        (预测结果, 分析结果)
    """
    config = get_config()
    
    print("🔗 开始集成模型预测流程")
    print("="*50)
    
    # 收集测试图像
    if test_path is None:
        test_path = str(config.TEST_PATH)
    
    image_paths = []
    for root, dirs, files in os.walk(test_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in config.SUPPORTED_IMAGE_FORMATS):
                image_paths.append(os.path.join(root, file))
    
    if not image_paths:
        raise RuntimeError("未找到测试图像")
    
    # 创建集成预测器
    ensemble = EnsemblePredictor(model_paths, config)
    
    # 执行集成预测
    results = ensemble.predict_ensemble(image_paths)
    if not results:
        raise RuntimeError("集成预测失败")
    
    # 分析结果
    analyzer = ResultsAnalyzer(config)
    analysis = analyzer.analyze_predictions(results)
    analyzer.print_analysis_report(analysis)
    
    # 保存结果
    if save_results:
        saver = ResultsSaver(config)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        json_path = config.PREDICTIONS_PATH / f"ensemble_predictions_{timestamp}.json"
        report_path = config.RESULTS_PATH / f"ensemble_analysis_{timestamp}.txt"
        
        # 添加集成信息
        ensemble_info = {
            'ensemble_method': 'mean',
            'model_count': len(model_paths),
            'model_paths': model_paths
        }
        
        saver.save_predictions_json(results, str(json_path), analysis, ensemble_info)
        saver.save_analysis_report(analysis, str(report_path))
    
    return results, analysis


def quick_predict(model_path: str, image_path: str) -> Dict[str, Any]:
    """
    快速预测单张图像
    
    Args:
        model_path: 模型路径
        image_path: 图像路径
        
    Returns:
        预测结果字典
    """
    predictor = ModelPredictor(model_path)
    if not predictor.model:
        raise RuntimeError("模型加载失败")
    
    result = predictor.predict_single_image(image_path)
    if result:
        print(f"📊 预测结果:")
        print(f"  文件: {result['filename']}")
        print(f"  预测: {result['prediction']} ({'COVID-19 阳性' if result['prediction'] == 'yes' else 'COVID-19 阴性'})")
        print(f"  置信度: {result['confidence']:.4f}")
        print(f"  不确定性: {result['uncertainty']:.4f}")
        
    return result


def compare_model_predictions(model_paths: List[str], test_path: str = None,
                            model_names: List[str] = None) -> Dict[str, List[Dict]]:
    """
    比较多个模型的预测结果
    
    Args:
        model_paths: 模型路径列表
        test_path: 测试数据路径
        model_names: 模型名称列表
        
    Returns:
        包含各模型预测结果的字典
    """
    config = get_config()
    
    if model_names is None:
        model_names = [f"模型{i+1}" for i in range(len(model_paths))]
    
    print(f"🔄 比较 {len(model_paths)} 个模型的预测结果")
    print("="*50)
    
    all_results = {}
    
    for model_path, model_name in zip(model_paths, model_names):
        print(f"\n📊 预测 {model_name}...")
        try:
            predictor = ModelPredictor(model_path, config)
            results = predictor.predict_test_directory(test_path)
            all_results[model_name] = results
            print(f"✅ {model_name} 预测完成: {len(results)} 张图像")
        except Exception as e:
            print(f"❌ {model_name} 预测失败: {str(e)}")
            all_results[model_name] = []
    
    # 分析一致性
    if len(all_results) >= 2:
        print(f"\n📈 预测一致性分析:")
        _analyze_prediction_consistency(all_results)
    
    return all_results


def _analyze_prediction_consistency(all_results: Dict[str, List[Dict]]) -> None:
    """分析多个模型预测结果的一致性"""
    model_names = list(all_results.keys())
    
    if len(model_names) < 2:
        return
    
    # 找到共同预测的图像
    common_files = set()
    for results in all_results.values():
        if results:
            files = {r['filename'] for r in results}
            if not common_files:
                common_files = files
            else:
                common_files &= files
    
    if not common_files:
        print("  ❌ 没有共同预测的图像")
        return
    
    print(f"  📊 共同预测图像数: {len(common_files)}")
    
    # 计算一致性
    agreements = 0
    disagreements = []
    
    for filename in common_files:
        predictions = []
        for model_name in model_names:
            for result in all_results[model_name]:
                if result['filename'] == filename:
                    predictions.append(result['prediction'])
                    break
        
        if len(set(predictions)) == 1:
            agreements += 1
        else:
            disagreements.append((filename, predictions))
    
    consistency_rate = agreements / len(common_files) * 100
    print(f"  📈 预测一致性: {consistency_rate:.1f}% ({agreements}/{len(common_files)})")
    
    if disagreements:
        print(f"  ⚠️ 不一致预测数: {len(disagreements)}")
        # 显示前几个不一致的案例
        for i, (filename, preds) in enumerate(disagreements[:3]):
            pred_str = ", ".join([f"{model_names[j]}:{p}" for j, p in enumerate(preds)])
            print(f"    {filename}: {pred_str}")
        if len(disagreements) > 3:
            print(f"    ... 还有 {len(disagreements) - 3} 个不一致案例")


# 测试和演示函数
def demo_prediction_pipeline():
    """演示完整的预测流程"""
    print("🎬 COVID-19 预测模块演示")
    print("="*50)
    
    config = get_config()
    
    # 创建模拟结果用于演示
    mock_results = [
        {
            'filename': 'covid_positive_001.jpg',
            'path': '/test/covid_positive_001.jpg',
            'subdirectory': 'test',
            'prediction': 'yes',
            'confidence': 0.95,
            'uncertainty': 0.1,
            'raw_probability': 0.95,
            'predicted_class': 1
        },
        {
            'filename': 'covid_negative_001.jpg',
            'path': '/test/covid_negative_001.jpg',
            'subdirectory': 'test',
            'prediction': 'no',
            'confidence': 0.88,
            'uncertainty': 0.24,
            'raw_probability': 0.12,
            'predicted_class': 0
        },
        {
            'filename': 'uncertain_case_001.jpg',
            'path': '/test/uncertain_case_001.jpg',
            'subdirectory': 'test',
            'prediction': 'yes',
            'confidence': 0.62,
            'uncertainty': 0.76,
            'raw_probability': 0.62,
            'predicted_class': 1
        }
    ]
    
    print("\n📊 演示结果分析...")
    analyzer = ResultsAnalyzer(config)
    analysis = analyzer.analyze_predictions(mock_results)
    analyzer.print_analysis_report(analysis)
    
    print("\n💾 演示结果保存...")
    saver = ResultsSaver(config)
    
    # 创建临时保存路径
    temp_json = config.PREDICTIONS_PATH / "demo_predictions.json"
    temp_report = config.RESULTS_PATH / "demo_report.txt"
    
    saver.save_predictions_json(mock_results, str(temp_json), analysis)
    saver.save_analysis_report(analysis, str(temp_report))
    
    print("\n✅ 演示完成!")


if __name__ == "__main__":
    # 运行演示
    demo_prediction_pipeline()