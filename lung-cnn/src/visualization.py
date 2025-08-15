"""
COVID-19 肺部CT图像分类项目 - 可视化工具模块
==========================================

负责训练过程可视化、结果分析图表、模型比较等功能
基于notebook中的可视化代码重构而成

"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union

# 设置matplotlib中文支持和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('default')

from config import get_config


class TrainingVisualizer:
    """训练过程可视化器"""
    
    def __init__(self, history=None, config=None):
        """
        初始化可视化器
        
        Args:
            history: 训练历史对象或字典
            config: 配置对象
        """
        self.config = config or get_config()
        self.history = history
        
        if isinstance(history, dict):
            # 如果是字典格式的历史记录
            self.history_dict = history
        elif hasattr(history, 'history'):
            # 如果是Keras History对象
            self.history_dict = history.history
        else:
            self.history_dict = None
    
    def load_history_from_file(self, history_path: str) -> bool:
        """
        从文件加载训练历史
        
        Args:
            history_path: 历史文件路径
            
        Returns:
            bool: 加载成功返回True
        """
        try:
            with open(history_path, 'rb') as f:
                data = pickle.load(f)
            
            if isinstance(data, dict) and 'history' in data:
                self.history_dict = data['history']
            else:
                self.history_dict = data
            
            print(f"✅ 训练历史加载成功: {history_path}")
            return True
            
        except Exception as e:
            print(f"❌ 加载训练历史失败: {str(e)}")
            return False
    
    def plot_training_history(self, save_path: str = None, 
                            title_suffix: str = "", figsize: Tuple[int, int] = (20, 15)) -> None:
        """
        绘制完整的训练历史图表
        
        Args:
            save_path: 保存路径
            title_suffix: 标题后缀
            figsize: 图形大小
        """
        if not self.history_dict:
            print("❌ 没有可用的训练历史数据")
            return
        
        # 创建综合图表
        fig = plt.figure(figsize=figsize)
        fig.suptitle(f'COVID-19 分类模型训练结果 {title_suffix}', fontsize=16, fontweight='bold')
        
        epochs_range = range(1, len(self.history_dict['accuracy']) + 1)
        
        # 1. 准确率图表
        plt.subplot(3, 3, 1)
        plt.plot(epochs_range, self.history_dict['accuracy'], 'b-', 
                label='训练准确率', linewidth=2, marker='o', markersize=3)
        plt.plot(epochs_range, self.history_dict['val_accuracy'], 'r-', 
                label='验证准确率', linewidth=2, marker='s', markersize=3)
        plt.title('模型准确率', fontsize=14, fontweight='bold')
        plt.xlabel('轮次')
        plt.ylabel('准确率')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. 损失图表
        plt.subplot(3, 3, 2)
        plt.plot(epochs_range, self.history_dict['loss'], 'b-', 
                label='训练损失', linewidth=2, marker='o', markersize=3)
        plt.plot(epochs_range, self.history_dict['val_loss'], 'r-', 
                label='验证损失', linewidth=2, marker='s', markersize=3)
        plt.title('模型损失', fontsize=14, fontweight='bold')
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. 精确度图表（如果有）
        if 'precision' in self.history_dict:
            plt.subplot(3, 3, 3)
            plt.plot(epochs_range, self.history_dict['precision'], 'g-', 
                    label='训练精确度', linewidth=2, marker='o', markersize=3)
            if 'val_precision' in self.history_dict:
                plt.plot(epochs_range, self.history_dict['val_precision'], 'orange', 
                        label='验证精确度', linewidth=2, marker='s', markersize=3)
            plt.title('模型精确度', fontsize=14, fontweight='bold')
            plt.xlabel('轮次')
            plt.ylabel('精确度')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 4. 召回率图表（如果有）
        if 'recall' in self.history_dict:
            plt.subplot(3, 3, 4)
            plt.plot(epochs_range, self.history_dict['recall'], 'purple', 
                    label='训练召回率', linewidth=2, marker='o', markersize=3)
            if 'val_recall' in self.history_dict:
                plt.plot(epochs_range, self.history_dict['val_recall'], 'brown', 
                        label='验证召回率', linewidth=2, marker='s', markersize=3)
            plt.title('模型召回率', fontsize=14, fontweight='bold')
            plt.xlabel('轮次')
            plt.ylabel('召回率')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 5. 学习率图表（简化表示）
        plt.subplot(3, 3, 5)
        # 由于实际LR变化需要回调记录，这里显示初始学习率
        plt.axhline(y=self.config.LEARNING_RATE, color='k', linestyle='--', label='初始学习率')
        plt.title('学习率调度', fontsize=14, fontweight='bold')
        plt.xlabel('轮次')
        plt.ylabel('学习率')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # 6. 最终性能对比
        plt.subplot(3, 3, 6)
        final_train_acc = self.history_dict['accuracy'][-1]
        final_val_acc = self.history_dict['val_accuracy'][-1]
        best_val_acc = max(self.history_dict['val_accuracy'])
        
        metrics = ['最终训练', '最终验证', '最佳验证']
        values = [final_train_acc, final_val_acc, best_val_acc]
        colors = ['blue', 'red', 'green']
        
        bars = plt.bar(metrics, values, color=colors, alpha=0.7)
        plt.title('准确率对比', fontsize=14, fontweight='bold')
        plt.ylabel('准确率')
        plt.ylim(0, 1)
        
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, value + 0.01, 
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 7. 损失对比
        plt.subplot(3, 3, 7)
        final_train_loss = self.history_dict['loss'][-1]
        final_val_loss = self.history_dict['val_loss'][-1]
        min_val_loss = min(self.history_dict['val_loss'])
        
        loss_metrics = ['最终训练', '最终验证', '最小验证']
        loss_values = [final_train_loss, final_val_loss, min_val_loss]
        
        bars = plt.bar(loss_metrics, loss_values, color=colors, alpha=0.7)
        plt.title('损失对比', fontsize=14, fontweight='bold')
        plt.ylabel('损失')
        
        for bar, value in zip(bars, loss_values):
            plt.text(bar.get_x() + bar.get_width()/2, value + max(loss_values) * 0.02, 
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 8. 平滑趋势图
        plt.subplot(3, 3, 8)
        window_size = max(1, len(epochs_range) // 10)
        
        def moving_average(data, window_size):
            return [sum(data[max(0, i-window_size):i+1]) / min(i+1, window_size) 
                   for i in range(len(data))]
        
        smooth_train_acc = moving_average(self.history_dict['accuracy'], window_size)
        smooth_val_acc = moving_average(self.history_dict['val_accuracy'], window_size)
        
        plt.plot(epochs_range, smooth_train_acc, 'b-', label='平滑训练准确率', linewidth=2)
        plt.plot(epochs_range, smooth_val_acc, 'r-', label='平滑验证准确率', linewidth=2)
        plt.title('平滑准确率趋势', fontsize=14, fontweight='bold')
        plt.xlabel('轮次')
        plt.ylabel('准确率')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 9. 过拟合分析
        plt.subplot(3, 3, 9)
        acc_diff = [train - val for train, val in zip(
            self.history_dict['accuracy'], self.history_dict['val_accuracy'])]
        plt.plot(epochs_range, acc_diff, 'purple', linewidth=2, label='训练-验证准确率差')
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.axhline(y=0.1, color='red', linestyle=':', alpha=0.7, label='过拟合警戒线')
        plt.title('过拟合分析', fontsize=14, fontweight='bold')
        plt.xlabel('轮次')
        plt.ylabel('准确率差异')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 训练图表已保存: {save_path}")
        
        plt.show()
    
    def plot_metrics_comparison(self, save_path: str = None) -> None:
        """绘制指标对比图"""
        if not self.history_dict:
            print("❌ 没有可用的训练历史数据")
            return
        
        # 准备数据
        metrics_data = {}
        for key in self.history_dict:
            if not key.startswith('val_'):
                val_key = f'val_{key}'
                if val_key in self.history_dict:
                    metrics_data[key] = {
                        'train': self.history_dict[key][-1],
                        'val': self.history_dict[val_key][-1]
                    }
        
        if not metrics_data:
            print("❌ 没有可用的指标数据")
            return
        
        # 创建对比图
        fig, axes = plt.subplots(1, len(metrics_data), figsize=(5*len(metrics_data), 6))
        if len(metrics_data) == 1:
            axes = [axes]
        
        for i, (metric, values) in enumerate(metrics_data.items()):
            ax = axes[i]
            
            x = ['训练', '验证']
            y = [values['train'], values['val']]
            colors = ['skyblue', 'lightcoral']
            
            bars = ax.bar(x, y, color=colors, alpha=0.8)
            ax.set_title(f'{metric.title()}', fontsize=14, fontweight='bold')
            ax.set_ylabel(metric.title())
            
            # 添加数值标签
            for bar, value in zip(bars, y):
                ax.text(bar.get_x() + bar.get_width()/2, value + max(y) * 0.01,
                       f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
            
            # 设置y轴范围
            if metric == 'accuracy':
                ax.set_ylim(0, 1)
            elif metric == 'loss':
                ax.set_ylim(0, max(y) * 1.2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 指标对比图已保存: {save_path}")
        
        plt.show()
    
    def get_training_summary(self) -> Dict[str, Any]:
        """获取训练摘要统计"""
        if not self.history_dict:
            return {}
        
        summary = {
            'epochs_completed': len(self.history_dict['accuracy']),
            'final_train_accuracy': self.history_dict['accuracy'][-1],
            'final_val_accuracy': self.history_dict['val_accuracy'][-1],
            'final_train_loss': self.history_dict['loss'][-1],
            'final_val_loss': self.history_dict['val_loss'][-1],
            'best_val_accuracy': max(self.history_dict['val_accuracy']),
            'best_val_accuracy_epoch': np.argmax(self.history_dict['val_accuracy']) + 1,
            'min_val_loss': min(self.history_dict['val_loss']),
            'min_val_loss_epoch': np.argmin(self.history_dict['val_loss']) + 1,
            'overfitting_score': self.history_dict['accuracy'][-1] - self.history_dict['val_accuracy'][-1]
        }
        
        # 添加额外指标（如果有）
        if 'precision' in self.history_dict:
            summary['final_precision'] = self.history_dict['precision'][-1]
        if 'recall' in self.history_dict:
            summary['final_recall'] = self.history_dict['recall'][-1]
        
        return summary


class PredictionVisualizer:
    """预测结果可视化器"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
    
    def plot_prediction_results(self, results: List[Dict[str, Any]], 
                              analysis: Dict[str, Any], save_path: str = None) -> None:
        """
        绘制预测结果综合图表
        
        Args:
            results: 预测结果列表
            analysis: 分析结果
            save_path: 保存路径
        """
        if not results:
            print("❌ 没有预测结果可视化")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('COVID-19 分类预测结果分析', fontsize=16, fontweight='bold')
        
        # 1. 置信度分布直方图
        confidences = [r['confidence'] for r in results]
        ax1.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(np.mean(confidences), color='red', linestyle='--', 
                   label=f'平均值: {np.mean(confidences):.3f}')
        ax1.set_title('预测置信度分布', fontweight='bold')
        ax1.set_xlabel('置信度')
        ax1.set_ylabel('频次')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 类别分布饼图
        class_counts = analysis['class_distribution']
        labels = [f'{k.upper()}\n({v} 张图像)' for k, v in class_counts.items()]
        colors = ['lightcoral', 'lightblue']
        
        wedges, texts, autotexts = ax2.pie(class_counts.values(), labels=labels, 
                                          autopct='%1.1f%%', startangle=90, colors=colors)
        ax2.set_title('类别分布', fontweight='bold')
        
        # 3. 置信度vs不确定性散点图
        confidences = [r['confidence'] for r in results]
        uncertainties = [r['uncertainty'] for r in results]
        colors_scatter = ['red' if r['prediction'] == 'yes' else 'blue' for r in results]
        
        ax3.scatter(confidences, uncertainties, c=colors_scatter, alpha=0.6, s=30)
        ax3.set_xlabel('置信度')
        ax3.set_ylabel('不确定性')
        ax3.set_title('置信度 vs 不确定性', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', label='COVID-19 阳性'),
                          Patch(facecolor='blue', label='COVID-19 阴性')]
        ax3.legend(handles=legend_elements)
        
        # 4. 置信度等级分布
        conf_levels = analysis['confidence_levels']
        level_names = ['极高\n(>0.95)', '高\n(0.9-0.95)', '中等\n(0.7-0.9)', '低\n(<0.7)']
        level_counts = [conf_levels['very_high'], conf_levels['high'], 
                       conf_levels['medium'], conf_levels['low']]
        colors_bar = ['darkgreen', 'green', 'orange', 'red']
        
        bars = ax4.bar(level_names, level_counts, color=colors_bar, alpha=0.7)
        ax4.set_title('置信度等级分布', fontweight='bold')
        ax4.set_ylabel('图像数量')
        
        for bar, count in zip(bars, level_counts):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(level_counts) * 0.01,
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 预测结果图表已保存: {save_path}")
        
        plt.show()
    
    def plot_confidence_analysis(self, results: List[Dict[str, Any]], 
                               save_path: str = None) -> None:
        """绘制详细的置信度分析图"""
        if not results:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('置信度详细分析', fontsize=16, fontweight='bold')
        
        # 按类别分组
        yes_results = [r for r in results if r['prediction'] == 'yes']
        no_results = [r for r in results if r['prediction'] == 'no']
        
        # 1. 按类别的置信度分布
        yes_conf = [r['confidence'] for r in yes_results]
        no_conf = [r['confidence'] for r in no_results]
        
        ax1.hist([yes_conf, no_conf], bins=15, alpha=0.7, 
                label=['COVID-19 阳性', 'COVID-19 阴性'], color=['red', 'blue'])
        ax1.set_title('按类别的置信度分布')
        ax1.set_xlabel('置信度')
        ax1.set_ylabel('频次')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 置信度箱线图
        conf_data = [yes_conf, no_conf]
        ax2.boxplot(conf_data, labels=['阳性', '阴性'])
        ax2.set_title('置信度箱线图对比')
        ax2.set_ylabel('置信度')
        ax2.grid(True, alpha=0.3)
        
        # 3. 不确定性分析
        yes_unc = [r['uncertainty'] for r in yes_results]
        no_unc = [r['uncertainty'] for r in no_results]
        
        ax3.hist([yes_unc, no_unc], bins=15, alpha=0.7,
                label=['COVID-19 阳性', 'COVID-19 阴性'], color=['red', 'blue'])
        ax3.set_title('按类别的不确定性分布')
        ax3.set_xlabel('不确定性')
        ax3.set_ylabel('频次')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 原始概率分布
        raw_probs = [r['raw_probability'] for r in results]
        ax4.hist(raw_probs, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax4.axvline(0.5, color='red', linestyle='--', label='决策边界 (0.5)')
        ax4.set_title('原始预测概率分布')
        ax4.set_xlabel('预测概率')
        ax4.set_ylabel('频次')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 置信度分析图已保存: {save_path}")
        
        plt.show()


class ModelComparator:
    """模型比较器"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
    
    def compare_training_histories(self, history_path_1: str, history_path_2: str,
                                 labels: List[str] = None, save_path: str = None) -> None:
        """
        比较两个模型的训练历史
        
        Args:
            history_path_1: 第一个模型的历史文件路径
            history_path_2: 第二个模型的历史文件路径
            labels: 模型标签列表
            save_path: 保存路径
        """
        # 加载历史数据
        try:
            with open(history_path_1, 'rb') as f:
                data1 = pickle.load(f)
            with open(history_path_2, 'rb') as f:
                data2 = pickle.load(f)
            
            hist1 = data1['history'] if isinstance(data1, dict) and 'history' in data1 else data1
            hist2 = data2['history'] if isinstance(data2, dict) and 'history' in data2 else data2
            
        except Exception as e:
            print(f"❌ 加载历史文件失败: {str(e)}")
            return
        
        if labels is None:
            labels = ['10轮训练', '50轮训练']
        
        # 创建比较图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('模型训练对比分析', fontsize=16, fontweight='bold')
        
        epochs1 = range(1, len(hist1['accuracy']) + 1)
        epochs2 = range(1, len(hist2['accuracy']) + 1)
        
        # 1. 训练准确率对比
        ax1.plot(epochs1, hist1['accuracy'], 'b-', label=f'{labels[0]} 训练', linewidth=2)
        ax1.plot(epochs1, hist1['val_accuracy'], 'b--', label=f'{labels[0]} 验证', linewidth=2)
        ax1.plot(epochs2, hist2['accuracy'], 'r-', label=f'{labels[1]} 训练', linewidth=2)
        ax1.plot(epochs2, hist2['val_accuracy'], 'r--', label=f'{labels[1]} 验证', linewidth=2)
        ax1.set_title('准确率对比')
        ax1.set_xlabel('轮次')
        ax1.set_ylabel('准确率')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 训练损失对比
        ax2.plot(epochs1, hist1['loss'], 'b-', label=f'{labels[0]} 训练', linewidth=2)
        ax2.plot(epochs1, hist1['val_loss'], 'b--', label=f'{labels[0]} 验证', linewidth=2)
        ax2.plot(epochs2, hist2['loss'], 'r-', label=f'{labels[1]} 训练', linewidth=2)
        ax2.plot(epochs2, hist2['val_loss'], 'r--', label=f'{labels[1]} 验证', linewidth=2)
        ax2.set_title('损失对比')
        ax2.set_xlabel('轮次')
        ax2.set_ylabel('损失')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 最终性能对比
        final_metrics = {
            labels[0]: {
                'train_acc': hist1['accuracy'][-1],
                'val_acc': hist1['val_accuracy'][-1],
                'train_loss': hist1['loss'][-1],
                'val_loss': hist1['val_loss'][-1]
            },
            labels[1]: {
                'train_acc': hist2['accuracy'][-1],
                'val_acc': hist2['val_accuracy'][-1],
                'train_loss': hist2['loss'][-1],
                'val_loss': hist2['val_loss'][-1]
            }
        }
        
        # 准确率对比柱状图
        x_pos = np.arange(2)
        train_accs = [final_metrics[labels[0]]['train_acc'], final_metrics[labels[1]]['train_acc']]
        val_accs = [final_metrics[labels[0]]['val_acc'], final_metrics[labels[1]]['val_acc']]
        
        width = 0.35
        ax3.bar(x_pos - width/2, train_accs, width, label='训练准确率', alpha=0.8)
        ax3.bar(x_pos + width/2, val_accs, width, label='验证准确率', alpha=0.8)
        ax3.set_title('最终准确率对比')
        ax3.set_ylabel('准确率')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(labels)
        ax3.legend()
        
        # 添加数值标签
        for i, (train_acc, val_acc) in enumerate(zip(train_accs, val_accs)):
            ax3.text(i - width/2, train_acc + 0.01, f'{train_acc:.3f}', 
                    ha='center', va='bottom', fontweight='bold')
            ax3.text(i + width/2, val_acc + 0.01, f'{val_acc:.3f}', 
                    ha='center', va='bottom', fontweight='bold')
        
        # 4. 收敛速度分析
        # 计算达到特定准确率阈值的轮次
        threshold = 0.8
        
        def find_convergence_epoch(accuracy_list, threshold):
            for i, acc in enumerate(accuracy_list):
                if acc >= threshold:
                    return i + 1
            return len(accuracy_list)
        
        conv1 = find_convergence_epoch(hist1['val_accuracy'], threshold)
        conv2 = find_convergence_epoch(hist2['val_accuracy'], threshold)
        
        convergence_data = [conv1, conv2]
        ax4.bar(labels, convergence_data, alpha=0.8, color=['blue', 'red'])
        ax4.set_title(f'收敛速度对比 (达到{threshold}准确率)')
        ax4.set_ylabel('所需轮次')
        
        for i, conv in enumerate(convergence_data):
            ax4.text(i, conv + max(convergence_data) * 0.01, str(conv), 
                    ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 模型对比图已保存: {save_path}")
        
        plt.show()
        
        # 打印详细对比报告
        self._print_comparison_report(final_metrics, labels)
    
    def _print_comparison_report(self, metrics: Dict[str, Dict], labels: List[str]) -> None:
        """打印详细的模型对比报告"""
        print("\n" + "="*60)
        print("模型性能对比报告")
        print("="*60)
        
        for label in labels:
            print(f"\n{label}:")
            print(f"  最终训练准确率: {metrics[label]['train_acc']:.4f}")
            print(f"  最终验证准确率: {metrics[label]['val_acc']:.4f}")
            print(f"  最终训练损失: {metrics[label]['train_loss']:.4f}")
            print(f"  最终验证损失: {metrics[label]['val_loss']:.4f}")
            
            # 过拟合分析
            overfitting = metrics[label]['train_acc'] - metrics[label]['val_acc']
            print(f"  过拟合程度: {overfitting:.4f}")
            
            if overfitting > 0.1:
                print("    ⚠️ 存在过拟合")
            elif overfitting < 0:
                print("    📈 验证性能更好")
            else:
                print("    ✅ 拟合良好")
        
        # 性能对比
        print(f"\n性能对比:")
        val_acc_diff = metrics[labels[1]]['val_acc'] - metrics[labels[0]]['val_acc']
        val_loss_diff = metrics[labels[0]]['val_loss'] - metrics[labels[1]]['val_loss']
        
        print(f"验证准确率差异: {val_acc_diff:+.4f} ({labels[1]} vs {labels[0]})")
        print(f"验证损失差异: {val_loss_diff:+.4f} ({labels[0]} vs {labels[1]})")
        
        if val_acc_diff > 0.02:
            print(f"✅ {labels[1]} 显著优于 {labels[0]}")
        elif val_acc_diff < -0.02:
            print(f"✅ {labels[0]} 显著优于 {labels[1]}")
        else:
            print("⚖️ 两个模型性能相近")
        
        print("="*60)


# 便捷函数
def visualize_training_results(history_path: str, save_dir: str = None, 
                              version: str = None) -> None:
    """
    可视化训练结果的便捷函数
    
    Args:
        history_path: 训练历史文件路径
        save_dir: 保存目录
        version: 版本标识
    """
    config = get_config()
    
    # 创建可视化器
    visualizer = TrainingVisualizer(config=config)
    
    # 加载历史
    if not visualizer.load_history_from_file(history_path):
        return
    
    # 设置保存路径
    if save_dir and version:
        save_path = Path(save_dir) / f"training_results_{version}.png"
    else:
        save_path = None
    
    # 绘制图表
    title_suffix = f"- {version}" if version else ""
    visualizer.plot_training_history(str(save_path) if save_path else None, title_suffix)
    
    # 打印摘要
    summary = visualizer.get_training_summary()
    if summary:
        print(f"\n📊 {version or '模型'} 训练摘要:")
        for key, value in summary.items():
            print(f"  {key}: {value}")


def visualize_prediction_results(results: List[Dict], analysis: Dict, 
                                save_dir: str = None, version: str = None) -> None:
    """
    可视化预测结果的便捷函数
    
    Args:
        results: 预测结果列表
        analysis: 分析结果
        save_dir: 保存目录
        version: 版本标识
    """
    # 创建可视化器
    visualizer = PredictionVisualizer()
    
    # 设置保存路径
    if save_dir and version:
        save_path = Path(save_dir) / f"prediction_results_{version}.png"
        conf_path = Path(save_dir) / f"confidence_analysis_{version}.png"
    else:
        save_path = None
        conf_path = None
    
    # 绘制图表
    visualizer.plot_prediction_results(results, analysis, str(save_path) if save_path else None)
    visualizer.plot_confidence_analysis(results, str(conf_path) if conf_path else None)


if __name__ == "__main__":
    # 测试可视化模块
    print("测试可视化模块...")
    
    config = get_config()
    
    # 创建模拟训练历史
    mock_history = {
        'accuracy': [0.6, 0.7, 0.8, 0.85, 0.9],
        'val_accuracy': [0.55, 0.65, 0.75, 0.8, 0.82],
        'loss': [0.8, 0.6, 0.4, 0.3, 0.2],
        'val_loss': [0.85, 0.7, 0.5, 0.4, 0.35]
    }
    
    # 测试训练可视化
    print("\n1. 测试训练历史可视化...")
    visualizer = TrainingVisualizer(mock_history, config)
    visualizer.plot_training_history(title_suffix="测试版本")
    
    # 测试摘要
    summary = visualizer.get_training_summary()
    print(f"\n训练摘要: {summary}")
    
    print("\n✅ 可视化模块测试完成!")