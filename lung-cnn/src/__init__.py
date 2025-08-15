"""
COVID-19 肺部CT图像分类项目 - 主包初始化
========================================

这个包提供了完整的COVID-19肺部CT图像分类解决方案，包括：
- 数据处理和增强
- 模型构建和训练  
- 预测和评估
- 结果可视化和分析

主要模块:
- config: 配置管理
- data_utils: 数据处理工具
- model_utils: 模型构建和训练
- prediction_utils: 预测和评估
- visualization: 可视化工具

"""

# 版本信息
__version__ = "1.0.0"
__author__ = "COVID-19 Classification Team"
__email__ = "your.email@example.com"
__description__ = "COVID-19 肺部CT图像智能分类系统"

# 导入核心组件
try:
    # 配置管理
    from .config import (
        Config, 
        get_config, 
        switch_version, 
        setup_environment
    )
    
    # 数据处理
    from .data_utils import (
        DataProcessor,
        DataStructureValidator,
        TestDataLoader,
        process_data,
        load_test_data
    )
    
    # 模型相关
    from .model_utils import (
        GPUManager,
        ModelBuilder,
        ModelTrainer,
        TrainingCallbacks,
        create_and_train_model
    )
    
    # 预测和评估
    from .prediction_utils import (
        ModelPredictor,
        ResultsAnalyzer,
        ResultsSaver,
        EnsemblePredictor,
        predict_and_analyze,
        predict_with_ensemble,
        quick_predict,
        compare_model_predictions
    )
    
    # 可视化
    from .visualization import (
        TrainingVisualizer,
        PredictionVisualizer,
        ModelComparator,
        visualize_training_results,
        visualize_prediction_results
    )
    
    # 标记导入成功
    _imports_successful = True
    
except ImportError as e:
    # 如果导入失败，记录错误但不中断
    import warnings
    warnings.warn(f"部分模块导入失败: {str(e)}", ImportWarning)
    _imports_successful = False


# 包级别的便捷函数
def get_version() -> str:
    """获取包版本"""
    return __version__


def get_package_info() -> dict:
    """获取包信息"""
    return {
        "name": "covid_classification",
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "imports_successful": _imports_successful
    }


def print_package_info():
    """打印包信息"""
    info = get_package_info()
    print("🦠 COVID-19 肺部CT图像分类系统")
    print("="*50)
    print(f"版本: {info['version']}")
    print(f"作者: {info['author']}")
    print(f"描述: {info['description']}")
    print(f"模块导入: {'✅ 成功' if info['imports_successful'] else '❌ 部分失败'}")
    print("="*50)


def check_dependencies():
    """检查依赖包"""
    required_packages = [
        'tensorflow',
        'numpy', 
        'opencv-python',
        'matplotlib',
        'scikit-learn',
        'tqdm',
        'pathlib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
            elif package == 'scikit-learn':
                import sklearn
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ 缺少以下依赖包:")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        print("\n请运行: pip install -r requirements.txt")
        return False
    else:
        print("✅ 所有依赖包已安装")
        return True


def quick_start_guide():
    """快速开始指南"""
    print("\n🚀 快速开始指南")
    print("="*50)
    print("1. 数据准备:")
    print("   将训练数据放入 data/train_covid19/")
    print("   将测试数据放入 data/test_healthcare/")
    print()
    print("2. 环境设置:")
    print("   from src import setup_environment")
    print("   setup_environment()")
    print()
    print("3. 训练模型:")
    print("   from src import create_and_train_model, process_data")
    print("   train_gen, val_gen = process_data('10_epochs')")
    print("   model, history = create_and_train_model(train_gen, val_gen)")
    print()
    print("4. 进行预测:")
    print("   from src import predict_and_analyze")
    print("   results, analysis = predict_and_analyze('models/model.h5')")
    print()
    print("5. 可视化结果:")
    print("   from src import visualize_training_results")
    print("   visualize_training_results('models/history.pkl')")
    print("="*50)


# 模块级别的工具函数
def create_project_structure():
    """创建项目目录结构"""
    try:
        config = get_config()
        config.create_directories()
        print("✅ 项目目录结构创建成功")
        return True
    except Exception as e:
        print(f"❌ 创建目录结构失败: {str(e)}")
        return False


def validate_project_setup():
    """验证项目设置"""
    print("🔍 验证项目设置...")
    
    checks = []
    
    # 检查依赖
    deps_ok = check_dependencies()
    checks.append(("依赖包", deps_ok))
    
    # 检查配置
    try:
        config = get_config()
        config_ok = True
    except Exception:
        config_ok = False
    checks.append(("配置模块", config_ok))
    
    # 检查数据路径
    try:
        config = get_config()
        data_ok = config.validate_data_paths()
    except Exception:
        data_ok = False
    checks.append(("数据路径", data_ok))
    
    # 检查GPU
    try:
        from .model_utils import GPUManager
        gpu_info = GPUManager.get_gpu_memory_info()
        gpu_ok = gpu_info.get("gpu_available", False)
    except Exception:
        gpu_ok = False
    checks.append(("GPU可用性", gpu_ok))
    
    # 打印检查结果
    print("\n📋 检查结果:")
    for check_name, status in checks:
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {check_name}")
    
    all_critical_ok = checks[0][1] and checks[1][1]  # 依赖和配置是关键的
    
    if all_critical_ok:
        print("\n🎉 项目设置验证通过！可以开始使用。")
    else:
        print("\n⚠️ 项目设置存在问题，请检查上述失败项。")
    
    return all_critical_ok


# 预定义的工作流程
class WorkflowManager:
    """工作流程管理器"""
    
    @staticmethod
    def complete_training_workflow(version: str = "10_epochs"):
        """完整的训练工作流程"""
        print(f"🚀 开始完整训练工作流程 - {version}")
        
        try:
            # 1. 环境设置
            if not setup_environment():
                raise RuntimeError("环境设置失败")
            
            # 2. 数据处理
            train_gen, val_gen = process_data(version)
            
            # 3. 模型训练
            model, history = create_and_train_model(train_gen, val_gen, version)
            
            # 4. 结果可视化
            config = get_config()
            history_path = config.get_file_paths(version)["history"]
            visualize_training_results(str(history_path), 
                                     str(config.PLOTS_PATH), version)
            
            print(f"✅ {version} 训练工作流程完成")
            return True
            
        except Exception as e:
            print(f"❌ 训练工作流程失败: {str(e)}")
            return False
    
    @staticmethod
    def complete_prediction_workflow(model_path: str, version: str = None):
        """完整的预测工作流程"""
        print("🔍 开始完整预测工作流程")
        
        try:
            # 执行预测和分析
            results, analysis = predict_and_analyze(model_path, save_results=True, version=version)
            
            # 可视化预测结果
            config = get_config()
            visualize_prediction_results(results, analysis, 
                                       str(config.PLOTS_PATH), version)
            
            print("✅ 预测工作流程完成")
            return results, analysis
            
        except Exception as e:
            print(f"❌ 预测工作流程失败: {str(e)}")
            return None, None


# 包初始化时的操作
def _initialize_package():
    """包初始化操作"""
    if _imports_successful:
        # 尝试创建基本目录结构（静默模式）
        try:
            config = get_config()
            config.create_directories()
        except Exception:
            pass  # 静默处理，不影响包的导入


# 执行初始化
_initialize_package()


# 公开的API
__all__ = [
    # 版本和信息
    '__version__',
    'get_version',
    'get_package_info',
    'print_package_info',
    
    # 设置和验证
    'check_dependencies',
    'validate_project_setup',
    'quick_start_guide',
    'create_project_structure',
    
    # 核心组件 (如果导入成功)
    'Config', 'get_config', 'switch_version', 'setup_environment',
    'DataProcessor', 'process_data', 'load_test_data',
    'ModelBuilder', 'ModelTrainer', 'create_and_train_model',
    'ModelPredictor', 'predict_and_analyze', 'quick_predict',
    'TrainingVisualizer', 'visualize_training_results',
    
    # 工作流程
    'WorkflowManager'
]

# 条件性添加成功导入的组件
if _imports_successful:
    __all__.extend([
        'DataStructureValidator', 'TestDataLoader',
        'GPUManager', 'TrainingCallbacks',
        'ResultsAnalyzer', 'ResultsSaver', 'EnsemblePredictor',
        'predict_with_ensemble', 'compare_model_predictions',
        'PredictionVisualizer', 'ModelComparator',
        'visualize_prediction_results'
    ])


# 用户友好的入口点
if __name__ == "__main__":
    print_package_info()
    quick_start_guide() 
