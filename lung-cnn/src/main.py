"""
COVID-19 肺部CT图像分类项目 - 主程序入口
====================================

提供命令行界面和完整的工作流程管理
支持训练、预测、比较和可视化等所有功能

Usage:
    python src/main.py --mode train --version 10_epochs
    python src/main.py --mode predict --model models/model.h5
    python src/main.py --mode compare
    python src/main.py --all

"""

import sys
import os
import argparse
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    # 导入项目模块
    from config import get_config, setup_environment, switch_version
    from data_utils import process_data
    from model_utils import create_and_train_model, GPUManager
    from prediction_utils import predict_and_analyze, predict_with_ensemble, compare_model_predictions
    from visualization import visualize_training_results, visualize_prediction_results, ModelComparator
    
    # 导入包信息
    from __init__ import (
        get_package_info, print_package_info, validate_project_setup,
        WorkflowManager, quick_start_guide
    )
    
    IMPORTS_SUCCESSFUL = True
    
except ImportError as e:
    print(f"❌ 模块导入失败: {str(e)}")
    print("请确保所有依赖已正确安装: pip install -r requirements.txt")
    IMPORTS_SUCCESSFUL = False


class COVID19ClassificationCLI:
    """COVID-19分类系统命令行界面"""
    
    def __init__(self):
        self.config = None
        if IMPORTS_SUCCESSFUL:
            self.config = get_config()
    
    def setup_argument_parser(self) -> argparse.ArgumentParser:
        """设置命令行参数解析器"""
        parser = argparse.ArgumentParser(
            description='COVID-19 肺部CT图像分类系统',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
使用示例:
  训练模型:
    python src/main.py --mode train --version 10_epochs
    python src/main.py --mode train --version 50_epochs
  
  预测图像:
    python src/main.py --mode predict --version 10_epochs
    python src/main.py --mode predict --model models/custom_model.h5
  
  比较模型:
    python src/main.py --mode compare
    python src/main.py --mode compare --models model1.h5 model2.h5
  
  可视化结果:
    python src/main.py --mode visualize --history models/history.pkl
  
  运行完整流程:
    python src/main.py --all
    python src/main.py --all --versions 10_epochs 50_epochs
  
  项目信息:
    python src/main.py --info
    python src/main.py --validate
            """
        )
        
        # 主要模式参数
        parser.add_argument('--mode', 
                          choices=['train', 'predict', 'compare', 'visualize', 'info'], 
                          help='运行模式')
        
        # 版本参数
        parser.add_argument('--version', 
                          choices=['10_epochs', '50_epochs'],
                          default='10_epochs',
                          help='模型版本 (默认: 10_epochs)')
        
        parser.add_argument('--versions',
                          nargs='+',
                          choices=['10_epochs', '50_epochs'],
                          help='多个版本 (用于批量操作)')
        
        # 文件路径参数
        parser.add_argument('--model', 
                          type=str,
                          help='模型文件路径')
        
        parser.add_argument('--models',
                          nargs='+',
                          help='多个模型文件路径')
        
        parser.add_argument('--data',
                          type=str,
                          help='测试数据路径')
        
        parser.add_argument('--history',
                          type=str,
                          help='训练历史文件路径')
        
        # 特殊操作参数
        parser.add_argument('--all',
                          action='store_true',
                          help='运行完整流程 (训练+预测+比较)')
        
        parser.add_argument('--ensemble',
                          action='store_true',
                          help='使用集成预测')
        
        parser.add_argument('--gpu',
                          action='store_true',
                          help='检查GPU状态')
        
        # 信息和验证参数
        parser.add_argument('--info',
                          action='store_true',
                          help='显示项目信息')
        
        parser.add_argument('--validate',
                          action='store_true',
                          help='验证项目设置')
        
        parser.add_argument('--guide',
                          action='store_true',
                          help='显示快速开始指南')
        
        # 输出控制参数
        parser.add_argument('--save',
                          action='store_true',
                          default=True,
                          help='保存结果 (默认开启)')
        
        parser.add_argument('--no-save',
                          action='store_true',
                          help='不保存结果')
        
        parser.add_argument('--verbose',
                          action='store_true',
                          help='详细输出模式')
        
        parser.add_argument('--quiet',
                          action='store_true',
                          help='静默模式')
        
        return parser
    
    def run_training_mode(self, args) -> bool:
        """运行训练模式"""
        print("🚀 启动训练模式")
        print("="*60)
        
        version = args.version
        print(f"📊 训练版本: {version}")
        
        try:
            # 环境设置
            if not setup_environment():
                print("❌ 环境设置失败")
                return False
            
            # 切换版本
            switch_version(version)
            
            # 数据处理
            print("\n📂 数据处理阶段...")
            train_gen, val_gen = process_data(version)
            
            # 模型训练
            print("\n🤖 模型训练阶段...")
            model, history = create_and_train_model(train_gen, val_gen, version)
            
            if history is None:
                print("❌ 训练失败或被中断")
                return False
            
            # 可视化训练结果
            if not args.no_save:
                print("\n📊 生成训练可视化...")
                history_path = self.config.get_file_paths(version)["history"]
                visualize_training_results(str(history_path), 
                                         str(self.config.PLOTS_PATH), version)
            
            print(f"\n✅ {version} 训练完成!")
            return True
            
        except KeyboardInterrupt:
            print("\n⏹️ 训练被用户中断")
            return False
        except Exception as e:
            print(f"\n❌ 训练过程出错: {str(e)}")
            if args.verbose:
                traceback.print_exc()
            return False
    
    def run_prediction_mode(self, args) -> bool:
        """运行预测模式"""
        print("🔍 启动预测模式")
        print("="*60)
        
        # 确定模型路径
        if args.model:
            model_path = args.model
            version = None
        else:
            version = args.version
            switch_version(version)
            model_path = str(self.config.get_file_paths(version)["model"])
        
        # 检查模型是否存在
        if not Path(model_path).exists():
            print(f"❌ 模型文件不存在: {model_path}")
            return False
        
        print(f"🤖 使用模型: {model_path}")
        
        try:
            # 集成预测
            if args.ensemble:
                if args.models:
                    model_paths = args.models
                else:
                    # 使用默认的两个版本
                    model_paths = [
                        str(self.config.get_file_paths("10_epochs")["model"]),
                        str(self.config.get_file_paths("50_epochs")["model"])
                    ]
                
                print(f"🔗 集成预测模式，使用 {len(model_paths)} 个模型")
                results, analysis = predict_with_ensemble(
                    model_paths, args.data, not args.no_save
                )
            else:
                # 单模型预测
                results, analysis = predict_and_analyze(
                    model_path, args.data, not args.no_save, version
                )
            
            if not results:
                print("❌ 预测失败或无结果")
                return False
            
            # 可视化预测结果
            if not args.no_save:
                print("\n📊 生成预测可视化...")
                visualize_prediction_results(results, analysis, 
                                           str(self.config.PLOTS_PATH), version)
            
            print(f"\n✅ 预测完成! 处理了 {len(results)} 张图像")
            return True
            
        except Exception as e:
            print(f"\n❌ 预测过程出错: {str(e)}")
            if args.verbose:
                traceback.print_exc()
            return False
    
    def run_compare_mode(self, args) -> bool:
        """运行模型比较模式"""
        print("📊 启动模型比较模式")
        print("="*60)
        
        try:
            if args.models:
                # 比较指定的模型
                model_paths = args.models
                model_names = [f"模型{i+1}" for i in range(len(model_paths))]
                
                print(f"🔄 比较用户指定的 {len(model_paths)} 个模型")
                compare_model_predictions(model_paths, args.data, model_names)
                
            else:
                # 比较默认的两个版本
                print("🔄 比较10轮和50轮训练版本")
                
                # 检查历史文件
                hist_10 = self.config.get_file_paths("10_epochs")["history"]
                hist_50 = self.config.get_file_paths("50_epochs")["history"]
                
                if not hist_10.exists() or not hist_50.exists():
                    print("❌ 缺少训练历史文件，请先训练两个版本的模型")
                    print(f"需要的文件:")
                    print(f"  - {hist_10}")
                    print(f"  - {hist_50}")
                    return False
                
                # 创建比较可视化
                comparator = ModelComparator()
                comparator.compare_training_histories(
                    str(hist_10), str(hist_50),
                    labels=['10轮训练', '50轮训练'],
                    save_path=str(self.config.PLOTS_PATH / "model_comparison.png")
                )
                
                # 如果模型文件存在，也比较预测结果
                model_10 = self.config.get_file_paths("10_epochs")["model"]
                model_50 = self.config.get_file_paths("50_epochs")["model"]
                
                if model_10.exists() and model_50.exists():
                    print("\n🔄 比较预测结果...")
                    compare_model_predictions(
                        [str(model_10), str(model_50)], args.data,
                        ['10轮模型', '50轮模型']
                    )
            
            print("\n✅ 模型比较完成!")
            return True
            
        except Exception as e:
            print(f"\n❌ 比较过程出错: {str(e)}")
            if args.verbose:
                traceback.print_exc()
            return False
    
    def run_visualize_mode(self, args) -> bool:
        """运行可视化模式"""
        print("📊 启动可视化模式")
        print("="*60)
        
        try:
            if args.history:
                # 可视化指定的历史文件
                history_path = args.history
                if not Path(history_path).exists():
                    print(f"❌ 历史文件不存在: {history_path}")
                    return False
                
                print(f"📈 可视化训练历史: {history_path}")
                visualize_training_results(history_path, str(self.config.PLOTS_PATH))
                
            else:
                # 可视化当前版本的历史
                version = args.version
                history_path = self.config.get_file_paths(version)["history"]
                
                if not history_path.exists():
                    print(f"❌ {version} 版本的历史文件不存在: {history_path}")
                    return False
                
                print(f"📈 可视化 {version} 训练历史")
                visualize_training_results(str(history_path), 
                                         str(self.config.PLOTS_PATH), version)
            
            print("\n✅ 可视化完成!")
            return True
            
        except Exception as e:
            print(f"\n❌ 可视化过程出错: {str(e)}")
            if args.verbose:
                traceback.print_exc()
            return False
    
    def run_all_mode(self, args) -> bool:
        """运行完整流程模式"""
        print("🔄 启动完整流程模式")
        print("="*70)
        
        versions = args.versions or ['10_epochs', '50_epochs']
        print(f"📊 将训练和测试以下版本: {versions}")
        
        success_count = 0
        
        try:
            # 训练所有版本
            for version in versions:
                print(f"\n{'='*20} {version} 训练 {'='*20}")
                
                # 临时修改参数进行训练
                train_args = argparse.Namespace(**vars(args))
                train_args.version = version
                train_args.mode = 'train'
                
                if self.run_training_mode(train_args):
                    success_count += 1
                    
                    # 训练成功后立即进行预测
                    print(f"\n{'='*20} {version} 预测 {'='*20}")
                    pred_args = argparse.Namespace(**vars(args))
                    pred_args.version = version
                    pred_args.mode = 'predict'
                    pred_args.model = None  # 使用默认模型路径
                    
                    self.run_prediction_mode(pred_args)
                else:
                    print(f"❌ {version} 训练失败，跳过预测")
            
            # 如果有多个版本成功训练，进行比较
            if success_count >= 2:
                print(f"\n{'='*20} 模型比较 {'='*20}")
                comp_args = argparse.Namespace(**vars(args))
                comp_args.mode = 'compare'
                comp_args.models = None  # 使用默认比较
                
                self.run_compare_mode(comp_args)
            
            print(f"\n🎉 完整流程完成! 成功训练 {success_count}/{len(versions)} 个版本")
            return success_count > 0
            
        except KeyboardInterrupt:
            print("\n⏹️ 完整流程被用户中断")
            return False
        except Exception as e:
            print(f"\n❌ 完整流程出错: {str(e)}")
            if args.verbose:
                traceback.print_exc()
            return False
    
    def run_info_mode(self, args) -> bool:
        """运行信息模式"""
        # 显示项目信息
        print_package_info()
        
        if args.gpu:
            print("\n🔧 GPU信息:")
            try:
                gpu_info = GPUManager.get_gpu_memory_info()
                if gpu_info.get("gpu_available", False):
                    print(f"✅ 发现 {gpu_info['gpu_count']} 个GPU")
                    for i, name in enumerate(gpu_info.get("gpu_names", [])):
                        print(f"  GPU {i}: {name}")
                else:
                    print("❌ 未发现可用GPU")
            except Exception as e:
                print(f"❌ GPU检查失败: {str(e)}")
        
        return True
    
    def main(self) -> int:
        """主函数"""
        if not IMPORTS_SUCCESSFUL:
            print("❌ 无法启动：模块导入失败")
            return 1
        
        # 解析命令行参数
        parser = self.setup_argument_parser()
        args = parser.parse_args()
        
        # 设置输出级别
        if args.quiet:
            import logging
            logging.getLogger().setLevel(logging.ERROR)
        
        # 处理信息类请求
        if args.info:
            self.run_info_mode(args)
            return 0
        
        if args.validate:
            success = validate_project_setup()
            return 0 if success else 1
        
        if args.guide:
            quick_start_guide()
            return 0
        
        # 检查是否有操作模式
        if not args.mode and not args.all:
            print("❌ 请指定运行模式或使用 --all")
            parser.print_help()
            return 1
        
        # 运行对应模式
        try:
            success = False
            
            if args.all:
                success = self.run_all_mode(args)
            elif args.mode == 'train':
                success = self.run_training_mode(args)
            elif args.mode == 'predict':
                success = self.run_prediction_mode(args)
            elif args.mode == 'compare':
                success = self.run_compare_mode(args)
            elif args.mode == 'visualize':
                success = self.run_visualize_mode(args)
            elif args.mode == 'info':
                success = self.run_info_mode(args)
            else:
                print(f"❌ 未知模式: {args.mode}")
                return 1
            
            return 0 if success else 1
            
        except KeyboardInterrupt:
            print("\n⏹️ 程序被用户中断")
            return 130  # 标准的中断退出码
        except Exception as e:
            print(f"\n💥 程序执行出现未预期错误: {str(e)}")
            if args.verbose:
                traceback.print_exc()
            return 1


def interactive_mode():
    """交互模式"""
    print("🎮 COVID-19分类系统 - 交互模式")
    print("="*50)
    
    if not IMPORTS_SUCCESSFUL:
        print("❌ 模块导入失败，无法启动交互模式")
        return
    
    cli = COVID19ClassificationCLI()
    
    while True:
        print("\n📋 可用操作:")
        print("1. 训练模型 (10轮)")
        print("2. 训练模型 (50轮)")
        print("3. 预测图像")
        print("4. 模型比较")
        print("5. 可视化结果")
        print("6. 运行完整流程")
        print("7. 项目信息")
        print("8. 验证设置")
        print("0. 退出")
        
        try:
            choice = input("\n请选择操作 (0-8): ").strip()
            
            if choice == '0':
                print("👋 再见!")
                break
            elif choice == '1':
                # 10轮训练
                args = argparse.Namespace(
                    mode='train', version='10_epochs', no_save=False,
                    verbose=False, quiet=False
                )
                cli.run_training_mode(args)
            elif choice == '2':
                # 50轮训练
                args = argparse.Namespace(
                    mode='train', version='50_epochs', no_save=False,
                    verbose=False, quiet=False
                )
                cli.run_training_mode(args)
            elif choice == '3':
                # 预测
                print("选择预测模式:")
                print("1. 使用10轮模型")
                print("2. 使用50轮模型")
                print("3. 指定模型文件")
                print("4. 集成预测")
                
                pred_choice = input("请选择 (1-4): ").strip()
                
                if pred_choice == '1':
                    args = argparse.Namespace(
                        mode='predict', version='10_epochs', model=None,
                        data=None, ensemble=False, no_save=False,
                        verbose=False, quiet=False
                    )
                elif pred_choice == '2':
                    args = argparse.Namespace(
                        mode='predict', version='50_epochs', model=None,
                        data=None, ensemble=False, no_save=False,
                        verbose=False, quiet=False
                    )
                elif pred_choice == '3':
                    model_path = input("请输入模型文件路径: ").strip()
                    args = argparse.Namespace(
                        mode='predict', version='10_epochs', model=model_path,
                        data=None, ensemble=False, no_save=False,
                        verbose=False, quiet=False
                    )
                elif pred_choice == '4':
                    args = argparse.Namespace(
                        mode='predict', version='10_epochs', model=None,
                        data=None, ensemble=True, models=None, no_save=False,
                        verbose=False, quiet=False
                    )
                else:
                    print("❌ 无效选择")
                    continue
                
                cli.run_prediction_mode(args)
            elif choice == '4':
                # 模型比较
                args = argparse.Namespace(
                    mode='compare', models=None, data=None,
                    verbose=False, quiet=False
                )
                cli.run_compare_mode(args)
            elif choice == '5':
                # 可视化
                print("选择可视化类型:")
                print("1. 10轮训练历史")
                print("2. 50轮训练历史")
                print("3. 指定历史文件")
                
                vis_choice = input("请选择 (1-3): ").strip()
                
                if vis_choice == '1':
                    args = argparse.Namespace(
                        mode='visualize', version='10_epochs', history=None,
                        verbose=False, quiet=False
                    )
                elif vis_choice == '2':
                    args = argparse.Namespace(
                        mode='visualize', version='50_epochs', history=None,
                        verbose=False, quiet=False
                    )
                elif vis_choice == '3':
                    history_path = input("请输入历史文件路径: ").strip()
                    args = argparse.Namespace(
                        mode='visualize', version='10_epochs', history=history_path,
                        verbose=False, quiet=False
                    )
                else:
                    print("❌ 无效选择")
                    continue
                
                cli.run_visualize_mode(args)
            elif choice == '6':
                # 完整流程
                print("选择要训练的版本:")
                print("1. 仅10轮")
                print("2. 仅50轮")
                print("3. 两个版本")
                
                all_choice = input("请选择 (1-3): ").strip()
                
                if all_choice == '1':
                    versions = ['10_epochs']
                elif all_choice == '2':
                    versions = ['50_epochs']
                elif all_choice == '3':
                    versions = ['10_epochs', '50_epochs']
                else:
                    print("❌ 无效选择")
                    continue
                
                args = argparse.Namespace(
                    all=True, versions=versions, no_save=False,
                    verbose=False, quiet=False, data=None, ensemble=False
                )
                cli.run_all_mode(args)
            elif choice == '7':
                # 项目信息
                args = argparse.Namespace(gpu=True)
                cli.run_info_mode(args)
            elif choice == '8':
                # 验证设置
                validate_project_setup()
            else:
                print("❌ 无效选择，请输入0-8之间的数字")
                
        except KeyboardInterrupt:
            print("\n⏹️ 操作被中断")
            continue
        except EOFError:
            print("\n👋 再见!")
            break
        except Exception as e:
            print(f"❌ 操作执行出错: {str(e)}")
            continue


def quick_demo():
    """快速演示模式"""
    print("🎬 COVID-19分类系统 - 快速演示")
    print("="*50)
    
    if not IMPORTS_SUCCESSFUL:
        print("❌ 模块导入失败，无法运行演示")
        return
    
    try:
        # 1. 显示项目信息
        print("\n1️⃣ 项目信息:")
        print_package_info()
        
        # 2. 验证设置
        print("\n2️⃣ 验证项目设置:")
        setup_ok = validate_project_setup()
        
        if not setup_ok:
            print("❌ 项目设置存在问题，演示可能无法正常运行")
            return
        
        # 3. 检查数据
        print("\n3️⃣ 检查数据:")
        config = get_config()
        if config.TRAIN_PATH.exists():
            print(f"✅ 训练数据路径存在: {config.TRAIN_PATH}")
        else:
            print(f"❌ 训练数据路径不存在: {config.TRAIN_PATH}")
        
        if config.TEST_PATH.exists():
            print(f"✅ 测试数据路径存在: {config.TEST_PATH}")
        else:
            print(f"❌ 测试数据路径不存在: {config.TEST_PATH}")
        
        # 4. 检查已有模型
        print("\n4️⃣ 检查已训练模型:")
        for version in ['10_epochs', '50_epochs']:
            model_path = config.get_file_paths(version)["model"]
            if model_path.exists():
                print(f"✅ {version} 模型存在: {model_path}")
            else:
                print(f"❌ {version} 模型不存在: {model_path}")
        
        # 5. 演示命令
        print("\n5️⃣ 可用命令演示:")
        print("训练命令示例:")
        print("  python src/main.py --mode train --version 10_epochs")
        print("  python src/main.py --mode train --version 50_epochs")
        print("\n预测命令示例:")
        print("  python src/main.py --mode predict --version 10_epochs")
        print("  python src/main.py --mode predict --ensemble")
        print("\n比较命令示例:")
        print("  python src/main.py --mode compare")
        print("\n完整流程命令:")
        print("  python src/main.py --all")
        
        print("\n✅ 演示完成!")
        print("使用 'python src/main.py --guide' 查看详细指南")
        
    except Exception as e:
        print(f"❌ 演示过程出错: {str(e)}")


def main():
    """程序入口点"""
    # 检查是否有命令行参数
    if len(sys.argv) == 1:
        # 没有参数，显示帮助信息
        print("🦠 COVID-19 肺部CT图像分类系统")
        print("="*50)
        print("使用方式:")
        print("  python src/main.py --help          # 显示帮助")
        print("  python src/main.py --info          # 显示项目信息")
        print("  python src/main.py --guide         # 显示快速指南")
        print("  python src/main.py --validate      # 验证项目设置")
        print("  python src/main.py --demo          # 运行演示")
        print("  python src/main.py --interactive   # 交互模式")
        print("\n快速开始:")
        print("  python src/main.py --all           # 运行完整流程")
        print("  python src/main.py --mode train --version 10_epochs")
        print("  python src/main.py --mode predict --version 10_epochs")
        return 0
    
    # 检查特殊参数
    if '--demo' in sys.argv:
        quick_demo()
        return 0
    
    if '--interactive' in sys.argv:
        interactive_mode()
        return 0
    
    # 运行CLI
    cli = COVID19ClassificationCLI()
    return cli.main()


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️ 程序被用户中断")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 程序出现未处理的错误: {str(e)}")
        print("请检查您的输入和环境配置")
        sys.exit(1)