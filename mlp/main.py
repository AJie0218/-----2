# main.py
# 主程序入口

import torch
import argparse
from torch.utils.data import DataLoader
from config import CONFIG
from trainer import Trainer

def main():
    """主程序入口函数"""
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='训练代谢通量预测模型')
    parser.add_argument('--no-early-stopping', action='store_true', 
                        help='禁用早停机制，运行所有epoch')
    args = parser.parse_args()
    
    print("初始化训练器...")
    trainer = Trainer(CONFIG)
    
    print(f"开始训练模型，设备: {CONFIG['device']}")
    use_early_stopping = not args.no_early_stopping
    if not use_early_stopping:
        print("早停机制已禁用，将训练所有epoch")
    trainer.run_training(use_early_stopping=use_early_stopping)
    
    # 在测试集上评估最终性能
    print("在测试集上评估模型性能...")
    test_loader = DataLoader(trainer.test_set, batch_size=CONFIG["batch_size"])
    
    # 加载最佳模型
    trainer.model.load_state_dict(torch.load("best_model.pth"))
    final_loss = trainer.evaluate(test_loader)
    print(f"测试集损失: {final_loss:.4f}")
    
    print("训练完成！最佳模型已保存为 'best_model.pth'")

if __name__ == "__main__":
    main() 