import pandas as pd
import numpy as np
from sklearn.base import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from mlp import Mlp
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# 数据加载与预处理
def load_and_preprocess_data(bulk_data_path, pfba_flux_path):
    """加载并预处理数据
    
    Args:
        bulk_data_path: Bulk数据文件路径（如基因表达、代谢物浓度等）
        pfba_flux_path: PFBA生成的通量数据文件路径
    """
    # 加载数据
    bulk_data = pd.read_csv(bulk_data_path, index_col=0)
    pfba_fluxes = pd.read_csv(pfba_flux_path, index_col=0)
    
    # 数据预处理
    X = bulk_data.values.T  # (特征数, 样本数)
    y = pfba_fluxes.values.T  # (通量数, 样本数)
    
    # 标准化输入数据
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X.T).T
    
    # 通量数据归一化（确保在[0,1]范围内）
    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y.T).T
    
    return X_scaled, y_scaled, scaler_X, scaler_y, bulk_data.index, pfba_fluxes.index

# 模型评估函数
def evaluate_model(model, X, y, scaler_y, reaction_names=None):
    """评估模型性能
    
    Args:
        model: 训练好的MLP模型
        X: 输入数据
        y: 真实通量值
        scaler_y: 用于反归一化的scaler
        reaction_names: 反应名称列表
    """
    pred = model.predict(X)
    
    # 反归一化预测结果
    pred_original = scaler_y.inverse_transform(pred.T).T
    y_original = scaler_y.inverse_transform(y.T).T
    
    # 计算各项指标
    mse = np.mean((pred_original - y_original)**2, axis=1)
    mae = np.mean(np.abs(pred_original - y_original), axis=1)
    r2_scores = []
    pearson_corrs = []
    
    for i in range(y.shape[0]):
        r2 = r2_score(y_original[i], pred_original[i])
        r2_scores.append(r2)
        corr, _ = pearsonr(y_original[i], pred_original[i])
        pearson_corrs.append(corr)
        
        if reaction_names is not None:
            print(f"Reaction {reaction_names[i]}:")
            print(f"  MSE: {mse[i]:.4f}")
            print(f"  MAE: {mae[i]:.4f}")
            print(f"  R²: {r2:.4f}")
            print(f"  Pearson correlation: {corr:.4f}")
    
    return pred_original, y_original, mse, mae, r2_scores, pearson_corrs

# 主程序
if __name__ == "__main__":
    # 加载数据
    X_scaled, y_scaled, scaler_X, scaler_y, feature_names, reaction_names = load_and_preprocess_data(
        'bulk_data.csv',
        'pfba_fluxes.csv'
    )
    
    # 划分数据集
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled.T, y_scaled.T, test_size=0.3)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)
    
    # 转置回(特征数,样本数)格式
    X_train, X_val, X_test = X_train.T, X_val.T, X_test.T
    y_train, y_val, y_test = y_train.T, y_val.T, y_test.T
    
    # 创建模型
    input_dim = X_train.shape[0]
    output_dim = y_train.shape[0]
    
    model = Mlp(learning_rate=1e-4)
    model.add_layer(input_dim)
    model.add_layer(256, function="relu")
    model.add_layer(256, function="relu")
    model.add_layer(output_dim, function="sigmoid")  # 使用sigmoid因为通量已归一化到[0,1]
    
    # 训练参数
    epochs = 2000
    batch_size = 32
    best_val_loss = float('inf')
    patience = 50
    no_improve = 0
    
    # 记录训练历史
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    # 训练循环
    for epoch in range(epochs):
        # 批次训练
        for i in range(0, X_train.shape[1], batch_size):
            batch_indices = slice(i, min(i + batch_size, X_train.shape[1]))
            X_batch = X_train[:, batch_indices]
            y_batch = y_train[:, batch_indices]
            model.train(X_batch, y_batch)
        
        # 计算训练集和验证集损失
        train_pred = model.predict(X_train)
        train_loss = np.mean((train_pred - y_train)**2)
        
        val_pred = model.predict(X_val)
        val_loss = np.mean((val_pred - y_val)**2)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}")
            print(f"  Training Loss: {train_loss:.4f}")
            print(f"  Validation Loss: {val_loss:.4f}")
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
        else:
            no_improve += 1
            
        if no_improve >= patience:
            print("Early stopping!")
            break
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.show()
    
    # 在测试集上评估模型
    pred_test, y_test_original, mse, mae, r2_scores, pearson_corrs = evaluate_model(
        model, X_test, y_test, scaler_y, reaction_names
    )
    
    # 可视化预测结果
    n_fluxes = min(5, output_dim)  # 展示前5个通量或全部（如果少于5个）
    fig, axes = plt.subplots(1, n_fluxes, figsize=(4*n_fluxes, 4))
    if n_fluxes == 1:
        axes = [axes]
    
    for i in range(n_fluxes):
        ax = axes[i]
        ax.scatter(y_test_original[i], pred_test[i], alpha=0.5)
        ax.plot([y_test_original[i].min(), y_test_original[i].max()],
                [y_test_original[i].min(), y_test_original[i].max()],
                'r--')
        ax.set_xlabel('True Flux')
        ax.set_ylabel('Predicted Flux')
        ax.set_title(f'{reaction_names[i]}\nR²={r2_scores[i]:.3f}')
    
    plt.tight_layout()
    plt.show() 