import numpy as np
import math
import operator
import re

class Mlp:
    """多层感知器(Multi-Layer Perceptron)的实现
    
    这个类实现了一个可配置的神经网络，支持多个隐藏层，
    可以使用不同的激活函数（sigmoid、ReLU、soft plus）
    """
    
    def __init__(self, init_nodes: int = 0, learning_rate: float = .2) -> None:
        """初始化神经网络
        
        参数:
            init_nodes (int): 输入层节点数，默认为0
            learning_rate (float): 学习率，默认为0.2
        """
        self.number_of_nodes = []  # 存储每层节点数的列表
        if init_nodes > 0:
            self.number_of_nodes.append(init_nodes)
        self.weights = []          # 存储每层权重的列表
        self.biases = []           # 存储每层偏置的列表
        self.functions = []        # 存储每层激活函数的列表
        self.learning_rate = learning_rate

    def add_layer(self, number_of_nodes: int, weights = None, bias = None, function="sigmoid"):
        """添加新的网络层
        
        参数:
            number_of_nodes (int): 该层的节点数
            weights (numpy.array): 该层的权重矩阵，默认为None（随机初始化）
            bias (numpy.array): 该层的偏置向量，默认为None（初始化为0）
            function (str): 激活函数类型，可选"sigmoid"、"soft_plus"、"relu"，默认为"sigmoid"
        """
        self.number_of_nodes.append(number_of_nodes)
        if not weights is None:
            self.weights.append(weights)
            self.functions.append(function)
        elif len(self.number_of_nodes) > 1:
            # 使用He初始化方法初始化权重
            self.weights.append(np.random.randn(self.number_of_nodes[-1], self.number_of_nodes[-2]) * 
                              np.sqrt(2 / (self.number_of_nodes[-1] + self.number_of_nodes[-2])))
            self.functions.append(function)

        if not bias is None:
            self.biases.append(bias)
        elif len(self.number_of_nodes) > 1:
            self.biases.append(np.random.uniform(0, 0, size=(number_of_nodes, 1)))
    
    def save(self, location):
        """保存神经网络模型到文件
        
        参数:
            location (str): 保存文件的路径
        """
        f = open(location, "w+")
        for i in self.number_of_nodes:
            f.write(str(i) + " ")
        f.write("\t")
        for i in self.functions:
            f.write(i + " ")
        f.write("\n")
        for i in self.weights:
            for j in i:
                for k in j:
                    f.write(str(k) + " ")
                f.write("\t")
            f.write("\n")
        for b in self.biases:
            for i in b:
                for k in i:
                    f.write(str(k) + " ")
            f.write("\n")
        f.close()

    @staticmethod
    def load(location):
        """从文件加载神经网络模型
        
        参数:
            location (str): 模型文件的路径
            
        返回:
            Mlp: 加载好的神经网络模型
        """
        f = open(location, "r")
        lines = f.readlines()
        f_l = lines[0].strip()
        f_l = re.split(r'\t+', f_l)
        number_of_nodes = np.vectorize(lambda x: int(x))( f_l[0].split() )
        functions = f_l[1].split()
        weigths = []
        for i in range(1, len(number_of_nodes)):
            m = lines[i].strip().split("\t")
            for j in range(len(m)):
                m[j] = m[j].split()
            m = np.vectorize(lambda x: float(x))(np.matrix(m))
            weigths.append(m)
        biases = []
        for i in range(len(number_of_nodes), len(lines)):
            b = lines[i].strip().split("\t")
            for j in range(len(b)):
                b[j] = b[j].split()
            b = np.vectorize(lambda x: float(x))(np.matrix(b).T)
            biases.append(b)
        nn = Mlp()
        for i in range(len(number_of_nodes)):
            if i > 0:
                nn.add_layer(number_of_nodes=number_of_nodes[i], weights=weigths[i-1], bias=biases[i-1], function=functions[i-1])
            else:
                nn.add_layer(number_of_nodes[i])
        return nn

    @staticmethod
    def soft_plus(x):
        """Soft plus激活函数: f(x) = ln(1 + e^x)
        
        参数:
            x (numpy.array): 输入数据
        """
        sp = np.vectorize(lambda y: math.log(1 + math.exp(y)))
        return sp(x)

    @staticmethod
    def relu(x):
        """ReLU激活函数: f(x) = max(0, x)
        
        参数:
            x (numpy.array): 输入数据
        """
        re = np.vectorize(lambda y: max(0, y))
        return re(x)

    @staticmethod
    def sigmoid(x):
        """Sigmoid激活函数: f(x) = 1 / (1 + e^(-x))
        
        使用数值稳定的实现方式，避免上溢和下溢
        
        参数:
            x (numpy.array): 输入数据
        """
        sig = np.vectorize(lambda y:  (1 - 1 / (1 + math.exp(y))) if y < 0 else  (1 / (1 + math.exp(-y))))
        return sig(x)
    
    @staticmethod
    def squash(x, function):
        if function == "linear":
            return x  # 线性激活
        elif function == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif function == "relu":
            return np.maximum(0, x)
        else:
            raise ValueError(f"Unsupported activation function: {function}")

    @staticmethod
    def derivative(x, function):
        if function == "linear":
            return np.ones_like(x)
        elif function == "sigmoid":
            return x * (1 - x)
        elif function == "relu":
            return (x > 0).astype(float)
        else:
            raise ValueError(f"Unsupported activation function: {function}")

    def feed_forward(self, inp):
        """前向传播
        
        参数:
            inp: 输入数据，形状为 (特征数, 样本数)
            
        返回:
            list: 包含每一层输出的列表
        """
        # 确保输入是numpy数组
        current_output = np.array(inp)
        outputs = [current_output]
        
        # 对每一层进行前向传播
        for i in range(len(self.weights)):
            # 计算当前层的输出
            z = np.dot(self.weights[i], current_output) + self.biases[i]
            current_output = self.squash(z, self.functions[i])
            outputs.append(current_output)
        
        return outputs

    def train(self, X_batch, y_batch):
        """训练模型一个批次
        
        Args:
            X_batch: 输入数据批次，形状为(特征数, 批次大小)
            y_batch: 目标通量值，形状为(通量数, 批次大小)
        """
        outputs = self.feed_forward(X_batch)
        error = outputs[-1] - y_batch
        
        # 反向传播
        deltas = [error * self.derivative(outputs[-1], self.functions[-1])]
        for i in range(len(self.weights)-1, 0, -1):
            deltas.append(np.dot(self.weights[i].T, deltas[-1]) * 
                         self.derivative(outputs[i], self.functions[i]))
        deltas.reverse()
        
        # 更新参数（添加L2正则化）
        l2_lambda = 1e-4  # L2正则化系数
        for i in range(len(self.weights)):
            grad = deltas[i].dot(outputs[i].T) / X_batch.shape[1]
            # 添加L2正则化项
            grad += l2_lambda * self.weights[i]
            self.weights[i] -= self.learning_rate * grad
            self.biases[i] -= self.learning_rate * np.mean(deltas[i], axis=1, keepdims=True)

    def predict(self, inp):
        output = self.feed_forward(inp)[-1]
        return output  # 直接返回输出层结果
