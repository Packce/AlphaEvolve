# 金融因子分析平台

基于遗传编程的量化因子挖掘与因子分析系统。

## 项目结构

```
computer_design/
├── src/
│   ├── api/                    # API服务层
│   │   ├── api_server.py      # FastAPI主服务
│   │   ├── single_factor_api.py    # 单因子分析API
│   │   ├── multi_factor_api.py     # 多因子分析API
│   │   └── unified_api.py     # 统一API接口
│   └── core/
│       ├── factor/            # 因子分析模块
│       │   ├── single_factor_analysis.py   # 单因子分析
│       │   └── multi_factor_analysis.py    # 多因子分析
│       └── genetic/           # 遗传编程模块
│           └── factor_mining.py   # 遗传编程因子挖掘
├── docs/                      # 文档
│   ├── API文档.md
│   └── API统一接口文档.md
└── requirements_api.txt       # 依赖列表
```

## 功能特性

### 单因子分析
- IC/IR分析
- 分位数收益分析
- 滚动IC计算
- 多空累计收益
- 收益分布可视化

### 多因子分析
- LightGBM模型
- ElasticNet模型
- SHAP特征重要性
- 残差分析
- 预测vs真实对比

### 遗传编程因子挖掘
- 防止未来函数泄漏的滚动窗口计算
- 安全的数值计算
- 多种遗传操作（交叉、变异、选择）
- GPU加速支持（可选）

## 快速开始

### 安装依赖

```bash
pip install -r requirements_api.txt
```

### 启动API服务

```bash
cd D:\PythonCode\AlphaEvovle\back_end
D:\软件\期魔方\coder\python3116\python.exe d:\PythonCode\AlphaEvovle\back_end\src\api\unified_api.py
```

服务启动后访问 `http://localhost:8000/docs` 查看API文档。

## 技术栈

- **后端**: FastAPI
- **数据分析**: pandas, numpy
- **机器学习**: LightGBM, scikit-learn, PyTorch
- **量化分析**: alphalens
- **GPU加速**: CuPy (可选)

## 文档

详细API文档请参考 `docs/` 目录。
