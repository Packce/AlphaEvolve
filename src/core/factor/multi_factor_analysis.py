
# # 因子分析测试

# ## 导入库

import warnings
import asyncio
import nest_asyncio

try:
    from qmf_data import load_kline
    QMF_DATA_AVAILABLE = True
except ImportError:
    QMF_DATA_AVAILABLE = False
    load_kline = None

try:
    import qmf_model_sdk
    QMF_MODEL_SDK_AVAILABLE = True
except ImportError:
    QMF_MODEL_SDK_AVAILABLE = False
    qmf_model_sdk = None

try:
    import alphalens
    ALPHALENS_AVAILABLE = True
except ImportError:
    ALPHALENS_AVAILABLE = False
    alphalens = None

import datetime
from scipy.stats import rankdata as _rankdata
import numpy as np
import pandas as pd
import inspect
import re
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


nest_asyncio.apply()
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", message=".*Glyph.*")
warnings.filterwarnings("ignore", message=".*does not have a glyph.*")
warnings.filterwarnings("ignore", message=".*has no glyph.*")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans', 'FreeSans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.formatter.useoffset'] = False
plt.rcParams['axes.formatter.limits'] = (-3, 3)

COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3F88C5']
COMBINED_COLORS = ['#2E86AB', '#2E86AB', '#2E86AB', '#2E86AB', '#2E86AB']

# ## 需要调整的全局变量参数

# 使用转换器的因子表达式（多因子列表）
# formula = ["CORR(TS_ZSCORE(high, LV(high, 5)), EMV(85), open_interest)"]
formula = [
    "MIN(FORCAST(MIN(BOLL_UPPER(68, 8.56172617410534), open), 8), LOWRANGE(DPO(47, 36)))",
    "CORR(TS_ZSCORE(high, LV(high, 5)), EMV(85), open_interest)",
    "WR(2)",
    "HHVBARS(LLV(SMA(close, TS_RANK(ROC(33), BRAR_AR(92)), 5.0), 0.003), TAN(CCI(112)))"
]

# 是否使用转换器，不使用时需修改手动计算因子中的因子表达式
AUTO_ON = True

# 是否保存因子到当前目录的因子库中
SAVE_FACTOR = True

# ========== 多因子合成模型开关 ==========
# 是否使用 LightGBM 模型进行分析
USE_LIGHTGBM = True

# 是否使用 Elastic Net 模型进行分析
USE_ELASTIC_NET = True

# 是否使用 InstaSHAP 模型进行分析
USE_INSTASHAP = True

# ========== 板块与合约配置 ==========
# SELECTED_SECTOR 可选：
# - "all"
# - "农产品" / "金属" / "能源化工建材航运" / "金融"
# - ["农产品", "金属"]
# - ["时间分类", "有色金属"]
# - None（此时使用 MANUAL_SYMBOLS）
SELECTED_SECTOR = ["时间分类", "有色金属"]
# SELECTED_SECTOR = None

# 当 SELECTED_SECTOR 为 None 时使用
MANUAL_SYMBOLS = ["au888", "ag888"]

# ========== 时间范围 ==========
BEGIN_TIME = "2025-06-01"
END_TIME = "2025-08-31"
BEGIN_TIME_TEST = "2025-09-01"
END_TIME_TEST = "2025-12-31"
BEGIN_TIME_NOW = "2026-01-01"
END_TIME_NOW = datetime.datetime.now().strftime("%Y-%m-%d")

# ========== 数据周期 ==========
SYMBOL_CYCLE = "15分钟"
# SYMBOL_CYCLE = "30分钟"
# SYMBOL_CYCLE = "1天"

# 预测多少期后的收益
Y_PERIOD = 1

# ========== 适应度配置 ==========
IC_WEIGHT_A = 0.4
IC_WEIGHT_B = 0.6
FITNESS_W_TRAIN = 0.3
FITNESS_W_TEST = 0.7
FITNESS_SCHEME = "B"            # A: 加权平均；B: 加入过拟合惩罚
FITNESS_OVERFIT_LAMBDA = 0.2
MIN_CROSS_SECTION_COUNT = 2     # 单个时间截面最少有效合约数

# ========== 分层分析配置 ==========
QUANTILES = 5
PERIODS = (1, 3, 5)             # 对应 future_return / 分层收益等展示窗口
TRADING_DAYS_PER_YEAR = 252


# ## 选择并获取品种
# ========== 数据配置 ==========

# ========== 商品期货板块分类 ==========
FUTURES_SECTORS = {
    "农产品": {
        "油脂油料类": ["a", "b", "m", "y", "p", "OI", "PK", "RM", "RS"],
        "谷物类": ["c", "cs"],
        "软商品类": ["CF", "SR", "AP", "CY", "CJ"],
        "畜牧类": ["jd", "lh"],
    },
    "金属": {
        "有色金属": ["cu", "al", "zn", "pb", "ni", "sn", "bc", "ao", "lc", "si"],
        "贵金属": ["au", "ag"],
        "黑色金属": ["rb", "hc", "wr", "ss", "i", "jm", "j"],
    },
    "能源化工建材航运": {
        "能源类": ["sc", "fu", "bu", "lu", "pg"],
        "化工类": [
            "TA",
            "MA",
            "l",
            "pp",
            "v",
            "eg",
            "eb",
            "SA",
            "UR",
            "nr",
            "ru",
            "bz",
            "op",
            "PX",
            "PR",
            "PL",
            "ps",
            "SH",
            "PF",
            "sp",
        ],
        "建材类": ["fg", "fb", "lg"],
        "航运类": ["ec"],
    },
    "金融": {"股指期货": ["IC", "IF", "IH", "IM"], "国债期货": ["T", "TF", "TS", "TL"]},
    "时间分类":{
        # 日盘商品:无夜盘，标准日盘时间（09:00-15:00），主要为农产品和部分其他商品
        "日盘商品":['wr', 'AP', 'CJ', 'PK', 'RS', 'UR', 'fb', 'jd', 'lh', 'lg', 'ec',
         'lc', 'si', 'ps'],
        # 标准夜盘商品:夜盘21:00-23:00，品种最多，涵盖化工、农产品、黑色金属等
        "标准夜盘商品":['sp', 'ru', 'fu', 'hc', 'rb', 'op', 'CF', 'CY', 
         'fg', 'MA', 'OI', 'PF', 'PR', 'PL', 'PX', 'RM', 'SA', 'SH', 'SR', 'TA',
         'bz', 'a', 'b', 'c', 'cs', 'eb', 'eg', 'i', 'j', 'jm', 'l', 'm', 'p', 'pg',
         'pp','v', 'y', 'lu', 'nr'],
        # 有色金属:夜盘21:00-01:00（跨日），主要为有色金属品种
        "有色金属":['ao', 'ni', 'pb', 'sn', 'ss', 'zn', 'al', 'cu', 'bc'],
        # 贵金属与原油:夜盘21:00-02:30（跨日，最长）
        "贵金属与原油":['au', 'ag', 'sc'],
        # 股指期货:无夜盘，日盘时间09:30-15:00，股指期货品种
        "股指期货":["IC", "IF", "IH", "IM"],
        # 国债期货:无夜盘，日盘时间09:30-15:15，国债期货品种
        "国债期货":["T", "TF", "TS", "TL"],
        }
}


def get_symbols_by_sector(selected, sector_map, manual_symbols):
    if selected is None:
        return manual_symbols
    if selected == "all":
        return [f"{c}888" for s in sector_map.values() for cats in s.values() for c in cats]
    if isinstance(selected, str):
        if selected in sector_map:
            return [f"{c}888" for cats in sector_map[selected].values() for c in cats]
        print(f"警告：未知的板块选择 '{selected}'，使用手动指定的合约列表")
        return manual_symbols
    if isinstance(selected, list):
        # 指定 ["板块", "子板块"] 或多板块或混合（可嵌套）
        result = []
        all_valid = True
        def handle_item(item):
            nonlocal all_valid
            if isinstance(item, str):
                if item in sector_map:
                    for cats in sector_map[item].values():
                        result.extend([f"{c}888" for c in cats])
                else:
                    print(f"警告：未找到板块 '{item}'，跳过")
                    all_valid = False
            elif isinstance(item, list) and len(item) == 2:
                sec, cat = item
                if sec in sector_map and cat in sector_map[sec]:
                    result.extend([f"{c}888" for c in sector_map[sec][cat]])
                else:
                    print(f"警告：未找到板块 '{sec}' 的子板块 '{cat}'，跳过")
                    all_valid = False
            else:
                print(f"警告：无效的选择项 '{item}'，跳过")
                all_valid = False

        if len(selected) == 2 and all(isinstance(x, str) for x in selected):
            sec, cat = selected
            if sec in sector_map and cat in sector_map[sec]:
                result.extend([f"{c}888" for c in sector_map[sec][cat]])
                return result
        for item in selected:
            handle_item(item)
        if not all_valid:
            print("部分选择项无效，已使用有效的选择项生成合约列表")
        if result:
            return result
        return manual_symbols
    # fallback
    print("警告：列表格式不正确，使用手动指定的合约列表")
    return manual_symbols

SYMBOLS = get_symbols_by_sector(SELECTED_SECTOR, FUTURES_SECTORS, MANUAL_SYMBOLS)

print(f"当前使用的合约列表（共 {len(SYMBOLS)} 个合约）：")
print(SYMBOLS)


# ## 新的获取数据
# # 时间范围
# # BEGIN_TIME = "2020-01-01"
# # END_TIME = "2023-12-31"
# # BEGIN_TIME = "2025-06-01"
# # END_TIME = "2025-12-31"
# BEGIN_TIME = "2025-06-01"
# END_TIME = "2025-08-31"
# BEGIN_TIME_TEST = "2025-09-01"
# END_TIME_TEST = "2025-12-31"
# BEGIN_TIME_NOW = "2026-01-01"
# END_TIME_NOW = datetime.datetime.now().strftime("%Y-%m-%d")

# # 数据周期
# SYMBOL_CYCLE = "15分钟"  # 例如："1天", "1小时"等
# # SYMBOL_CYCLE = "1天"  # 例如："1天", "1小时"等
# # 构造Y值位移多少周期
# Y_PERIOD = 1

SYMBOL_CYCLE_MAP = {
    "1分钟": "1min",
    "3分钟": "3min",
    "5分钟": "5min",
    "10分钟": "10min",
    "15分钟": "15min",
    "30分钟": "30min",
    "1小时": "1H",
    "2小时": "2H",
    "4小时": "4H",
    "1天": "1D",
}

# 获取所有合约的数据，已注释掉
print("开始获取数据...")
# for symbol in SYMBOLS:
#     print(f"训练集：正在获取 {symbol} 的数据...")
#     asyncio.run(qmf_model_sdk.get_futures_data(symbol, BEGIN_TIME, END_TIME, SYMBOL_CYCLE))
# for symbol in SYMBOLS:
#     print(f"测试集：正在获取 {symbol} 的数据...")
#     asyncio.run(qmf_model_sdk.get_futures_data(symbol, BEGIN_TIME_TEST, END_TIME_TEST, SYMBOL_CYCLE))
# for symbol in SYMBOLS:
#     print(f"真实集：正在获取 {symbol} 的数据...")
#     asyncio.run(qmf_model_sdk.get_futures_data(symbol, BEGIN_TIME_NOW, END_TIME_NOW, SYMBOL_CYCLE))
print("数据获取完成！")

def get_futures_data(symbol, start_time=None, end_time=None):
    """获取单个合约的数据"""
    if load_kline is None:
        print(f"警告：qmf_data 不可用，{symbol} 无法获取数据")
        return pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume', 'open_interest', 'asset'])

    if start_time is None:
        start_time = f"{BEGIN_TIME} 00:00:00"
    elif len(str(start_time)) == 10:
        start_time = f"{start_time} 00:00:00"
    if end_time is None:
        end_time = f"{END_TIME} 23:59:59"
    elif len(str(end_time)) == 10:
        end_time = f"{end_time} 23:59:59"

    # data = load_kline(product=symbol, cycle="1D", start_time=start_time, end_time=end_time)
    # data = load_kline(product=symbol, cycle="15min", start_time=start_time, end_time=end_time)
    data = load_kline(product=symbol, cycle=SYMBOL_CYCLE_MAP[SYMBOL_CYCLE], start_time=start_time, end_time=end_time)

    # 检查数据是否为空或没有 'date' 列
    if data is None or len(data) == 0 or 'date' not in data.columns:
        print(f"警告：{symbol} 在指定时间范围内没有数据，跳过该合约")
        return pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume', 'open_interest', 'asset'])

    data["date"] = pd.to_datetime(data["date"]+data["time"], errors="coerce")
    # print(data)
    data.reset_index(inplace=True, drop=True)
    # print(data)
    data['asset'] = symbol
    return data

# 获取所有合约数据并合并
df_list = []
df_list_test = []
df_list_now = []
for symbol in SYMBOLS:
    data = get_futures_data(symbol)
    if len(data) > 0:
        df_list.append(data)
    data_test = get_futures_data(symbol, BEGIN_TIME_TEST, END_TIME_TEST)
    if len(data_test) > 0:
        df_list_test.append(data_test)
    data_now = get_futures_data(symbol, BEGIN_TIME_NOW, END_TIME_NOW)
    if len(data_now) > 0:
        df_list_now.append(data_now)

# 检查是否有有效数据
if len(df_list) == 0:
    if not QMF_DATA_AVAILABLE:
        print("警告：qmf_data 不可用，将使用空数据集继续")
    else:
        print("警告：所有合约在指定时间范围内都没有数据")

if len(df_list_test) == 0:
    if not QMF_DATA_AVAILABLE:
        print("警告：qmf_data 不可用，将使用空数据集继续")
    else:
        print("警告：所有合约在指定时间范围内都没有数据")

if len(df_list_now) == 0:
    if not QMF_DATA_AVAILABLE:
        print("警告：qmf_data 不可用，将使用空数据集继续")
    else:
        print("警告：所有合约在指定时间范围内都没有数据")

if df_list:
    df = pd.concat(df_list, ignore_index=True)
    df = df[['date', 'open', 'high', 'low', 'close', 'volume', 'open_interest', 'asset']]
else:
    df = pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume', 'open_interest', 'asset'])

if df_list_test:
    df_test = pd.concat(df_list_test, ignore_index=True)
    df_test = df_test[['date', 'open', 'high', 'low', 'close', 'volume', 'open_interest', 'asset']]
else:
    df_test = pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume', 'open_interest', 'asset'])

if df_list_now:
    df_now = pd.concat(df_list_now, ignore_index=True)
    df_now = df_now[['date', 'open', 'high', 'low', 'close', 'volume', 'open_interest', 'asset']]
else:
    df_now = pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume', 'open_interest', 'asset'])

df


# ## 数据预处理
# 数据预处理
df["open"] = pd.to_numeric(df["open"], errors="coerce")
df["high"] = pd.to_numeric(df["high"], errors="coerce")
df["low"] = pd.to_numeric(df["low"], errors="coerce")
df["close"] = pd.to_numeric(df["close"], errors="coerce")
df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
df["open_interest"] = pd.to_numeric(df["open_interest"], errors="coerce")

df_test["open"] = pd.to_numeric(df_test["open"], errors="coerce")
df_test["high"] = pd.to_numeric(df_test["high"], errors="coerce")
df_test["low"] = pd.to_numeric(df_test["low"], errors="coerce")
df_test["close"] = pd.to_numeric(df_test["close"], errors="coerce")
df_test["volume"] = pd.to_numeric(df_test["volume"], errors="coerce")
df_test["open_interest"] = pd.to_numeric(df_test["open_interest"], errors="coerce")

df_now["open"] = pd.to_numeric(df_now["open"], errors="coerce")
df_now["high"] = pd.to_numeric(df_now["high"], errors="coerce")
df_now["low"] = pd.to_numeric(df_now["low"], errors="coerce")
df_now["close"] = pd.to_numeric(df_now["close"], errors="coerce")
df_now["volume"] = pd.to_numeric(df_now["volume"], errors="coerce")
df_now["open_interest"] = pd.to_numeric(df_now["open_interest"], errors="coerce")

# 去重
df.drop_duplicates(subset=["asset", "date"], keep="last", inplace=True)
df.sort_values(["asset", "date"], ignore_index=True, inplace=True)

df_test.drop_duplicates(subset=["asset", "date"], keep="last", inplace=True)
df_test.sort_values(["asset", "date"], ignore_index=True, inplace=True)

df_now.drop_duplicates(subset=["asset", "date"], keep="last", inplace=True)
df_now.sort_values(["asset", "date"], ignore_index=True, inplace=True)

print(f"训练集数据预处理完成，共 {len(df)} 条记录")

print(f"测试集数据预处理完成，共 {len(df_test)} 条记录")

print(f"真实集数据预处理完成，共 {len(df_now)} 条记录")

# 计算未来5天收益（目标变量）
print("计算训练集目标变量...")
df = df.sort_values(["asset", "date"]).reset_index(drop=True)
df['future_return'] = df.groupby("asset")['close'].shift(-Y_PERIOD) / df['close'] - 1

print("计算测试集目标变量...")
df_test = df_test.sort_values(["asset", "date"]).reset_index(drop=True)
df_test['future_return'] = df_test.groupby("asset")['close'].shift(-Y_PERIOD) / df_test['close'] - 1

print("计算真实集目标变量...")
df_now = df_now.sort_values(["asset", "date"]).reset_index(drop=True)
df_now['future_return'] = df_now.groupby("asset")['close'].shift(-Y_PERIOD) / df_now['close'] - 1

# 删除目标变量为空的行
df = df.dropna(subset=['future_return']).reset_index(drop=True)
df_test = df_test.dropna(subset=['future_return']).reset_index(drop=True)
df_now = df_now.dropna(subset=['future_return']).reset_index(drop=True)


# 将列名适配：date -> time, asset -> code
df['time'] = df['date']
df['code'] = df['asset']
df_test['time'] = df_test['date']
df_test['code'] = df_test['asset']
df_now['time'] = df_now['date']
df_now['code'] = df_now['asset']

# 选择特征和目标（只保留基础行情数据，技术指标在遗传编程中通过函数调用生成）
features = ['open', 'close', 'high', 'low', 'volume', 'open_interest']
# features = []
target = 'future_return'

# 数据预处理
print(f"原始数据行数: {len(df)}")
data = df[features + [target, 'code', 'time']].copy()
data_test = df_test[features + [target, 'code', 'time']].copy()
data_now = df_now[features + [target, 'code', 'time']].copy()

# 构造训练集X, y（T, N）格式，T为时间，N为合约数
print("\n构造训练数据...")
pivoted = {}
all_cols = features + [target]
unique_cols = list(dict.fromkeys(all_cols))
if len(unique_cols) != len(all_cols):
    dup_cols = [c for c in set(all_cols) if all_cols.count(c) > 1]
    print("发现重复列，已去重: %s", dup_cols)
for col in unique_cols:
    try:
        pivoted[col] = data.pivot(index='time', columns='code', values=col).sort_index().sort_index(axis=1)
        print(f"特征 {col} 转换成功，形状: {pivoted[col].shape}")
    except Exception as e:
        print(f"特征 {col} 转换失败: {e}")
        # 创建全零矩阵作为替代
        unique_dates = pd.Series(data['time']).unique()
        unique_codes = pd.Series(data['code']).unique()
        pivoted[col] = pd.DataFrame(0, index=unique_dates, columns=unique_codes)

# 构造测试集X, y（T, N）格式，T为时间，N为合约数
print("\n构造测试数据...")
pivoted_test = {}
all_cols_test = features + [target]
unique_cols_test = list(dict.fromkeys(all_cols_test))
if len(unique_cols_test) != len(all_cols_test):
    dup_cols_test = [c for c in set(all_cols_test) if all_cols_test.count(c) > 1]
    print("发现重复列，已去重: %s", dup_cols_test)
for col in unique_cols_test:
    try:
        pivoted_test[col] = data_test.pivot(index='time', columns='code', values=col).sort_index().sort_index(axis=1)
        print(f"特征 {col} 转换成功，形状: {pivoted_test[col].shape}")
    except Exception as e:
        print(f"特征 {col} 转换失败: {e}")
        # 创建全零矩阵作为替代
        unique_dates_test = pd.Series(data_test['time']).unique()
        unique_codes_test = pd.Series(data_test['code']).unique()
        pivoted_test[col] = pd.DataFrame(0, index=unique_dates_test, columns=unique_codes_test)

# 构造真实集X, y（T, N）格式，T为时间，N为合约数
print("\n构造真实数据...")
pivoted_now = {}
all_cols_now = features + [target]
unique_cols_now = list(dict.fromkeys(all_cols_now))
if len(unique_cols_now) != len(all_cols_now):
    dup_cols_now = [c for c in set(all_cols_now) if all_cols_now.count(c) > 1]
    print("发现重复列，已去重: %s", dup_cols_now)
for col in unique_cols_now:
    try:
        pivoted_now[col] = data_now.pivot(index='time', columns='code', values=col).sort_index().sort_index(axis=1)
        print(f"特征 {col} 转换成功，形状: {pivoted_now[col].shape}")
    except Exception as e:
        print(f"特征 {col} 转换失败: {e}")
        # 创建全零矩阵作为替代
        unique_dates_now = pd.Series(data_now['time']).unique()
        unique_codes_now = pd.Series(data_now['code']).unique()
        pivoted_now[col] = pd.DataFrame(0, index=unique_dates_now, columns=unique_codes_now)

# 训练集
X_dict = {f: pivoted[f].values for f in features}
y = pivoted[target].values  # (T, N)
print("\n训练数据准备完成:")
print(f"时间点数量: {y.shape[0]}")
print(f"合约数量: {y.shape[1]}")
print(f"特征数量: {len(features)}")
print(f"特征列表: {features[:10]}..." if len(features) > 10 else f"特征列表: {features}")
# print({k: v.shape for k, v in X_dict.items()})

# 将X_dict整体保存为一个CSV文件
# 合并为多层列索引的DataFrame，索引为时间，列为MultiIndex(特征, 合约)
# multi_cols = pd.MultiIndex.from_product([features, pivoted[features[0]].columns], names=["feature", "code"])
# X_df = pd.DataFrame(
#     np.hstack([X_dict[f] for f in features]),
#     index=pivoted[features[0]].index,
#     columns=multi_cols
# )
# X_df.to_csv("train_X_dict_all.csv")
# print("训练集X_dict已保存为 train_X_dict_all.csv。")

# 测试集
X_dict_test = {f: pivoted_test[f].values for f in features}
y_test = pivoted_test[target].values  # (T, N)
print("\n真实数据准备完成:")
print(f"时间点数量: {y_test.shape[0]}")
print(f"合约数量: {y_test.shape[1]}")
print(f"特征数量: {len(features)}")
print(f"特征列表: {features[:10]}..." if len(features) > 10 else f"特征列表: {features}")

# 将X_dict_test整体保存为一个CSV文件
# multi_cols_test = pd.MultiIndex.from_product([features, pivoted_test[features[0]].columns], names=["feature", "code"])
# X_df_test = pd.DataFrame(
#     np.hstack([X_dict_test[f] for f in features]),
#     index=pivoted_test[features[0]].index,
#     columns=multi_cols_test
# )
# X_df_test.to_csv("test_X_dict_all.csv")
# print("测试集X_dict_test已保存为 test_X_dict_all.csv。")

# 真实集
X_dict_now = {f: pivoted_now[f].values for f in features}
y_now = pivoted_now[target].values  # (T, N)
print("\n测试数据准备完成:")
print(f"时间点数量: {y_now.shape[0]}")
print(f"合约数量: {y_now.shape[1]}")
print(f"特征数量: {len(features)}")
print(f"特征列表: {features[:10]}..." if len(features) > 10 else f"特征列表: {features}")

# 将X_dict_now整体保存为一个CSV文件
# multi_col_now = pd.MultiIndex.from_product([features, pivoted_now[features[0]].columns], names=["feature", "code"])
# X_df_now = pd.DataFrame(
#     np.hstack([X_dict_now[f] for f in features]),
#     index=pivoted_now[features[0]].index,
#     columns=multi_cols_now
# )
# X_df_now.to_csv("test_X_dict_all.csv")
# print("真实集X_dict_now已保存为 test_X_dict_all.csv。")


# 检查训练集数据质量
print("\n数据质量检查:")
print(f"y中NaN比例: {np.isnan(y).sum() / y.size:.2%}")
print(f"y中有效值数量: {np.sum(~np.isnan(y))}")
# 打印训练集最小时间和最大时间
train_times = pivoted[features[0]].index if features and features[0] in pivoted else None
if train_times is not None and len(train_times) > 0:
    print(f"训练集最小时间: {train_times.min()}, 最大时间: {train_times.max()}")

for f in features[:5]:  # 只检查前5个特征
    if f in X_dict:
        nan_ratio = np.isnan(X_dict[f]).sum() / X_dict[f].size
        print(f"  {f}: NaN比例={nan_ratio:.2%}")

# 检查测试集数据质量
print("\n测试集数据质量检查:")
print(f"y_test中NaN比例: {np.isnan(y_test).sum() / y_test.size:.2%}")
print(f"y_test中有效值数量: {np.sum(~np.isnan(y_test))}")
# 打印测试集最小时间和最大时间
test_times = pivoted_test[features[0]].index if features and features[0] in pivoted_test else None
if test_times is not None and len(test_times) > 0:
    print(f"测试集最小时间: {test_times.min()}, 最大时间: {test_times.max()}")

for f in features[:5]:  # 只检查前5个特征
    if f in X_dict_test:
        nan_ratio = np.isnan(X_dict_test[f]).sum() / X_dict_test[f].size
        print(f"  {f}: NaN比例={nan_ratio:.2%}")

# 检查真实集数据质量
print("\n真实集数据质量检查:")
print(f"y_now中NaN比例: {np.isnan(y_now).sum() / y_now.size:.2%}")
print(f"y_now中有效值数量: {np.sum(~np.isnan(y_now))}")
# 打印测试集最小时间和最大时间
now_times = pivoted_now[features[0]].index if features and features[0] in pivoted_now else None
if now_times is not None and len(now_times) > 0:
    print(f"真实集最小时间: {now_times.min()}, 最大时间: {now_times.max()}")

for f in features[:5]:  # 只检查前5个特征
    if f in X_dict_now:
        nan_ratio = np.isnan(X_dict_now[f]).sum() / X_dict_now[f].size
        print(f"  {f}: NaN比例={nan_ratio:.2%}")

X_TRAIN_SHAPE = X_dict["close"].shape
X_TEST_SHAPE = X_dict_test["close"].shape
X_NOW_SHAPE = X_dict_now["close"].shape

# print(y)




# ========= 定义InstaSHAP模型 =========
class Subnet(nn.Module):
    """单个特征子集 T 的贡献网络"""
    def __init__(self, in_features: int, hidden_dims: List[int] = [64, 32], 
                 output_dim: int = 1, activation: str = "relu"):
        super().__init__()
        layers = []
        prev_dim = in_features
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            else:
                layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class InstaSHAPGAM(nn.Module):
    """
    InstaSHAP-GAM-k 模型：可解释的加性模型，支持特征交互
    严格遵循论文 Equation 8 和 Equation 20 的实现
    
    Args:
        n_features: 输入特征维度 d
        k: 最大交互阶数（1 = 仅主效应，2 = 包含二阶交互）
        feature_names: 特征名称列表（可选）
        hidden_dims: 每个子网隐藏层维度
        activation: 激活函数类型
    """
    def __init__(
        self,
        n_features: int,
        k: int = 2,
        feature_names: Optional[List[str]] = None,
        hidden_dims: List[int] = [64, 32],
        activation: str = "relu"
    ):
        super().__init__()
        self.n_features = n_features
        self.k = min(k, n_features)
        self.feature_names = feature_names or [f"x{i}" for i in range(n_features)]
        
        # 生成所有 |T| <= k 的特征子集（仅前向传播时用于构造输出）
        self.interaction_sets = self._generate_interaction_sets()
        print(f"InstaSHAP-GAM-{k}: 共 {len(self.interaction_sets)} 个加性项")
        
        # 为每个子集 T 创建独立的子网络
        self.subnets = nn.ModuleDict()
        for i, T in enumerate(self.interaction_sets):
            subnet_name = self._subset_to_key(T)
            self.subnets[subnet_name] = Subnet(
                in_features=len(T),
                hidden_dims=hidden_dims,
                output_dim=1,
                activation=activation
            )
        
        # 偏置项（常数项 f_0）
        self.bias = nn.Parameter(torch.zeros(1))
        
    def _generate_interaction_sets(self) -> List[Tuple[int, ...]]:
        """生成所有 |T| <= k 的特征子集"""
        from itertools import combinations
        sets = []
        for order in range(1, self.k + 1):
            for combo in combinations(range(self.n_features), order):
                sets.append(combo)
        # 空集（常数项）单独处理
        return sets
    
    def _subset_to_key(self, subset: Tuple[int, ...]) -> str:
        return "_".join(map(str, subset))
    
    def _key_to_subset(self, key: str) -> Tuple[int, ...]:
        return tuple(map(int, key.split("_")))
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播（支持掩码训练模式）
        
        Args:
            x: 输入张量，shape (batch, d)
            mask: 掩码张量，shape (batch, d)，1=特征被保留，0=特征被掩码
                  若为 None，则正常前向传播
        
        Returns:
            y_pred: 预测值，shape (batch, 1) 或 (batch,)
        """
        batch_size = x.shape[0]
        output = self.bias.expand(batch_size)
        
        for T in self.interaction_sets:
            # Instant Mask 检查：仅当 T 中所有特征都被保留时才启用
            if mask is not None:
                enabled = torch.all(mask[:, list(T)], dim=1)  # (batch,)
                if not enabled.any():
                    continue
            else:
                enabled = torch.ones(batch_size, dtype=torch.bool)
            
            # 提取特征子集 x_T
            x_T = x[:, list(T)]  # (batch, |T|)
            
            # 计算贡献 phi_T(x_T)
            subnet = self.subnets[self._subset_to_key(T)]
            phi_T = subnet(x_T).squeeze(-1)  # (batch,)
            
            if mask is not None:
                output = output + torch.where(enabled, phi_T, torch.zeros_like(phi_T))
            else:
                output = output + phi_T
        
        return output
    
    def get_shapley_values(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算每个样本的 Shapley 值（Equation 13 风格）
        ϕ_i(x) = f_i(x_i) + Σ_{T⊇{i},|T|>1} f_T(x_T) / |T|
        
        Args:
            x: 输入张量，shape (batch, d)
        
        Returns:
            shapley: Shapley 值矩阵，shape (batch, d)
        """
        batch_size = x.shape[0]
        shapley = torch.zeros(batch_size, self.n_features, device=x.device)
        
        for T in self.interaction_sets:
            x_T = x[:, list(T)]
            subnet = self.subnets[self._subset_to_key(T)]
            phi_T = subnet(x_T).squeeze(-1)  # (batch,)
            
            # 将 f_T 均匀分配给 T 中的所有特征
            contribution = phi_T.unsqueeze(-1) / len(T)  # (batch, 1)
            
            for idx, feat_idx in enumerate(T):
                shapley[:, feat_idx] += contribution.squeeze()
        
        return shapley
    
    def get_shape_functions(self) -> Dict[str, callable]:
        """获取所有形状函数（用于可视化）"""
        shape_functions = {}
        model_device = next(self.parameters()).device
        
        def make_function(subnet, feature_indices):
            def func(x_T):
                with torch.no_grad():
                    x_tensor = torch.as_tensor(x_T, dtype=torch.float32, device=model_device)
                    if x_tensor.dim() == 1:
                        x_tensor = x_tensor.unsqueeze(0)
                    return subnet(x_tensor).squeeze().cpu().numpy()
            return func
        
        for T in self.interaction_sets:
            key = self._subset_to_key(T)
            shape_functions[f"f_{key}"] = make_function(self.subnets[key], T)
        
        return shape_functions

class ShapleyMaskSampler:
    """
    SHAP 核分布掩码采样器
    
    严格按照 SHAP 核分布 p(S) ∝ C(d,s)^(-1) * 1/(s*(d-s))
    实现掩码的批量采样
    """
    def __init__(self, n_features: int, device: torch.device = torch.device("cpu")):
        self.n_features = n_features
        self.device = device
        self._precompute_weights()
    
    def _precompute_weights(self):
        """预计算所有 2^d 种掩码的权重（实际可采样时动态计算）"""
        from math import comb
        weights = {}
        from itertools import product
        total_weight = 0
        
        for mask_tuple in product([0, 1], repeat=self.n_features):
            s = sum(mask_tuple)
            if s == 0 or s == self.n_features:
                # 极端情况特殊处理（避免分母为零）
                weight = 1e-6
            else:
                weight = 1.0 / (comb(self.n_features, s) * s * (self.n_features - s))
            weights[mask_tuple] = weight
            total_weight += weight
        
        # 归一化
        for mask_tuple in weights:
            weights[mask_tuple] /= total_weight
        
        self.masks = list(weights.keys())
        self.probs = np.array([weights[m] for m in self.masks])
    
    def sample(self, batch_size: int) -> torch.Tensor:
        """
        采样 batch_size 个掩码
        
        Returns:
            masks: shape (batch_size, n_features), dtype=torch.float32
        """
        indices = np.random.choice(len(self.masks), size=batch_size, p=self.probs)
        masks = np.array([self.masks[i] for i in indices], dtype=np.float32)
        return torch.tensor(masks, device=self.device)
    
    @staticmethod
    def sample_fast(n_features: int, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        高效近似采样（避免预计算 2^d 空间）
        采样流程：1) 从 p(s) ∝ 1/(s*(d-s)) 采样子集大小；2) 均匀选择具体特征
        """
        s_probs = np.array([1.0 / (s * (n_features - s)) if 1 <= s <= n_features - 1 else 0 
                            for s in range(n_features + 1)])
        s_probs = s_probs / s_probs.sum()
        
        masks = []
        for _ in range(batch_size):
            s = np.random.choice(n_features + 1, p=s_probs)
            if s == 0 or s == n_features:
                # 全0或全1掩码，概率极低
                mask = np.ones(n_features) if s == n_features else np.zeros(n_features)
            else:
                chosen = np.random.choice(n_features, size=s, replace=False)
                mask = np.zeros(n_features)
                mask[chosen] = 1
            masks.append(mask)
        
        return torch.tensor(np.array(masks, dtype=np.float32), device=device)

class InstaSHAPTrainer:
    """
    InstaSHAP 模型训练器
    
    严格遵循论文 Equation 20 的损失函数
    """
    def __init__(
        self,
        model: InstaSHAPGAM,
        blackbox_model: Optional[callable] = None,
        lr: float = 1e-3,
        device: torch.device = torch.device("cpu"),
        mask_sampler: Optional[ShapleyMaskSampler] = None
    ):
        self.model = model.to(device)
        self.blackbox_model = blackbox_model
        self.device = device
        self.lr = lr
        self.mask_sampler = mask_sampler or ShapleyMaskSampler(model.n_features, device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
    
    def _get_blackbox_prediction(
        self, 
        x: torch.Tensor, 
        mask: torch.Tensor,
        X_dict: Dict[str, np.ndarray],
        feature_indices: Dict[str, int]
    ) -> torch.Tensor:
        """
        获取 f(x;S)：仅保留掩码特征时的黑盒模型输出
        论文 Section 2.1 定义的移除方法
        
        这里需要根据你的数据格式实现具体的条件期望计算
        """
        batch_size = x.shape[0]
        predictions = torch.zeros(batch_size, device=self.device)
        
        if self.blackbox_model is None:
            raise ValueError("需要提供 blackbox_model 或自定义预测逻辑")
        
        # 简化实现：假设 blackbox_model 接收 (x, mask) 作为输入
        # 实际应用中应使用条件期望 M_p 来计算
        with torch.no_grad():
            predictions = self.blackbox_model(x, mask)
        
        return predictions
    
    def compute_instashap_loss(
        self,
        x: torch.Tensor,
        masks: torch.Tensor,
        f_masked: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算 InstaSHAP 损失
        
        L = E_x[ E_{S~p(S)}[ (f(x;S) - Σ_T 1(T⊆S) φ_T(x_T))^2 ] ]
        """
        batch_size = x.shape[0]
        total_loss = 0.0
        
        for i in range(batch_size):
            x_i = x[i:i+1]          # (1, d)
            mask_i = masks[i]       # (d,)
            
            # 1. 计算黑盒输出 f(x;S)
            if f_masked is not None:
                f_val = f_masked[i]
            elif self.blackbox_model is not None:
                f_val = self._get_blackbox_prediction(x_i, mask_i, None, None)
            else:
                raise ValueError("必须提供 f_masked 或 blackbox_model")
            
            # 2. 计算 GAM 预测（带掩码）
            y_pred = self.model(x_i, mask=mask_i.unsqueeze(0))  # (1,)
            
            # 3. MSE 损失
            loss = F.mse_loss(y_pred, f_val.reshape_as(y_pred))
            total_loss += loss
        
        return total_loss / batch_size
    
    def train_step(
        self,
        x_batch: torch.Tensor,
        f_masked_batch: Optional[torch.Tensor] = None,
        return_loss: bool = True
    ) -> float:
        """
        单步训练
        """
        batch_size = x_batch.shape[0]
        
        # 采样掩码（SHAP 核分布）
        masks = self.mask_sampler.sample_fast(self.model.n_features, batch_size, self.device)
        
        self.optimizer.zero_grad()
        
        if f_masked_batch is not None:
            loss = self.compute_instashap_loss(x_batch, masks, f_masked_batch)
        else:
            loss = self.compute_instashap_loss(x_batch, masks)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item() if return_loss else 0.0
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        val_freq: int = 10,
        early_stopping_patience: int = 20,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        完整的训练循环
        
        注意：论文中提到 InstaSHAP 的优势在于训练完成后，
        可在单次前向传播中计算 SHAP 值（get_shapley_values 方法）
        """
        dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32, device=self.device),
            torch.tensor(y_train, dtype=torch.float32, device=self.device)
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        history = {'train_loss': [], 'val_loss': [], 'val_ic': []}
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            epoch_losses = []
            
            for x_batch, y_batch in tqdm(dataloader, desc=f"Epoch {epoch+1}", disable=not verbose):
                loss = self.train_step(x_batch, f_masked_batch=y_batch)
                epoch_losses.append(loss)
            
            avg_train_loss = np.mean(epoch_losses)
            history['train_loss'].append(avg_train_loss)
            
            if X_val is not None and y_val is not None and (epoch + 1) % val_freq == 0:
                val_loss, val_ic = self.validate(X_val, y_val)
                history['val_loss'].append(val_loss)
                history['val_ic'].append(val_ic)
                self.scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
                
                if verbose:
                    print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.6f}, val_loss={val_loss:.6f}, val_ic={val_ic:.6f}")
            elif verbose:
                print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.6f}")
        
        return history
    
    def validate(self, X_val: np.ndarray, y_val: np.ndarray) -> Tuple[float, float]:
        """验证并计算损失和 IC"""
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(X_val, dtype=torch.float32, device=self.device)
            y_tensor = torch.tensor(y_val, dtype=torch.float32, device=self.device)
            
            # 正常前向传播（无掩码）
            y_pred = self.model(x_tensor).squeeze()
            val_loss = F.mse_loss(y_pred, y_tensor).item()
            
            # 计算 IC
            pred_np = y_pred.cpu().numpy()
            y_np = y_val.flatten()
            mask = ~(np.isnan(pred_np) | np.isnan(y_np))
            if mask.sum() > 2:
                from scipy.stats import rankdata
                ic = np.corrcoef(rankdata(pred_np[mask]), rankdata(y_np[mask]))[0, 1]
            else:
                ic = 0.0
            
        return val_loss, ic
    
    def explain(self, X: np.ndarray) -> np.ndarray:
        """
        单次前向传播计算 SHAP 值（论文核心优势）
        """
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
            shapley_values = self.model.get_shapley_values(x_tensor)
        return shapley_values.cpu().numpy()


def show_img(fig, filename='plot.png'):
    import os
    output_dir = '多因子分析可视化'
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _get_color_cycle(n, base_colors=None):
    if base_colors is None:
        base_colors = COLORS
    if n <= len(base_colors):
        return [base_colors[i] for i in range(n)]
    cmap = plt.get_cmap('tab20')
    return [cmap(i / max(1, n - 1)) for i in range(n)]


def plot_ic_ir_multi(ic_dict, ir_dict):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("多因子 IC / IR 对比", fontsize=16)

    colors = _get_color_cycle(len(ic_dict))
    for i, (name, ic) in enumerate(ic_dict.items()):
        ax1.plot(ic.index, ic.values, color=colors[i], label=name, linewidth=1.8)
    ax1.axhline(0, c='k', ls='--')
    ax1.set_title('IC 序列')
    ax1.legend(loc='best', fontsize='small')

    for i, (name, ic) in enumerate(ic_dict.items()):
        ax2.plot(ic.index, ic.rolling(20, min_periods=1).mean(), color=colors[i], linewidth=1.8)
    ax2.axhline(0, c='k', ls='--')
    ax2.set_title('滚动 IC')
    ax2.legend(list(ic_dict.keys()), loc='best', fontsize='small')

    all_ic = [ic.dropna().values for ic in ic_dict.values() if len(ic.dropna()) > 0]
    if len(all_ic) > 0:
        hist_colors = _get_color_cycle(len(all_ic))
        ax3.hist(all_ic, bins=20, color=hist_colors, label=list(ic_dict.keys()), alpha=0.45, edgecolor='black')
    ax3.set_title('IC 分布')
    ax3.legend(loc='best', fontsize='small')

    names, vals = list(ir_dict.keys()), list(ir_dict.values())
    bar_colors = _get_color_cycle(len(names))
    ax4.bar(names, vals, color=bar_colors)
    ax4.axhline(0.2, c='orange', ls='--')
    ax4.axhline(0.5, c='red', ls='--')
    ax4.set_title('IR')
    ax4.set_ylabel('IR')
    ax4.set_xticklabels(names, rotation=45, ha='right')

    show_img(fig, '多因子_IC_IR分析.png')


def plot_multi_quantile(quantile_return_dict):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9))
    fig.suptitle("多因子分层收益", fontsize=16)

    x = np.arange(1, QUANTILES + 1)
    w = 0.15
    colors = _get_color_cycle(len(quantile_return_dict))
    for i, (name, q) in enumerate(quantile_return_dict.items()):
        mean_values = q.mean(axis=0).reindex(range(1, QUANTILES + 1), fill_value=np.nan)
        ax1.bar(x + i * w, mean_values, w, color=colors[i], label=name, alpha=0.9)
    ax1.axhline(0, c='k', ls='--')
    ax1.set_title('分层平均收益')
    ax1.set_xlabel('分层')
    ax1.set_ylabel('平均收益')
    ax1.legend(loc='best', fontsize='small')

    for i, (name, q) in enumerate(quantile_return_dict.items()):
        if 1 in q.columns and QUANTILES in q.columns:
            diff = q[QUANTILES] - q[1]
            ax2.plot(diff.cumsum(), color=colors[i], label=name, linewidth=1.8)
        else:
            ax2.plot(np.zeros(len(q)), color=colors[i], label=name, linewidth=1.8)
    ax2.set_title('高低分层收益差累积')
    ax2.set_xlabel('时间点')
    ax2.set_ylabel('累积收益差')
    ax2.legend(loc='best', fontsize='small')
    show_img(fig, '多因子_分位数收益.png')


def plot_corr(factor_df):
    import seaborn as sns

    corr = factor_df.corr()
    fig = plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("因子相关性")
    show_img(fig, '因子相关性.png')


def plot_combined(combined_qret):
    fig = plt.figure(figsize=(9, 5))
    plt.bar(range(1, QUANTILES + 1), combined_qret.mean(), color=COMBINED_COLORS[:QUANTILES])
    plt.axhline(0, c='k', ls='--')
    plt.title("合成因子分层平均收益")
    plt.xlabel('分层')
    plt.ylabel('平均收益')
    show_img(fig, '合成因子_分层平均收益.png')


def cross_sectional_quantile_returns(factor_mat, y_mat, quantiles=5):
    T, N = factor_mat.shape
    qret = pd.DataFrame(np.nan, index=range(T), columns=range(1, quantiles + 1), dtype=float)

    for t in range(T):
        f = factor_mat[t]
        y = y_mat[t]
        valid = np.isfinite(f) & np.isfinite(y)
        if np.sum(valid) < quantiles:
            continue
        try:
            labels = pd.qcut(pd.Series(f[valid]), q=quantiles, labels=False, duplicates='drop') + 1
        except Exception:
            continue
        for q in range(1, quantiles + 1):
            mask = labels == q
            if mask.sum() > 0:
                qret.iat[t, q - 1] = np.nanmean(y[valid][mask])
    qret.columns = range(1, quantiles + 1)
    return qret


def row_rankdata(mat):
    ranked = np.full(mat.shape, np.nan)
    for t in range(mat.shape[0]):
        row = mat[t]
        mask = np.isfinite(row)
        if np.sum(mask) > 0:
            ranked[t, mask] = _rankdata(row[mask], method="average")
    return ranked


def build_real_data_visualizations():
    try:
        factor_now
    except NameError:
        print("未找到因子数据，跳过可视化")
        return

    if not isinstance(factor_now, list) or len(factor_now) == 0:
        print("因子数据为空，跳过可视化")
        return

    factor_matrices = {f"因子{i+1}": factor_now[i] for i in range(len(factor_now))}
    if len(factor_matrices) == 0:
        print("真实数据中未找到可用因子矩阵，跳过可视化")
        return

    ic_dict = {}
    ir_dict = {}
    quantile_return_dict = {}

    for name, mat in factor_matrices.items():
        stats = calc_ic_stats(mat, y_now)
        ic_dict[name] = pd.Series(stats["ic_series"]).reset_index(drop=True)
        ir_dict[name] = float(stats["icir"]) if np.isfinite(stats["icir"]) else 0.0
        quantile_return_dict[name] = cross_sectional_quantile_returns(mat, y_now, quantiles=QUANTILES)

    plot_ic_ir_multi(ic_dict, ir_dict)
    plot_multi_quantile(quantile_return_dict)

    factor_ts = {
        name: np.nanmean(mat, axis=1)
        for name, mat in factor_matrices.items()
    }
    index = pivoted_now[features[0]].index if features and features[0] in pivoted_now else None
    if index is not None:
        plot_corr(pd.DataFrame(factor_ts, index=index))

    rank_matrices = [row_rankdata(mat) for mat in factor_matrices.values()]
    combined_mat = np.nanmean(np.stack(rank_matrices, axis=-1), axis=-1)
    combined_qret = cross_sectional_quantile_returns(combined_mat, y_now, quantiles=QUANTILES)
    plot_combined(combined_qret)


# ## 因子计算逻辑函数
def _normalize_window_length(d, x=None, other=None, min_len=1, default=None, max_len=120, return_type="int"):
    """
    规范化窗口长度参数，确保窗口长度合法且鲁棒。
    
    参数:
        d: int/float/list/np.ndarray
            原始窗口长度。可为单个标量或数组。如果为数组，则优先取首个有限值。
        x: optional, None/array-like
            用于辅助判断可用长度（如主时间序列），如不为标量，则以长度限制窗口最大值。
        other: optional, None/array-like
            链接窗口的另一序列，也会用于辅助判断窗口最大长度。
        min_len: int, default 1
            窗口的最小长度。确保d不会小于这个值。
        default: int/float, default None
            当d无效时的默认窗口长度。如果未设置，则取min_len。
        max_len: int, default 120
            窗口允许的最大长度。
        return_type: str, "int" 或 "float"，结果的返回类型

    返回:
        int/float: 最终规范化的窗口长度，类型根据return_type指定
    """
    # 步骤1：如果default未给定，则用最小长度填充
    if default is None:
        default = min_len

    # 步骤2：解析输入d，如果是标量则直接取，否则优先用数组的首个有限值
    if np.isscalar(d):
        try:
            # 无效值或者非数值，强制用default
            if not np.isfinite(d):
                d = default
        except Exception:
            # d不是数字类型时，也fallback
            d = default
    else:
        # d是数组，尝试提取第一个有限值
        d_arr = np.asarray(d)
        if d_arr.size > 0:
            finite_vals = d_arr[np.isfinite(d_arr)]
            # 如果有有限值，取第一个，否则用default
            d = float(finite_vals.flat[0]) if finite_vals.size > 0 else default
        else:
            d = default

    # 步骤3：四舍五入（默认int类型），至少为min_len
    if return_type == "float":
        d = float(np.round(d, 6))   # 保留小数点后6位，可修改
        d = max(float(min_len), d)
    else:
        d = int(np.round(d))
        d = max(min_len, d)

    # 步骤4：尝试根据x和other自动推断最大窗口（防止超出数据长度）
    available_len = None
    if x is not None and not np.isscalar(x):
        x_arr = np.asarray(x)
        if x_arr.size > 0 and len(x_arr.shape) > 0:
            available_len = x_arr.shape[0]
    if other is not None and not np.isscalar(other):
        other_arr = np.asarray(other)
        if other_arr.size > 0 and len(other_arr.shape) > 0:
            # 如果available_len已设置，则取两个序列中较小的长度
            available_len = other_arr.shape[0] if available_len is None else min(available_len, other_arr.shape[0])

    # 步骤5：窗口长度不得超过max_len和推断所得数据可用长度
    max_len_val = float(max_len) if return_type == "float" else max_len
    if available_len is not None:
        max_lim = float(available_len) if return_type == "float" else available_len
        d = min(d, max_len_val, max_lim)
    else:
        d = min(d, max_len_val)

    # 返回最终规范化的窗口长度
    return d


class My:
    @staticmethod
    def _ensure_array(data):
        """确保输入数据为numpy数组"""
        if isinstance(data, (list, tuple)):
            return np.array(data)
        elif isinstance(data, pd.Series):
            return data.values
        elif isinstance(data, np.ndarray):
            return data
        else:
            return np.array([data])

    @staticmethod
    def _ensure_series(data):
        """确保输入数据为pandas Series"""
        if isinstance(data, pd.Series):
            return data
        elif isinstance(data, (list, tuple, np.ndarray)):
            return pd.Series(data)
        else:
            return pd.Series([data])

    @staticmethod
    def _ensure_np_output(data):
        """确保输出为numpy格式"""
        if isinstance(data, tuple):
            return tuple(My._ensure_np_output(item) for item in data)
        if isinstance(data, pd.Series):
            return data.values
        if isinstance(data, np.ndarray):
            return data
        if isinstance(data, (list, tuple)):
            return np.asarray(data)
        return np.asarray(data)

    # ------------------ 0级:核心工具函数 ------------------
    @staticmethod
    def ADD(A,B):
        """返回A+B"""
        A = My._ensure_array(A)
        B = My._ensure_array(B)
        return My._ensure_np_output(A+B)

    @staticmethod
    def SUB(A,B):
        """返回A-B"""
        A = My._ensure_array(A)
        B = My._ensure_array(B)
        return My._ensure_np_output(A-B)

    @staticmethod
    def MUL(A,B):
        """返回A*B"""
        A = My._ensure_array(A)
        B = My._ensure_array(B)
        return My._ensure_np_output(A * B)

    @staticmethod
    def DIV(A,B):
        """返回A/B（安全除零）"""
        A = My._ensure_array(A)
        B = My._ensure_array(B)
        B_safe = np.where(np.abs(B) < 1e-10, 1e-10, B)
        return My._ensure_np_output(A / B_safe)

    @staticmethod
    def ABS(S):
        """返回N的绝对值"""
        S = My._ensure_array(S)
        return My._ensure_np_output(np.abs(S))

    @staticmethod
    def LN(S):
        """求底是e的自然对数"""
        S = My._ensure_array(S)
        return My._ensure_np_output(np.log(np.abs(S) + 1e-8))

    @staticmethod
    def INV(S):
        """求S的倒数"""
        S = My._ensure_array(S)
        S_safe = np.where(np.abs(S) < 1e-10, 1e-10, S)
        return My._ensure_np_output(1/S_safe)

    @staticmethod
    def POW(S, N):
        """求S的N次方"""
        S = My._ensure_array(S)
        return My._ensure_np_output(np.power(S, N))

    @staticmethod
    def SQRT(S):
        """求S的平方根"""
        S = My._ensure_array(S)
        return My._ensure_np_output(np.sqrt(S))

    @staticmethod
    def SIN(S):
        """求S的正弦值(弧度)"""
        S = My._ensure_array(S)
        return My._ensure_np_output(np.sin(S))

    @staticmethod
    def COS(S):
        """求S的余弦值(弧度)"""
        S = My._ensure_array(S)
        return My._ensure_np_output(np.cos(S))

    @staticmethod
    def TAN(S):
        """求S的正切值(弧度)"""
        S = My._ensure_array(S)
        return My._ensure_np_output(np.tan(S))

    @staticmethod
    def MAX(S1, S2):
        """序列max"""
        S1 = My._ensure_array(S1)
        S2 = My._ensure_array(S2)
        return My._ensure_np_output(np.maximum(S1, S2))

    @staticmethod
    def MIN(S1, S2):
        """序列min"""
        S1 = My._ensure_array(S1)
        S2 = My._ensure_array(S2)
        return My._ensure_np_output(np.minimum(S1, S2))

    @staticmethod
    def IF(S, A, B):
        """序列布尔判断 return=A if S==True else B"""
        S = My._ensure_array(S)
        A = My._ensure_array(A)
        B = My._ensure_array(B)
        return My._ensure_np_output(np.where(S, A, B))

    @staticmethod
    def AND(S1, S2):
        """逻辑与运算"""
        S1 = My._ensure_array(S1)
        S2 = My._ensure_array(S2)
        return My._ensure_np_output(np.logical_and(S1, S2))

    @staticmethod
    def OR(S1, S2):
        """逻辑或运算"""
        S1 = My._ensure_array(S1)
        S2 = My._ensure_array(S2)
        return My._ensure_np_output(np.logical_or(S1, S2))

    @staticmethod
    def NOT(S):
        """逻辑非运算"""
        S = My._ensure_array(S)
        return My._ensure_np_output(np.logical_not(S))

    @staticmethod
    def RANK(x, d=10):
        """
        滚动窗口排名归一化（防止未来函数泄漏）
        
        在指定窗口长度内计算当前值的排名，并归一化到[0,1]区间
        这是量化研究中常用的因子标准化方法
        
        参数:
            x: 输入时间序列数据
            d: 窗口长度，默认10期。不满d期的位置填nan
        
        返回:
            归一化排名序列，前d-1期为nan
        
        注意:
            - 使用滚动窗口避免未来函数泄漏
            - 排名归一化有助于因子标准化
        """
        # 如果输入是标量，直接返回
        if np.isscalar(x):
            return x
        # 转换输入为numpy数组
        x = np.asarray(x)
        # 如果输入为空数组，直接返回
        if x.size == 0:
            return x
        d = _normalize_window_length(d, x=x, min_len=4, default=10, max_len=120)
        # 如果是一维数组
        if x.ndim == 1:
            # 创建与x同形状的全nan数组用于存放结果
            res = np.full_like(x, np.nan, dtype=np.float64)
            # 从第d-1个元素开始，遍历每个位置
            for i in range(d-1, len(x)):
                # 取当前窗口的d个元素
                window = x[i-d+1:i+1]
                # 对窗口内元素进行排名归一化
                ranks = window.argsort().argsort() / (len(window) - 1 + 1e-8)
                # 将当前时刻的排名结果赋值给res
                res[i] = ranks[-1]
            # 返回结果
            return res
        else:
            # 如果是二维数组，创建同形状的全nan数组
            res = np.full_like(x, np.nan, dtype=np.float64)
            # 遍历每一列（通常每列代表一只股票）
            for j in range(x.shape[1]):
                # 取第j列
                col = x[:, j]
                # 从第d-1个元素开始，遍历每个位置
                for i in range(d-1, len(col)):
                    # 取当前窗口的d个元素
                    window = col[i-d+1:i+1]
                    # 对窗口内元素进行排名归一化
                    ranks = window.argsort().argsort() / (len(window) - 1 + 1e-8)
                    # 将当前时刻的排名结果赋值给res
                    res[i, j] = ranks[-1]
            # 返回结果
            return res

    @staticmethod
    def TS_RANK(x, d):
        """
        滚动窗口排名
        
        计算当前值在滚动窗口内的排名位置
        
        参数:
            x: 输入时间序列
            d: 窗口长度
        
        返回:
            排名序列，前d-1期为nan
        
        注意:
            排名越高表示当前值在窗口内越大
        """
        x = np.asarray(x)
        if np.isscalar(x):
            return x
        d = _normalize_window_length(d, x=x, min_len=4, default=10, max_len=120)
        if x.ndim == 1:
            res = np.full_like(x, np.nan, dtype=np.float64)
            for i in range(d-1, len(x)):
                window = x[i-d+1:i+1]
                # 计算当前值在窗口中的排名
                current_value = x[i]
                rank_position = np.sum(window < current_value)
                normalized_rank = rank_position / (len(window) + 1e-8)
                res[i] = normalized_rank
            return res
        else:
            res = np.full_like(x, np.nan, dtype=np.float64)
            for j in range(x.shape[1]):
                for i in range(d-1, x.shape[0]):
                    window = x[i-d+1:i+1, j]
                    current_value = x[i, j]
                    rank_position = np.sum(window < current_value)
                    normalized_rank = rank_position / (len(window) + 1e-8)
                    res[i, j] = normalized_rank
            return res

    @staticmethod
    def TS_ZSCORE(x, d):
        """
        滚动窗口Z-score标准化
        
        计算 (当前值 - 窗口均值) / 窗口标准差
        
        参数:
            x: 输入时间序列
            d: 窗口长度
        
        返回:
            Z-score标准化序列，前d-1期为nan
        
        注意:
            Z-score衡量当前值偏离均值的程度（以标准差为单位）
        """
        x = np.asarray(x)
        if np.isscalar(x):
            return 0.0
        d = _normalize_window_length(d, x=x, min_len=4, default=10, max_len=120)
        if x.ndim == 1:
            res = np.full_like(x, np.nan, dtype=np.float64)
            for i in range(d-1, len(x)):
                window = x[i-d+1:i+1]
                mean = np.mean(window)
                std = np.std(window, ddof=1)
                res[i] = (x[i] - mean) / (std + 1e-8)
            return res
        else:
            res = np.full_like(x, np.nan, dtype=np.float64)
            for j in range(x.shape[1]):
                for i in range(d-1, x.shape[0]):
                    window = x[i-d+1:i+1, j]
                    mean = np.mean(window)
                    std = np.std(window, ddof=1)
                    res[i, j] = (x[i, j] - mean) / (std + 1e-8)
            return res

    @staticmethod
    def RANK_SUB(x, y, d=10):
        """
        滚动窗口排名差值（防止未来函数泄漏）
        
        计算两个序列排名的差值
        
        参数:
            x, y: 两个输入时间序列
            d: 排名窗口长度
        
        返回:
            排名差值序列
        
        用途:
            构建相对强度因子，比较两个指标的相对表现
        """
        d = _normalize_window_length(d, x=x, other=y, min_len=1, default=10, max_len=120)
        rx = My.RANK(x, d)
        ry = My.RANK(y, d)
        try:
            return rx - ry
        except:
            # 形状不匹配时返回0数组
            return np.zeros_like(rx) if hasattr(rx, 'size') and rx.size > 0 else 0

    @staticmethod
    def RANK_DIV(x, y, d=10):
        """
        滚动窗口排名比值（防止未来函数泄漏）
        
        计算两个序列排名的比值
        
        参数:
            x, y: 两个输入时间序列
            d: 排名窗口长度
        
        返回:
            排名比值序列
        
        用途:
            构建相对强度因子，比较两个指标的相对表现
        """
        d = _normalize_window_length(d, x=x, other=y, min_len=1, default=10, max_len=120)
        rx = My.RANK(x, d)
        ry = My.RANK(y, d)
        try:
            ry_safe = np.where(ry < 1e-10, 1e-10, ry)  # 避免除零
            return rx / ry_safe
        except:
            return np.ones_like(rx) if hasattr(rx, 'size') and rx.size > 0 else 1

    @staticmethod
    def SIGMOID(x):
        """
        安全sigmoid函数（避免溢出）
        
        计算 1 / (1 + exp(-x))，将输入映射到(0,1)区间
        
        参数:
            x: 输入数值或数组
        
        返回:
            sigmoid变换后的结果
        
        特点:
            - 输出范围(0,1)
            - 截断极端值避免数值溢出
            - 常用于因子标准化
        """

        x = np.asarray(x)
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))  # 截断极端值


    @staticmethod
    def CORR(x, y, d):
        """
        滚动窗口相关系数计算（防止未来函数泄漏）
        
        计算两个时间序列在滚动窗口内的皮尔逊相关系数
        
        参数:
            x, y: 两个输入时间序列
            d: 滚动窗口长度
        
        返回:
            滚动相关系数序列，前d-1期为nan
        
        注意:
            - 相关系数范围为[-1, 1]
            - 窗口长度至少需要2个观测值
        """
        x = np.asarray(x)
        y = np.asarray(y)
        if x.size == 0 or y.size == 0 or np.isscalar(x) or np.isscalar(y):
            return np.nan
        # print(np.nanstd(x),np.nanstd(y))
        if np.nanstd(x) < 1e-10 or np.nanstd(y) < 1e-10:
            return np.full_like(x, 0, dtype=np.float64)
        d = _normalize_window_length(d, x=x, min_len=4, default=2, max_len=120)
        
        if x.ndim == 1:
            res = np.full_like(x, np.nan, dtype=np.float64)
            for i in range(d-1, len(x)):
                # 取当前窗口的d个元素
                x_window = x[i-d+1:i+1]
                y_window = y[i-d+1:i+1]
                if len(x_window) >= 2:
                    try:
                        corr = np.corrcoef(x_window, y_window)[0, 1]
                        res[i] = corr if not np.isnan(corr) else np.nan
                    except:
                        res[i] = np.nan
            return res
        else:
            res = np.full_like(x, np.nan, dtype=np.float64)
            for j in range(x.shape[1]):
                for i in range(d-1, x.shape[0]):
                    x_window = x[i-d+1:i+1, j]
                    y_window = y[i-d+1:i+1, j]
                    if len(x_window) >= 2:
                        try:
                            corr = np.corrcoef(x_window, y_window)[0, 1]
                            res[i, j] = corr if not np.isnan(corr) else np.nan
                        except:
                            res[i, j] = np.nan
            return res

    @staticmethod
    def COVA(x, y, d):
        """
        滚动窗口协方差计算（防止未来函数泄漏）
        
        计算两个时间序列在滚动窗口内的协方差
        
        参数:
            x, y: 两个输入时间序列
            d: 滚动窗口长度
        
        返回:
            滚动协方差序列，前d-1期为nan
        
        注意:
            协方差反映两个变量的线性关系强度和方向
        """
        x = np.asarray(x)
        y = np.asarray(y)
        if x.size == 0 or y.size == 0 or np.isscalar(x) or np.isscalar(y):
            return np.nan
        if np.nanstd(x) < 1e-10 or np.nanstd(y) < 1e-10:
            return np.full_like(x, 0, dtype=np.float64)
        d = _normalize_window_length(d, x=x, other=y, min_len=4, default=10, max_len=120)
        
        if x.ndim == 1:
            res = np.full_like(x, np.nan, dtype=np.float64)
            for i in range(d-1, len(x)):
                # 取当前窗口的d个元素
                x_window = x[i-d+1:i+1]
                y_window = y[i-d+1:i+1]
                if len(x_window) >= 2:
                    try:
                        cov = np.cov(x_window, y_window)[0, 1]
                        res[i] = cov if not np.isnan(cov) else np.nan
                    except:
                        res[i] = np.nan
            return res
        else:
            res = np.full_like(x, np.nan, dtype=np.float64)
            for j in range(x.shape[1]):
                for i in range(d-1, x.shape[0]):
                    x_window = x[i-d+1:i+1, j]
                    y_window = y[i-d+1:i+1, j]
                    if len(x_window) >= 2:
                        try:
                            cov = np.cov(x_window, y_window)[0, 1]
                            res[i, j] = cov if not np.isnan(cov) else np.nan
                        except:
                            res[i, j] = np.nan
            return res

    @staticmethod
    def SCALE(x, d=10): 
        """
        滚动窗口缩放至绝对值和为1（防止未来函数泄漏）
        
        将当前值除以滚动窗口内所有值的绝对值和，实现标准化
        
        参数:
            x: 输入时间序列
            d: 滚动窗口长度，默认10
        
        返回:
            标准化后的序列，前d-1期为nan
        
        用途:
            因子标准化，使因子值在合理范围内
        """
        if np.isscalar(x):
            return x
        x = np.asarray(x)
        if x.size == 0:
            return x
        d = _normalize_window_length(d, x=x, min_len=4, default=10, max_len=120)
        
        if x.ndim == 1:
            res = np.full_like(x, np.nan, dtype=np.float64)
            for i in range(d-1, len(x)):
                # 取当前窗口的d个元素
                window = x[i-d+1:i+1]
                # 计算窗口内绝对值和
                s = np.sum(np.abs(window))
                s = s if s > 1e-10 else 1e-10  # 避免除零
                # 缩放当前值
                res[i] = x[i] / s
            return res
        else:
            res = np.full_like(x, np.nan, dtype=np.float64)
            for j in range(x.shape[1]):
                col = x[:, j]
                for i in range(d-1, len(col)):
                    # 取当前窗口的d个元素
                    window = col[i-d+1:i+1]
                    # 计算窗口内绝对值和
                    s = np.sum(np.abs(window))
                    s = s if s > 1e-10 else 1e-10  # 避免除零
                    # 缩放当前值
                    res[i, j] = col[i] / s
            return res

    def SIGNEDPOWER(x): 
        """
        带符号二次幂运算 sign(x) * (abs(x) ** 2)
        
        保持原始符号的同时放大数值差异
        
        参数:
            x: 输入数值或数组
        
        返回:
            带符号的二次幂结果
        
        特点:
            - 正数保持正，负数保持负
            - 放大绝对值大的数，压缩绝对值小的数
        """
        return np.sign(x) * (np.abs(x) ** 2)

    @staticmethod
    def REF(S, N=1):
        """引用S在N个周期前的值
        支持N为单个数值:
        - 当N为数值时:引用S在N个周期前的值

        注:
        1.当N为有效值,但当前的k线数不足N根,返回空值
        2.N为0时返回当前S值
        3.N为空值时返回空值
        """
        # 确保输入为numpy数组
        S = My._ensure_array(S)
        
        # 处理二维数组：对每一列分别计算REF
        if isinstance(S, np.ndarray) and S.ndim == 2:
            T, N_cols = S.shape
            result = np.full((T, N_cols), np.nan, dtype=np.float64)
            for col_idx in range(N_cols):
                s_col = S[:, col_idx]
                try:
                    result[:, col_idx] = My.REF(s_col, N)
                except Exception:
                    # 如果某一列计算失败，保持NaN
                    pass
            return My._ensure_np_output(result)
        
        # 一维数组的处理逻辑
        S = My._ensure_series(S)
        N = _normalize_window_length(N, x=S.values, min_len=1, default=1, max_len=120)

        if np.isnan(N) or N is None:
            # N为空值时返回空值
            return My._ensure_np_output(np.full(len(S), np.nan))
        elif N == 0:
            # N为0时返回当前X值
            return My._ensure_np_output(S.values)
        else:
            # 原来的逻辑:对序列整体下移动N
            return My._ensure_np_output(S.shift(N).values)

    @staticmethod
    def DIFF(S, N=1):
        """前一个值减后一个值,前面会产生nan"""
         # 确保输入为numpy数组
        S = My._ensure_array(S)
        
        # 处理二维数组：对每一列分别计算DIFF
        if isinstance(S, np.ndarray) and S.ndim == 2:
            T, N_cols = S.shape
            result = np.full((T, N_cols), np.nan, dtype=np.float64)
            for col_idx in range(N_cols):
                s_col = S[:, col_idx]
                try:
                    result[:, col_idx] = My.DIFF(s_col, N)
                except Exception:
                    # 如果某一列计算失败，保持NaN
                    pass
            return My._ensure_np_output(result)
        
        # 一维数组的处理逻辑
        S = My._ensure_series(S)
        N = _normalize_window_length(N, x=S.values, min_len=1, default=1, max_len=120)
        return My._ensure_np_output(S.diff(N).values)
    
    def STD(S, N, ddof=1):
        """求序列的N日标准差,返回序列"""
        # 确保输入为numpy数组
        S = My._ensure_array(S)
        
        # 处理二维数组：对每一列分别计算STD
        if isinstance(S, np.ndarray) and S.ndim == 2:
            T, N_cols = S.shape
            result = np.full((T, N_cols), np.nan, dtype=np.float64)
            for col_idx in range(N_cols):
                s_col = S[:, col_idx]
                try:
                    result[:, col_idx] = My.STD(s_col, N, ddof)
                except Exception:
                    # 如果某一列计算失败，保持NaN
                    pass
            return My._ensure_np_output(result)
        
        # 一维数组的处理逻辑
        S = My._ensure_series(S)
        N = _normalize_window_length(N, x=S.values, min_len=2, default=2, max_len=120)
        return My._ensure_np_output(S.rolling(N).std(ddof=ddof).values)

    @staticmethod
    def SUM(S, N):
        """对序列求N天累计和,返回序列 N=0对序列所有依次求和"""
        # 确保输入为numpy数组
        S = My._ensure_array(S)
        
        # 处理二维数组：对每一列分别计算SUM
        if isinstance(S, np.ndarray) and S.ndim == 2:
            T, N_cols = S.shape
            result = np.full((T, N_cols), np.nan, dtype=np.float64)
            for col_idx in range(N_cols):
                s_col = S[:, col_idx]
                try:
                    result[:, col_idx] = My.SUM(s_col, N)
                except Exception as e:
                    # 如果某一列计算失败，保持NaN
                    pass
            return My._ensure_np_output(result)
        
        # 一维数组的处理逻辑
        S_series = My._ensure_series(S)
        N = _normalize_window_length(N, x=S_series.values, min_len=2, default=2, max_len=120)
        # 标量窗口: 直接使用 pandas 计算
        if not isinstance(N, (list, np.ndarray, pd.Series)):
            window = int(round(N))
            if window > 0:
                return My._ensure_np_output(S_series.rolling(window).sum().values)
            else:
                return My._ensure_np_output(S_series.cumsum().values)
        # 向量窗口: 对每个位置使用对应窗口长度进行求和
        N_arr = My._ensure_array(N).astype(int)
        values = S_series.values.astype(float)
        n = len(values)
        # 前缀和加速区间求和
        csum = np.concatenate(([0.0], np.cumsum(values)))
        out = np.zeros(n, dtype=float)
        for i in range(n):
            k = N_arr[i] if i < len(N_arr) else N_arr[-1]
            # N<=0 表示从第一个有效值累计到当前; N>i+1 则退化为从起点到当前
            if k <= 0 or k > i + 1:
                start_idx = 0
            else:
                start_idx = i + 1 - k
            out[i] = csum[i + 1] - csum[start_idx]
        return My._ensure_np_output(out)

    @staticmethod
    def CONST(S):
        """返回序列S最后的值组成常量序列"""
        S = My._ensure_array(S)
        return My._ensure_np_output(np.full(len(S), S[-1]))

    @staticmethod
    def HHV(S, N):
        """求X在N个周期内的最高值
        支持N为单个数值或列表:
        - 当N为数值时:求X在N个周期内的最高值
        - 当N为列表时:每个位置使用对应的N值求对应周期内的最高值

        注:
        1.N包含当前k线
        2.若N为0则从第一个有效值开始算起
        3.当N为有效值,但当前的k线数不足N根,按照实际的根数计算
        4.N为空值时,返回空值
        5.N可以是列表
        """
        # 确保输入为numpy数组
        S = My._ensure_array(S)
        
        # 处理二维数组：对每一列分别计算HHV
        if isinstance(S, np.ndarray) and S.ndim == 2:
            T, N_cols = S.shape
            result = np.full((T, N_cols), np.nan, dtype=np.float64)
            for col_idx in range(N_cols):
                s_col = S[:, col_idx]
                try:
                    result[:, col_idx] = My.HHV(s_col, N)
                except Exception:
                    # 如果某一列计算失败，保持NaN
                    pass
            return My._ensure_np_output(result)
        
        # 一维数组的处理逻辑
        S = My._ensure_series(S)
        N = _normalize_window_length(N, x=S.values, min_len=2, default=2, max_len=120)
        series_length = len(S)

        # 单个数值的情况 - 尽可能使用pandas原生的向量化操作
        if not isinstance(N, (list, np.ndarray)):
            if pd.isna(N) or N is None:
                # N为空值时,返回空值
                return My._ensure_np_output(np.full(series_length, np.nan))
            elif N == 0:
                # 若N为0则从第一个有效值开始算起
                # 使用pandas的expanding窗口函数替代手动循环
                return My._ensure_np_output(S.expanding().max().values)
            else:
                # 处理单个数值N的情况,使用pandas的rolling窗口函数
                return My._ensure_np_output(S.rolling(window=int(N), min_periods=1).max().values)
        else:
            # 如果N是列表,进行向量化优化的逐元素计算
            N_array = np.asarray(N)
            result = np.full(series_length, np.nan)

            # 预先处理N数组,确保长度匹配
            if len(N_array) < series_length:
                # 如果N数组长度不足,用最后一个值填充
                last_N = N_array[-1] if len(N_array) > 0 else np.nan
                extended_N = np.full(series_length, last_N)
                extended_N[: len(N_array)] = N_array
                N_array = extended_N
            elif len(N_array) > series_length:
                # 如果N数组过长,截断
                N_array = N_array[:series_length]

            # 向量化处理N=0的特殊情况
            zero_mask = (N_array == 0) & ~np.isnan(N_array)
            if np.any(zero_mask):
                # 只对N=0的位置应用expanding max
                expanding_max = S.expanding().max().values
                result[zero_mask] = expanding_max[zero_mask]

            # 处理非0且非空的N值
            valid_mask = (N_array != 0) & ~np.isnan(N_array) & ~zero_mask
            if np.any(valid_mask):
                valid_indices = np.where(valid_mask)[0]

                # 预处理S为NumPy数组以加速访问
                S_values = S.values

                for i in valid_indices:
                    n_value = int(N_array[i])
                    # 当N为有效值,但当前的k线数不足N根,按照实际的根数计算
                    start_idx = max(0, i - n_value + 1)
                    # 直接使用NumPy的切片和max函数
                    window_values = S_values[start_idx : i + 1]
                    # 过滤NaN值并计算最大值
                    valid_window = window_values[~np.isnan(window_values)]
                    if len(valid_window) > 0:
                        result[i] = np.max(valid_window)

            return My._ensure_np_output(result)

    @staticmethod
    def HV(S, N):
        """求X在N个周期内的最高值(不包括当前K线)
        支持N为单个数值或列表:
        - 当N为数值时:求X在N个周期内的最高值(不包括当前K线)
        - 当N为列表时:每个位置使用对应的N值求对应周期内的最高值(不包括当前K线)

        注:
        1.N不包含当前k线
        2.若N为0则从第一个有效值开始算起(不包括当前K线)
        3.当N为有效值,但当前的k线数不足N根,按照实际的根数计算
        4.N为空值时,返回空值
        5.N可以是列表
        """
        # 确保输入为numpy数组
        S = My._ensure_array(S)
        
        # 处理二维数组：对每一列分别计算HV
        if isinstance(S, np.ndarray) and S.ndim == 2:
            T, N_cols = S.shape
            result = np.full((T, N_cols), np.nan, dtype=np.float64)
            for col_idx in range(N_cols):
                s_col = S[:, col_idx]
                try:
                    result[:, col_idx] = My.HV(s_col, N)
                except Exception:
                    # 如果某一列计算失败，保持NaN
                    pass
            return My._ensure_np_output(result)
        
        # 一维数组的处理逻辑
        S = My._ensure_series(S)
        N = _normalize_window_length(N, x=S.values, min_len=2, default=2, max_len=120)
        series_length = len(S)

        # 单个数值的情况
        if not isinstance(N, (list, np.ndarray)):
            if pd.isna(N) or N is None:
                # N为空值时,返回空值
                return My._ensure_np_output(np.full(series_length, np.nan))
            elif N == 0:
                # 若N为0则从第一个有效值开始算起(不包括当前K线)
                result = np.full(series_length, np.nan)
                for i in range(1, series_length):  # 从第二根K线开始
                    window_values = S.iloc[:i].values  # 不包括当前K线
                    valid_window = window_values[~np.isnan(window_values)]
                    if len(valid_window) > 0:
                        result[i] = np.max(valid_window)
                return My._ensure_np_output(result)
            else:
                # 处理单个数值N的情况,不包括当前K线
                result = np.full(series_length, np.nan)
                for i in range(series_length):
                    n_value = int(N)
                    # 不包括当前K线,所以从i-n_value开始到i-1结束
                    start_idx = max(0, i - n_value)
                    end_idx = i  # 不包括当前K线
                    if start_idx < end_idx:
                        window_values = S.iloc[start_idx:end_idx].values
                        valid_window = window_values[~np.isnan(window_values)]
                        if len(valid_window) > 0:
                            result[i] = np.max(valid_window)
                return My._ensure_np_output(result)
        else:
            # 如果N是列表,进行向量化优化的逐元素计算
            N_array = np.asarray(N)
            result = np.full(series_length, np.nan)

            # 预先处理N数组,确保长度匹配
            if len(N_array) < series_length:
                # 如果N数组长度不足,用最后一个值填充
                last_N = N_array[-1] if len(N_array) > 0 else np.nan
                extended_N = np.full(series_length, last_N)
                extended_N[: len(N_array)] = N_array
                N_array = extended_N
            elif len(N_array) > series_length:
                # 如果N数组过长,截断
                N_array = N_array[:series_length]

            # 向量化处理N=0的特殊情况(不包括当前K线)
            zero_mask = (N_array == 0) & ~np.isnan(N_array)
            if np.any(zero_mask):
                zero_indices = np.where(zero_mask)[0]
                for i in zero_indices:
                    if i > 0:  # 第一根K线无法计算(没有历史数据)
                        window_values = S.iloc[:i].values  # 不包括当前K线
                        valid_window = window_values[~np.isnan(window_values)]
                        if len(valid_window) > 0:
                            result[i] = np.max(valid_window)

            # 处理非0且非空的N值(不包括当前K线)
            valid_mask = (N_array != 0) & ~np.isnan(N_array) & ~zero_mask
            if np.any(valid_mask):
                valid_indices = np.where(valid_mask)[0]

                # 预处理S为NumPy数组以加速访问
                S_values = S.values

                for i in valid_indices:
                    n_value = int(N_array[i])
                    # 不包括当前K线,所以从i-n_value开始到i-1结束
                    start_idx = max(0, i - n_value)
                    end_idx = i  # 不包括当前K线
                    if start_idx < end_idx:
                        window_values = S_values[start_idx:end_idx]
                        # 过滤NaN值并计算最大值
                        valid_window = window_values[~np.isnan(window_values)]
                        if len(valid_window) > 0:
                            result[i] = np.max(valid_window)

            return My._ensure_np_output(result)

    @staticmethod
    def LLV(S, N):
        """求X在N个周期内的最低值
        支持N为单个数值或列表:
        - 当N为数值时:求X在N个周期内的最低值
        - 当N为列表时:每个位置使用对应的N值求对应周期内的最低值

        注:
        1.N包含当前k线
        2.若N为0则从第一个有效值开始算起
        3.当N为有效值,但当前的k线数不足N根,按照实际的根数计算
        4.N为空值时,返回空值
        5.N可以是列表
        """
        # 确保输入为numpy数组
        S = My._ensure_array(S)
        
        # 处理二维数组：对每一列分别计算LLV
        if isinstance(S, np.ndarray) and S.ndim == 2:
            T, N_cols = S.shape
            result = np.full((T, N_cols), np.nan, dtype=np.float64)
            for col_idx in range(N_cols):
                s_col = S[:, col_idx]
                try:
                    result[:, col_idx] = My.LLV(s_col, N)
                except Exception:
                    # 如果某一列计算失败，保持NaN
                    pass
            return My._ensure_np_output(result)
        
        # 一维数组的处理逻辑
        S = My._ensure_series(S)
        N = _normalize_window_length(N, x=S.values, min_len=2, default=2, max_len=120)
        len_S = len(S)

        # 处理空值情况
        if N is None or (isinstance(N, (int, float)) and np.isnan(N)):
            return My._ensure_np_output(np.full(len_S, np.nan))

        # 单个数值N的情况
        if isinstance(N, (int, float)):
            N = int(N)

            # 若N为0则从第一个有效值开始算起
            if N == 0:
                # 使用pandas的cummin函数优化累计最小值计算
                # 先处理NaN值,将NaN替换为一个很大的值,计算后再恢复
                mask = S.isna()
                temp_S = S.fillna(np.inf)
                cum_min = temp_S.cummin()
                # 将原始NaN位置和累计计算中没有有效数据的位置设为NaN
                result = cum_min.where(~mask & (cum_min != np.inf), np.nan)
                return My._ensure_np_output(result.values)
            else:
                # 处理单个数值N的情况,使用rolling窗口
                # 当N为有效值,但当前的k线数不足N根,pandas rolling会自动处理
                # min_periods=1确保即使窗口中有一个有效值也会计算
                result = S.rolling(window=N, min_periods=1).min()
                return My._ensure_np_output(result.values)

        # N是列表或数组的情况
        N_array = np.array(N)
        result = np.full(len_S, np.nan)

        # 先找到有效N值的位置
        valid_n_mask = ~np.isnan(N_array)

        # 向量化处理有效N值
        if len(N_array) >= len_S:
            N_array = N_array[:len_S]  # 截断到与S相同长度
        else:
            # 如果N数组长度不足,用NaN填充剩余部分
            padded_N = np.full(len_S, np.nan)
            padded_N[: len(N_array)] = N_array
            N_array = padded_N

        # 转换为整数
        N_int = np.where(valid_n_mask[:len_S], N_array[:len_S].astype(int), 0)

        # 对于每个元素,使用向量化思维计算
        for i in range(len_S):
            if valid_n_mask[i % len(N_array)]:
                n_value = N_int[i]

                if n_value == 0:
                    # 从开始到当前位置的最小值
                    window_data = S.iloc[: i + 1]
                    if window_data.notna().any():
                        result[i] = window_data.min()
                else:
                    # 固定窗口大小的最小值
                    start_idx = max(0, i - n_value + 1)
                    window_data = S.iloc[start_idx : i + 1]
                    if window_data.notna().any():
                        result[i] = window_data.min()

        return My._ensure_np_output(result)

    @staticmethod
    def LV(S, N):
        """求X在N个周期内的最低值(不包括当前K线)
        支持N为单个数值或列表:
        - 当N为数值时:求X在N个周期内的最低值(不包括当前K线)
        - 当N为列表时:每个位置使用对应的N值求对应周期内的最低值(不包括当前K线)

        注:
        1.N不包含当前k线
        2.若N为0则从第一个有效值开始算起(不包括当前K线)
        3.当N为有效值,但当前的k线数不足N根,按照实际的根数计算
        4.N为空值时,返回空值
        5.N可以是列表
        """
        # 确保输入为numpy数组
        S = My._ensure_array(S)
        
        # 处理二维数组：对每一列分别计算LV
        if isinstance(S, np.ndarray) and S.ndim == 2:
            T, N_cols = S.shape
            result = np.full((T, N_cols), np.nan, dtype=np.float64)
            for col_idx in range(N_cols):
                s_col = S[:, col_idx]
                try:
                    result[:, col_idx] = My.LV(s_col, N)
                except Exception:
                    # 如果某一列计算失败，保持NaN
                    pass
            return My._ensure_np_output(result)
        
        # 一维数组的处理逻辑
        S = My._ensure_series(S)
        N = _normalize_window_length(N, x=S.values, min_len=2, default=2, max_len=120)
        series_length = len(S)

        # 单个数值的情况
        if not isinstance(N, (list, np.ndarray)):
            if pd.isna(N) or N is None:
                # N为空值时,返回空值
                return My._ensure_np_output(np.full(series_length, np.nan))
            elif N == 0:
                # 若N为0则从第一个有效值开始算起(不包括当前K线)
                result = np.full(series_length, np.nan)
                for i in range(1, series_length):  # 从第二根K线开始
                    window_values = S.iloc[:i].values  # 不包括当前K线
                    valid_window = window_values[~np.isnan(window_values)]
                    if len(valid_window) > 0:
                        result[i] = np.min(valid_window)
                return My._ensure_np_output(result)
            else:
                # 处理单个数值N的情况,不包括当前K线
                result = np.full(series_length, np.nan)
                for i in range(series_length):
                    n_value = int(N)
                    # 不包括当前K线,所以从i-n_value开始到i-1结束
                    start_idx = max(0, i - n_value)
                    end_idx = i  # 不包括当前K线
                    if start_idx < end_idx:
                        window_values = S.iloc[start_idx:end_idx].values
                        valid_window = window_values[~np.isnan(window_values)]
                        if len(valid_window) > 0:
                            result[i] = np.min(valid_window)
                return My._ensure_np_output(result)
        else:
            # 如果N是列表,进行向量化优化的逐元素计算
            N_array = np.asarray(N)
            result = np.full(series_length, np.nan)

            # 预先处理N数组,确保长度匹配
            if len(N_array) < series_length:
                # 如果N数组长度不足,用最后一个值填充
                last_N = N_array[-1] if len(N_array) > 0 else np.nan
                extended_N = np.full(series_length, last_N)
                extended_N[: len(N_array)] = N_array
                N_array = extended_N
            elif len(N_array) > series_length:
                # 如果N数组过长,截断
                N_array = N_array[:series_length]

            # 向量化处理N=0的特殊情况(不包括当前K线)
            zero_mask = (N_array == 0) & ~np.isnan(N_array)
            if np.any(zero_mask):
                zero_indices = np.where(zero_mask)[0]
                for i in zero_indices:
                    if i > 0:  # 第一根K线无法计算(没有历史数据)
                        window_values = S.iloc[:i].values  # 不包括当前K线
                        valid_window = window_values[~np.isnan(window_values)]
                        if len(valid_window) > 0:
                            result[i] = np.min(valid_window)

            # 处理非0且非空的N值(不包括当前K线)
            valid_mask = (N_array != 0) & ~np.isnan(N_array) & ~zero_mask
            if np.any(valid_mask):
                valid_indices = np.where(valid_mask)[0]

                # 预处理S为NumPy数组以加速访问
                S_values = S.values

                for i in valid_indices:
                    n_value = int(N_array[i])
                    # 不包括当前K线,所以从i-n_value开始到i-1结束
                    start_idx = max(0, i - n_value)
                    end_idx = i  # 不包括当前K线
                    if start_idx < end_idx:
                        window_values = S_values[start_idx:end_idx]
                        # 过滤NaN值并计算最小值
                        valid_window = window_values[~np.isnan(window_values)]
                        if len(valid_window) > 0:
                            result[i] = np.min(valid_window)

            return My._ensure_np_output(result)

    @staticmethod
    def HHVBARS(S, N):
        """求N周期内S最高值到当前周期数, 返回序列"""
        # 确保输入为numpy数组
        S = My._ensure_array(S)
        
        # 处理二维数组：对每一列分别计算HHVBARS
        if isinstance(S, np.ndarray) and S.ndim == 2:
            T, N_cols = S.shape
            result = np.full((T, N_cols), np.nan, dtype=np.float64)
            for col_idx in range(N_cols):
                s_col = S[:, col_idx]
                try:
                    result[:, col_idx] = My.HHVBARS(s_col, N)
                except Exception:
                    # 如果某一列计算失败，保持NaN
                    pass
            return My._ensure_np_output(result)
        
        # 一维数组的处理逻辑
        S = My._ensure_series(S)
        N = _normalize_window_length(N, x=S.values, min_len=2, default=2, max_len=120)
        return My._ensure_np_output(
            S.rolling(N).apply(lambda x: np.argmax(x[::-1]) if len(x) > 0 else 0, raw=True).values
        )

    @staticmethod
    def LLVBARS(S, N):
        """求N周期内S最低值到当前周期数, 返回序列"""
        # 确保输入为numpy数组
        S = My._ensure_array(S)
        
        # 处理二维数组：对每一列分别计算LLVBARS
        if isinstance(S, np.ndarray) and S.ndim == 2:
            T, N_cols = S.shape
            result = np.full((T, N_cols), np.nan, dtype=np.float64)
            for col_idx in range(N_cols):
                s_col = S[:, col_idx]
                try:
                    result[:, col_idx] = My.LLVBARS(s_col, N)
                except Exception:
                    # 如果某一列计算失败，保持NaN
                    pass
            return My._ensure_np_output(result)
        
        # 一维数组的处理逻辑
        S = My._ensure_series(S)
        N = _normalize_window_length(N, x=S.values, min_len=2, default=2, max_len=120)
        return My._ensure_np_output(
            S.rolling(N).apply(lambda x: np.argmin(x[::-1]) if len(x) > 0 else 0, raw=True).values
        )

    @staticmethod
    def MA(S, N):
        """求序列的N日简单移动平均值,返回序列"""
        # 确保输入为numpy数组
        S = My._ensure_array(S)
        
        # 处理二维数组：对每一列分别计算MA
        if isinstance(S, np.ndarray) and S.ndim == 2:
            T, N_cols = S.shape
            result = np.full((T, N_cols), np.nan, dtype=np.float64)
            for col_idx in range(N_cols):
                s_col = S[:, col_idx]
                try:
                    result[:, col_idx] = My.MA(s_col, N)
                except Exception:
                    # 如果某一列计算失败，保持NaN
                    pass
            return My._ensure_np_output(result)
        
        # 一维数组的处理逻辑
        S = My._ensure_series(S)
        N = _normalize_window_length(N, x=S.values, min_len=2, default=2, max_len=120)
        # 确保N是整数,如果是浮点数则四舍五入
        N = int(round(N))
        return My._ensure_np_output(S.rolling(N).mean().values)

    @staticmethod
    def EMA(S, N):
        """指数移动平均,为了精度 S>4*N EMA至少需要120周期 alpha=2/(span+1)"""
        # 确保输入为numpy数组
        S = My._ensure_array(S)
        
        # 处理二维数组：对每一列分别计算EMA
        if isinstance(S, np.ndarray) and S.ndim == 2:
            T, N_cols = S.shape
            result = np.full((T, N_cols), np.nan, dtype=np.float64)
            for col_idx in range(N_cols):
                s_col = S[:, col_idx]
                try:
                    result[:, col_idx] = My.EMA(s_col, N)
                except Exception:
                    # 如果某一列计算失败，保持NaN
                    pass
            return My._ensure_np_output(result)
        
        # 一维数组的处理逻辑
        S = My._ensure_series(S)
        N = _normalize_window_length(N, x=S.values, min_len=2, default=2, max_len=120)
        return My._ensure_np_output(S.ewm(span=N, adjust=False).mean().values)

    @staticmethod
    def SMA(S, N, M=1):
        """中国式的SMA,至少需要120周期才精确 (雪球180周期) alpha=1/(1+com)"""
        # 确保输入为numpy数组
        S = My._ensure_array(S)
        
        # 处理二维数组：对每一列分别计算SMA
        if isinstance(S, np.ndarray) and S.ndim == 2:
            T, N_cols = S.shape
            result = np.full((T, N_cols), np.nan, dtype=np.float64)
            for col_idx in range(N_cols):
                s_col = S[:, col_idx]
                M_col = M[:, col_idx] if isinstance(M, np.ndarray) and M.ndim == 2 else M
                try:
                    result[:, col_idx] = My.SMA(s_col, N, M_col)
                except Exception:
                    # 如果某一列计算失败，保持NaN
                    pass
            return My._ensure_np_output(result)
        
        # 一维数组的处理逻辑
        S = My._ensure_series(S)
        N = _normalize_window_length(N, x=S.values, min_len=2, default=2, max_len=120)
        M = _normalize_window_length(M, x=S.values, min_len=1e-6, default=1e-6, max_len=N,return_type="float")
        return My._ensure_np_output(S.ewm(alpha=M / N, adjust=False).mean().values)

    @staticmethod
    def WMA(S, N):
        """通达信S序列的N日加权移动平均 Yn = (1*X1+2*X2+3*X3+...+n*Xn)/(1+2+3+...+Xn)"""
        # 确保输入为numpy数组
        S = My._ensure_array(S)
        
        # 处理二维数组：对每一列分别计算WMA
        if isinstance(S, np.ndarray) and S.ndim == 2:
            T, N_cols = S.shape
            result = np.full((T, N_cols), np.nan, dtype=np.float64)
            for col_idx in range(N_cols):
                s_col = S[:, col_idx]
                try:
                    result[:, col_idx] = My.WMA(s_col, N)
                except Exception:
                    # 如果某一列计算失败，保持NaN
                    pass
            return My._ensure_np_output(result)
        
        # 一维数组的处理逻辑
        S = My._ensure_series(S)
        N = _normalize_window_length(N, x=S.values, min_len=2, default=2, max_len=120)
        return My._ensure_np_output(
            S.rolling(N)
            .apply(
                lambda x: x[::-1].cumsum().sum() * 2 / N / (N + 1) if len(x) > 0 else np.nan,
                raw=True,
            )
            .values
        )

    @staticmethod
    def DMA(S, alpha_param):
        """求S的动态移动平均,A作平滑因子,必须 0<A<1 (此为核心函数,非指标)"""
        # 确保输入为numpy数组
        S = My._ensure_array(S)

        # 处理二维数组：对每一列分别计算DMA
        if isinstance(S, np.ndarray) and S.ndim == 2:
            T, N_cols = S.shape
            result = np.full((T, N_cols), np.nan, dtype=np.float64)
            for col_idx in range(N_cols):
                s_col = S[:, col_idx]
                alpha_param_col = alpha_param[:, col_idx] if isinstance(alpha_param, np.ndarray) and alpha_param.ndim == 2 else alpha_param
                try:
                    result[:, col_idx] = My.DMA(s_col, alpha_param_col)
                except Exception:
                    # 如果某一列计算失败，保持NaN
                    pass
            return My._ensure_np_output(result)
        
        # 一维数组的处理逻辑
        S = My._ensure_series(S)
        alpha_param = _normalize_window_length(alpha_param, x=S.values, min_len=1e-6, default=1e-6, max_len=1-1e-6,return_type="float")
        if isinstance(alpha_param, (int, float)):
            return My._ensure_np_output(S.ewm(alpha=alpha_param, adjust=False).mean().values)
        alpha_array = np.array(alpha_param)
        alpha_array[np.isnan(alpha_array)] = 1.0
        Y = np.zeros(len(S))
        Y[0] = S.iloc[0]
        for i in range(1, len(S)):
            Y[i] = alpha_array[i] * S.iloc[i] + (1 - alpha_array[i]) * Y[i - 1]
        return My._ensure_np_output(Y)

    @staticmethod
    def AVEDEV(S, N):
        """平均绝对偏差 (序列与其平均值的绝对差的平均值)"""
        # 确保输入为numpy数组
        S = My._ensure_array(S)
        
        # 处理二维数组：对每一列分别计算AVEDEV
        if isinstance(S, np.ndarray) and S.ndim == 2:
            T, N_cols = S.shape
            result = np.full((T, N_cols), np.nan, dtype=np.float64)
            for col_idx in range(N_cols):
                s_col = S[:, col_idx]
                try:
                    result[:, col_idx] = My.AVEDEV(s_col, N)
                except Exception:
                    # 如果某一列计算失败，保持NaN
                    pass
            return My._ensure_np_output(result)
        
        # 一维数组的处理逻辑
        S = My._ensure_series(S)
        N = _normalize_window_length(N, x=S.values, min_len=2, default=2, max_len=120)
        return My._ensure_np_output(
            S.rolling(N).apply(lambda x: (np.abs(x - x.mean())).mean() if len(x) > 0 else np.nan).values
        )

    @staticmethod
    def SLOPE(S, N):
        """返S序列N周期回线性回归斜率"""
        # 确保输入为numpy数组
        S = My._ensure_array(S)
        
        # 处理二维数组：对每一列分别计算SLOPE
        if isinstance(S, np.ndarray) and S.ndim == 2:
            T, N_cols = S.shape
            result = np.full((T, N_cols), np.nan, dtype=np.float64)
            for col_idx in range(N_cols):
                s_col = S[:, col_idx]
                try:
                    result[:, col_idx] = My.SLOPE(s_col, N)
                except Exception:
                    # 如果某一列计算失败，保持NaN
                    pass
            return My._ensure_np_output(result)
        
        # 一维数组的处理逻辑
        S = My._ensure_series(S)
        N = _normalize_window_length(N, x=S.values, min_len=2, default=2, max_len=120)
        return My._ensure_np_output(
            S.rolling(N)
            .apply(
                lambda x: np.polyfit(range(N), x, deg=1)[0] if len(x) == N else np.nan,
                raw=True,
            )
            .values
        )

    @staticmethod
    def FORCAST(S, N):
        """返回S序列N周期回线性回归后的预测值"""
        # 确保输入为numpy数组
        S = My._ensure_array(S)
        
        # 处理二维数组：对每一列分别计算FORCAST
        if isinstance(S, np.ndarray) and S.ndim == 2:
            T, N_cols = S.shape
            result = np.full((T, N_cols), np.nan, dtype=np.float64)
            for col_idx in range(N_cols):
                s_col = S[:, col_idx]
                try:
                    result[:, col_idx] = My.FORCAST(s_col, N)
                except Exception:
                    # 如果某一列计算失败，保持NaN
                    pass
            return My._ensure_np_output(result)
        
        # 一维数组的处理逻辑
        S = My._ensure_series(S)
        N = _normalize_window_length(N, x=S.values, min_len=2, default=2, max_len=120)
        return My._ensure_np_output(
            S.rolling(N)
            .apply(
                lambda x: np.polyval(np.polyfit(range(N), x, deg=1), N - 1) if len(x) == N else np.nan,
                raw=True,
            )
            .values
        )

    # ------------------ 1级:应用层函数(通过0级核心函数实现)------------------
    @staticmethod
    def COUNT(S, N):
        """统计N周期中满足COND条件的周期数
        支持N为单个数值或序列:
        - 当N为数值时:统计N周期中满足条件的周期数
        - 当N为序列时:每个位置使用对应的N值统计对应周期中满足条件的周期数

        注:
        1.N包含当前k线
        2.若N为0则从第一个有效值算起
        3.当N为有效值,但当前的k线数不足N根,从第一根统计到当前周期
        4.N为空值时返回值为空值
        5.N可以为序列
        """
        # 确保输入为numpy数组
        S = My._ensure_array(S)
        
        # 处理二维数组：对每一列分别计算COUNT
        if isinstance(S, np.ndarray) and S.ndim == 2:
            T, N_cols = S.shape
            result = np.full((T, N_cols), np.nan, dtype=np.float64)
            for col_idx in range(N_cols):
                s_col = S[:, col_idx]
                try:
                    result[:, col_idx] = My.COUNT(s_col, N)
                except Exception:
                    # 如果某一列计算失败，保持NaN
                    pass
            return My._ensure_np_output(result)
        
        # 一维数组的处理逻辑
        S = My._ensure_series(S)
        N = _normalize_window_length(N, x=S.values, min_len=0, default=0, max_len=120)

        # 单个数值的情况
        if np.isnan(N) or N is None:
            # N为空值时返回值为空值
            return My._ensure_np_output(np.full(len(S), np.nan))
        elif N == 0:
            # 若N为0则从第一个有效值算起
            result = np.zeros(len(S))
            for i in range(len(S)):
                valid_count = 0
                for j in range(i + 1):
                    if not np.isnan(S.iloc[j]) and S.iloc[j]:
                        valid_count += 1
                result[i] = valid_count
            return My._ensure_np_output(result)
        else:
            # 处理单个数值N的情况
            result = np.zeros(len(S))
            for i in range(len(S)):
                # 当N为有效值,但当前的k线数不足N根,从第一根统计到当前周期
                start_idx = max(0, i - N + 1)
                valid_count = 0
                for j in range(start_idx, i + 1):
                    if not np.isnan(S.iloc[j]) and S.iloc[j]:
                        valid_count += 1
                result[i] = valid_count
            return My._ensure_np_output(result)

    @staticmethod
    def EVERY(S, N):
        """EVERY(CLOSE>O, 5) 最近N天是否都是True"""
        return My._ensure_np_output(My.IF(My.SUM(S, N) == N, True, False))

    @staticmethod
    def EXIST(S, N):
        """EXIST(CLOSE>3010, N=5) n日内是否存在一天大于3000点"""
        return My._ensure_np_output(My.IF(My.SUM(S, N) > 0, True, False))

    @staticmethod
    def FILTER(S, N):
        """FILTER函数,S满足条件后,将其后N周期内的数据置为0, FILTER(C==H,5)"""
        S = My._ensure_array(S).copy()
        for i in range(len(S)):
            if i + 1 + N <= len(S):
                if S[i]:
                    S[i + 1 : i + 1 + N] = 0
        return My._ensure_np_output(S)

    @staticmethod
    def BARSLAST(S):
        """上一次条件成立到当前的周期, BARSLAST(C/REF(C,1)>=1.1) 上一次涨停到今天的天数"""
        S = My._ensure_array(S)
        M = np.concatenate(([0], np.where(S, 1, 0)))
        for i in range(1, len(M)):
            M[i] = 0 if M[i] else M[i - 1] + 1
        return My._ensure_np_output(M[1:])

    @staticmethod
    def BARSLASTCOUNT(S):
        """统计连续满足S条件的周期数"""
        S = My._ensure_array(S)
        rt = np.zeros(len(S) + 1)
        for i in range(len(S)):
            rt[i + 1] = rt[i] + 1 if S[i] else 0
        return My._ensure_np_output(rt[1:])

    @staticmethod
    def CROSS(S1, S2):
        """判断向上金叉穿越 CROSS(MA(C,5),MA(C,10)) 判断向下死叉穿越 CROSS(MA(C,10),MA(C,5))

        CROSS(A,B) 表示A从下方向上穿过B,成立返回1(True),(False)

        注:
        1.满足穿越的条件必须上根k线满足A<=B,当根k线满足A>B才被认定为穿越
        2.支持单个值自动扩展为列表
        """
        S1 = My._ensure_array(S1)
        S2 = My._ensure_array(S2)

        # 如果S1是单个值,扩展为与S2相同长度的列表
        if len(S1) == 1:
            S1 = np.full(len(S2), S1[0])

        # 如果S2是单个值,扩展为与S1相同长度的列表
        if len(S2) == 1:
            S2 = np.full(len(S1), S2[0])

        # 确保两个序列长度相同
        min_len = min(len(S1), len(S2))
        S1 = S1[:min_len]
        S2 = S2[:min_len]

        # 上一根K线满足A<=B,当前K线满足A>B
        return My._ensure_np_output(np.concatenate(([False], (S1 <= S2)[:-1] & (S1 > S2)[1:])))

    @staticmethod
    def VALUEWHEN(S, X):
        """当S条件成立时,取X的当前值,否则取VALUEWHEN的上个成立时的X值"""
        S = My._ensure_array(S)
        X = My._ensure_array(X)
        return My._ensure_np_output(pd.Series(np.where(S, X, np.nan)).ffill().values)

    @staticmethod
    def TOPRANGE(S):
        """TOPRANGE(HIGH)表示当前最高价是近多少周期内最高价的最大值"""
        # 确保输入为numpy数组
        S = My._ensure_array(S)
        
        # 处理二维数组：对每一列分别计算TOPRANGE
        if isinstance(S, np.ndarray) and S.ndim == 2:
            T, N_cols = S.shape
            result = np.full((T, N_cols), np.nan, dtype=np.float64)
            for col_idx in range(N_cols):
                s_col = S[:, col_idx]
                try:
                    result[:, col_idx] = My.TOPRANGE(s_col)
                except Exception:
                    # 如果某一列计算失败，保持NaN
                    pass
            return My._ensure_np_output(result)
        
        # 一维数组的处理逻辑
        S = My._ensure_array(S)
        rt = np.zeros(len(S))
        for i in range(1, len(S)):
            comparison = S[:i] < S[i]
            if np.any(comparison):
                rt[i] = np.argmin(np.flipud(comparison))
            else:
                rt[i] = i
        return My._ensure_np_output(rt.astype("int"))

    @staticmethod
    def LOWRANGE(S):
        """LOWRANGE(LOW)表示当前最低价是近多少周期内最低价的最小值"""
        # 确保输入为numpy数组
        S = My._ensure_array(S)
        
        # 处理二维数组：对每一列分别计算LOWRANGE
        if isinstance(S, np.ndarray) and S.ndim == 2:
            T, N_cols = S.shape
            result = np.full((T, N_cols), np.nan, dtype=np.float64)
            for col_idx in range(N_cols):
                s_col = S[:, col_idx]
                try:
                    result[:, col_idx] = My.LOWRANGE(s_col)
                except Exception:
                    # 如果某一列计算失败，保持NaN
                    pass
            return My._ensure_np_output(result)
        
        # 一维数组的处理逻辑
        S = My._ensure_array(S)
        rt = np.zeros(len(S))
        for i in range(1, len(S)):
            comparison = S[:i] > S[i]
            if np.any(comparison):
                rt[i] = np.argmin(np.flipud(comparison))
            else:
                rt[i] = i
        return My._ensure_np_output(rt.astype("int"))

    # ------------------ 2级:技术指标函数(全部通过0级,1级函数实现) ------------------
    @staticmethod
    def MACD_DIF(CLOSE, SHORT=12, LONG=26):
        """MACD指标,EMA的关系,S取120日,和雪球小数点2位相同"""
        DIF = My.EMA(CLOSE, SHORT) - My.EMA(CLOSE, LONG)
        return My._ensure_np_output(DIF)

    @staticmethod
    def MACD_DEA(CLOSE, SHORT=12, LONG=26, M=9):
        """MACD指标,EMA的关系,S取120日,和雪球小数点2位相同"""
        DIF = My.EMA(CLOSE, SHORT) - My.EMA(CLOSE, LONG)
        DEA = My.EMA(DIF, M)
        return My._ensure_np_output(DEA)

    @staticmethod
    def MACD_MACD(CLOSE, SHORT=12, LONG=26, M=9):
        """MACD指标,EMA的关系,S取120日,和雪球小数点2位相同"""
        DIF = My.EMA(CLOSE, SHORT) - My.EMA(CLOSE, LONG)
        DEA = My.EMA(DIF, M)
        MACD = (DIF - DEA) * 2
        return My._ensure_np_output(MACD)

    @staticmethod
    def KDJ_K(CLOSE, HIGH, LOW, N=9, M1=3):
        """KDJ指标"""
        RSV = (CLOSE - My.LLV(LOW, N)) / (My.HHV(HIGH, N) - My.LLV(LOW, N)) * 100
        K = My.EMA(RSV, (M1 * 2 - 1))
        return My._ensure_np_output(K)

    @staticmethod
    def KDJ_D(CLOSE, HIGH, LOW, N=9, M1=3, M2=3):
        """KDJ指标"""
        RSV = (CLOSE - My.LLV(LOW, N)) / (My.HHV(HIGH, N) - My.LLV(LOW, N)) * 100
        K = My.EMA(RSV, (M1 * 2 - 1))
        D = My.EMA(K, (M2 * 2 - 1))
        return My._ensure_np_output(D)

    @staticmethod
    def KDJ_J(CLOSE, HIGH, LOW, N=9, M1=3, M2=3):
        """KDJ指标"""
        RSV = (CLOSE - My.LLV(LOW, N)) / (My.HHV(HIGH, N) - My.LLV(LOW, N)) * 100
        K = My.EMA(RSV, (M1 * 2 - 1))
        D = My.EMA(K, (M2 * 2 - 1))
        J = K * 3 - D * 2
        return My._ensure_np_output(J)

    @staticmethod
    def RSI(CLOSE, N=14):
        """RSI指标,使用指数移动平均(EMA)计算,和通达信小数点2位相同"""
        # 计算价格变动
        DIF = CLOSE - My.REF(CLOSE, 1)

        # 分离上涨和下跌
        UP = My.MAX(DIF, 0)  # 上涨幅度(负数变为0)
        DOWN = My.MAX(-DIF, 0)  # 下跌幅度(正数变为0)

        # 计算EMA(平滑指数alpha=1/N)
        alpha = 1 / N
        EMA_UP = My.DMA(UP, alpha)  # 使用DMA函数实现EMA
        EMA_DOWN = My.DMA(DOWN, alpha)  # 使用DMA函数实现EMA

        # 计算相对强度RS
        RS = EMA_UP / (EMA_DOWN+1e-10)

        # 计算RSI
        RSI = 100 - (100 / (1 + RS))

        # 处理特殊情况:分母为0时设为50
        RSI = My.IF(EMA_DOWN == 0, 50, RSI)

        return My._ensure_np_output(RSI)

    @staticmethod
    def SAR(HIGH, LOW, N=10, S=2, M=20):
        """
        求抛物转向。 例如SAR(10,2,20)表示计算10日抛物转向,步长为2%,步长极限为20%

        :param HIGH: high序列
        :param LOW: low序列
        :param N: 计算周期
        :param S: 步长
        :param M: 步长极限
        :return: 抛物转向
        """
        HIGH = My._ensure_array(HIGH)
        LOW = My._ensure_array(LOW)
        # 确保输入为numpy数组
        N = _normalize_window_length(N, x=HIGH, min_len=1, default=20, max_len=120)
        
        # 处理二维数组：对每一列分别计算BOLL_LOWER
        if isinstance(HIGH, np.ndarray) and HIGH.ndim == 2:
            T, N_cols = HIGH.shape
            result = np.full((T, N_cols), np.nan, dtype=np.float64)
            for col_idx in range(N_cols):
                high_col = HIGH[:, col_idx]
                low_col = LOW[:, col_idx]
                try:
                    result[:, col_idx] = My.SAR(high_col, low_col, N, S, M)
                except Exception:
                    # 如果某一列计算失败，保持NaN
                    pass
            return My._ensure_np_output(result)
        
        # 一维数组的处理逻辑
        f_step = S / 100
        f_max = M / 100
        length = len(HIGH)

        # 确保有足够的数据
        if length < N:
            return My._ensure_np_output(np.repeat(np.nan, length))

        # 计算前N根K线的最高价和最低价
        high = HIGH[0]
        low = LOW[0]
        for i in range(N):
            if HIGH[i] > high:
                high = HIGH[i]
            if LOW[i] < low:
                low = LOW[i]

        # SAR_LONG = 0, SAR_SHORT = 1
        SAR_LONG = 0
        SAR_SHORT = 1
        position = SAR_LONG  # 初始为多头

        sar_x = np.repeat(np.nan, length)  # type: np.ndarray
        sar_x[N - 1] = low  # 第一个SAR值为最低价

        # 初始化sip（极值）为第一根K线的最高价,af为step/100
        sip = HIGH[0]
        af = f_step
        next_sar = low

        # 从第N根K线（索引N）开始计算
        for i in range(N, length):
            ysip = sip  # 保存上一轮的极值
            item_high = HIGH[i]
            item_low = LOW[i]
            yitem_high = HIGH[i - 1]
            yitem_low = LOW[i - 1]

            if position == SAR_LONG:  # 多头
                if item_low < sar_x[i - 1]:  # 反转为空头
                    position = SAR_SHORT
                    sip = item_low
                    af = f_step
                    next_sar = max(item_high, yitem_high)
                    next_sar = max(next_sar, ysip + af * (sip - ysip))
                else:  # 继续多头
                    position = SAR_LONG
                    if item_high > ysip:  # 创新高
                        sip = item_high
                        af = min(af + f_step, f_max)
                    next_sar = min(item_low, yitem_low)
                    next_sar = min(next_sar, sar_x[i - 1] + af * (sip - sar_x[i - 1]))

            elif position == SAR_SHORT:  # 空头
                if item_high > sar_x[i - 1]:  # 反转为多头
                    position = SAR_LONG
                    sip = item_high
                    af = f_step
                    next_sar = min(item_low, yitem_low)
                    next_sar = min(next_sar, sar_x[i - 1] + af * (sip - ysip))
                else:  # 继续空头
                    position = SAR_SHORT
                    if item_low < ysip:  # 创新低
                        sip = item_low
                        af = min(af + f_step, f_max)
                    next_sar = max(item_high, yitem_high)
                    next_sar = max(next_sar, sar_x[i - 1] + af * (sip - sar_x[i - 1]))

            sar_x[i] = next_sar

        return My._ensure_np_output(sar_x)

    @staticmethod
    def WR(CLOSE, HIGH, LOW, N=10):
        """W&R 威廉指标"""
        WR = (My.HHV(HIGH, N) - CLOSE) / (My.HHV(HIGH, N) - My.LLV(LOW, N)) * 100
        return My._ensure_np_output(WR)

    @staticmethod
    def BIAS(CLOSE, L=6):
        """BIAS乖离率"""
        BIAS = (CLOSE - My.MA(CLOSE, L)) / My.MA(CLOSE, L) * 100
        return My._ensure_np_output(BIAS)

    @staticmethod
    def BOLL_UPPER(CLOSE, N=20, P=2):
        """BOLL指标,布林带上轨"""
        MID = My.MA(CLOSE, N)
        UPPER = MID + My.STD(CLOSE, N, ddof=1) * P
        return My._ensure_np_output(UPPER)

    @staticmethod
    def BOLL_LOWER(CLOSE, N=20, P=2):
        """BOLL指标,布林带下轨"""
        MID = My.MA(CLOSE, N)
        LOWER = MID - My.STD(CLOSE, N, ddof=1) * P
        return My._ensure_np_output(LOWER)

    @staticmethod
    def PSY(CLOSE, N=12):
        """PSY心理线指标"""
        PSY = My.COUNT(CLOSE > My.REF(CLOSE, 1), N) / N * 100
        return My._ensure_np_output(PSY)

    @staticmethod
    def CCI(CLOSE, HIGH, LOW, N=14):
        """CCI顺势指标"""
        CLOSE = My._ensure_array(CLOSE)
        HIGH = My._ensure_array(HIGH)
        LOW = My._ensure_array(LOW)
        TP = (HIGH + LOW + CLOSE) / 3
        return My._ensure_np_output((TP - My.MA(TP, N)) / (0.015 * My.AVEDEV(TP, N)))

    @staticmethod
    def TR(CLOSE, HIGH, LOW):
        """真实波动"""
        HIGH = My._ensure_array(HIGH)
        LOW = My._ensure_array(LOW)
        TR = My.MAX(
            My.MAX((HIGH - LOW), My.ABS(My.REF(CLOSE, 1) - HIGH)),
            My.ABS(My.REF(CLOSE, 1) - LOW),
        )
        return My._ensure_np_output(TR)

    @staticmethod
    def ATR(CLOSE, HIGH, LOW, N=20):
        """真实波动N日平均值"""
        HIGH = My._ensure_array(HIGH)
        LOW = My._ensure_array(LOW)
        TR = My.MAX(
            My.MAX((HIGH - LOW), My.ABS(My.REF(CLOSE, 1) - HIGH)),
            My.ABS(My.REF(CLOSE, 1) - LOW),
        )
        return My._ensure_np_output(My.MA(TR, N))

    @staticmethod
    def BBI(CLOSE, M1=3, M2=6, M3=12, M4=20):
        """BBI多空指标"""
        return My._ensure_np_output((My.MA(CLOSE, M1) + My.MA(CLOSE, M2) + My.MA(CLOSE, M3) + My.MA(CLOSE, M4)) / 4)

    @staticmethod
    def DMI(CLOSE, HIGH, LOW, M1=14, M2=6):
        """动向指标:结果和同花顺,通达信完全一致"""
        HIGH = My._ensure_array(HIGH)
        LOW = My._ensure_array(LOW)
        TR = My.SUM(
            My.MAX(
                My.MAX(HIGH - LOW, My.ABS(HIGH - My.REF(CLOSE, 1))),
                My.ABS(LOW - My.REF(CLOSE, 1)),
            ),
            M1,
        )
        HD = HIGH - My.REF(HIGH, 1)
        LD = My.REF(LOW, 1) - LOW
        DMP = My.SUM(My.IF((HD > 0) & (HD > LD), HD, 0), M1)
        DMM = My.SUM(My.IF((LD > 0) & (LD > HD), LD, 0), M1)
        PDI = DMP * 100 / TR
        MDI = DMM * 100 / TR
        ADX = My.MA(My.ABS(MDI - PDI) / (PDI + MDI) * 100, M2)
        ADXR = (ADX + My.REF(ADX, M2)) / 2
        return My._ensure_np_output((PDI, MDI, ADX, ADXR))

    @staticmethod
    def ADX(CLOSE, HIGH, LOW, M1=14, M2=6):
        """ADX指标:结果和同花顺,通达信完全一致"""
        CLOSE = My._ensure_array(CLOSE)
        HIGH = My._ensure_array(HIGH)
        LOW = My._ensure_array(LOW)
        TR = My.SUM(
            My.MAX(
                My.MAX(HIGH - LOW, My.ABS(HIGH - My.REF(CLOSE, 1))),
                My.ABS(LOW - My.REF(CLOSE, 1)),
            ),
            M1,
        )
        HD = HIGH - My.REF(HIGH, 1)
        LD = My.REF(LOW, 1) - LOW
        DMP = My.SUM(My.IF((HD > 0) & (HD > LD), HD, 0), M1)
        DMM = My.SUM(My.IF((LD > 0) & (LD > HD), LD, 0), M1)
        PDI = DMP * 100 / TR
        MDI = DMM * 100 / TR
        ADX = My.MA(My.ABS(MDI - PDI) / (PDI + MDI + 1e-10) * 100, M2)
        return My._ensure_np_output(ADX)

    @staticmethod
    def TRIX(CLOSE, M1=12):
        """三重指数平滑平均线"""
        TR = My.EMA(My.EMA(My.EMA(CLOSE, M1), M1), M1)
        TRIX = (TR - My.REF(TR, 1)) / My.REF(TR, 1) * 100
        return My._ensure_np_output(TRIX)

    @staticmethod
    def VR(CLOSE, VOL, M1=26):
        """VR容量比率"""
        LC = My.REF(CLOSE, 1)
        return My._ensure_np_output(
            My.SUM(My.IF(CLOSE > LC, VOL, 0), M1) / My.SUM(My.IF(CLOSE <= LC, VOL, 0), M1) * 100
        )

    @staticmethod
    def CR(CLOSE, HIGH, LOW, N=20):
        """CR价格动量指标"""
        HIGH = My._ensure_array(HIGH)
        LOW = My._ensure_array(LOW)
        CLOSE = My._ensure_array(CLOSE)
        MID = My.REF(HIGH + LOW + CLOSE, 1) / 3
        return My._ensure_np_output(My.SUM(My.MAX(0, HIGH - MID), N) / My.SUM(My.MAX(0, MID - LOW), N) * 100)

    @staticmethod
    def EMV(HIGH, LOW, VOL, N=14):
        """简易波动指标"""
        HIGH = My._ensure_array(HIGH)
        LOW = My._ensure_array(LOW)
        VOL = My._ensure_array(VOL)
        VOLUME = My.MA(VOL, N) / VOL
        MID = 100 * (HIGH + LOW - My.REF(HIGH + LOW, 1)) / (HIGH + LOW)
        EMV = My.MA(MID * VOLUME * (HIGH - LOW) / My.MA(HIGH - LOW, N), N)
        return My._ensure_np_output(EMV)

    @staticmethod
    def DPO(CLOSE, M1=20, M2=10):
        """区间震荡线"""
        CLOSE = My._ensure_array(CLOSE)
        DPO = CLOSE - My.REF(My.MA(CLOSE, M1), M2)
        return My._ensure_np_output(DPO)

    @staticmethod
    def BRAR_AR(OPEN, HIGH, LOW, M1=26):
        """BRAR-ARBR 情绪指标"""
        HIGH = My._ensure_array(HIGH)
        LOW = My._ensure_array(LOW)
        OPEN = My._ensure_array(OPEN)
        AR = My.SUM(HIGH - OPEN, M1) / My.SUM(OPEN - LOW, M1) * 100
        return My._ensure_np_output(AR)

    @staticmethod
    def BRAR_BR(OPEN, CLOSE, HIGH, LOW, M1=26):
        """BRAR-ARBR 情绪指标"""
        HIGH = My._ensure_array(HIGH)
        LOW = My._ensure_array(LOW)
        OPEN = My._ensure_array(OPEN)
        BR = My.SUM(My.MAX(0, HIGH - My.REF(CLOSE, 1)), M1) / My.SUM(My.MAX(0, My.REF(CLOSE, 1) - LOW), M1) * 100
        return My._ensure_np_output(BR)

    @staticmethod
    def DIFMA(CLOSE, N1=10, N2=50, M=10):
        """平行线差指标"""
        DIF = My.MA(CLOSE, N1) - My.MA(CLOSE, N2)
        DIFMA = My.MA(DIF, M)  # 通达信指标叫DMA 同花顺叫新DMA
        return My._ensure_np_output(DIFMA)

    @staticmethod
    def MTM(CLOSE, N=12):
        """动量指标"""
        CLOSE = My._ensure_array(CLOSE)
        MTM = CLOSE - My.REF(CLOSE, N)
        return My._ensure_np_output(MTM)

    @staticmethod
    def MASS(HIGH, LOW, N1=9, N2=25):
        """梅斯线"""
        HIGH = My._ensure_array(HIGH)
        LOW = My._ensure_array(LOW)
        MASS = My.SUM(My.MA(HIGH - LOW, N1) / My.MA(My.MA(HIGH - LOW, N1), N1), N2)
        return My._ensure_np_output(MASS)

    @staticmethod
    def ROC(CLOSE, N=12):
        """变动率指标"""
        CLOSE = My._ensure_array(CLOSE)
        ROC = 100 * (CLOSE - My.REF(CLOSE, N)) / My.REF(CLOSE, N)
        return My._ensure_np_output(ROC)

    @staticmethod
    def OBV(CLOSE, VOL):
        """能量潮指标"""
        CLOSE = My._ensure_array(CLOSE)
        VOL = My._ensure_array(VOL)
        OBV = My.SUM(My.IF(CLOSE>My.REF(CLOSE,1),VOL,My.IF(CLOSE<My.REF(CLOSE,1),-VOL,0)),0)
        return My._ensure_np_output(OBV)

    @staticmethod
    def MFI(CLOSE, HIGH, LOW, VOL, N=14):
        """MFI指标是成交量的RSI指标"""
        HIGH = My._ensure_array(HIGH)
        LOW = My._ensure_array(LOW)
        CLOSE = My._ensure_array(CLOSE)
        VOL = My._ensure_array(VOL)
        TYP = (HIGH + LOW + CLOSE) / 3
        V1 = My.SUM(My.IF(TYP > My.REF(TYP, 1), TYP * VOL, 0), N) / My.SUM(My.IF(TYP < My.REF(TYP, 1), TYP * VOL, 0), N)
        return My._ensure_np_output(100 - (100 / (1 + V1)))

    @staticmethod
    def ASI(OPEN, CLOSE, HIGH, LOW, M1=26):
        """振动升降指标"""
        HIGH = My._ensure_array(HIGH)
        LOW = My._ensure_array(LOW)
        OPEN = My._ensure_array(OPEN)
        CLOSE = My._ensure_array(CLOSE)
        LC = My.REF(CLOSE, 1)
        AA = My.ABS(HIGH - LC)
        BB = My.ABS(LOW - LC)
        CC = My.ABS(HIGH - My.REF(LOW, 1))
        DD = My.ABS(LC - My.REF(OPEN, 1))
        R = My.IF(
            (AA > BB) & (AA > CC),
            AA + BB / 2 + DD / 4,
            My.IF((BB > CC) & (BB > AA), BB + AA / 2 + DD / 4, CC + DD / 4),
        )
        X = CLOSE - LC + (CLOSE - OPEN) / 2 + LC - My.REF(OPEN, 1)
        SI = 16 * X / R * My.MAX(AA, BB)
        ASI = My.SUM(SI, M1)
        return My._ensure_np_output(ASI)


# ## 测试计算因子

# ### 自动计算因子（基于转换器）

# #### 转换器代码
FIELD_ALIASES = {
    "close": "close",
    "open": "open",
    "high": "high",
    "low": "low",
    "volume": "volume",
    "vol": "volume",
    "open_interest": "open_interest",
    "oi": "open_interest",
    "openinterest": "open_interest",
}


class NumberNode:
    def __init__(self, raw: str):
        self.raw = raw


class IdentifierNode:
    def __init__(self, name: str):
        self.name = name


class CallNode:
    def __init__(self, name: str, args: List[Any]):
        self.name = name
        self.args = args


TOKEN_RE = re.compile(
    r"""
    \s*(
        (?P<number>-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?) |
        (?P<ident>[A-Za-z_][A-Za-z0-9_]*) |
        (?P<lpar>\() |
        (?P<rpar>\)) |
        (?P<comma>,)
    )
    """,
    re.VERBOSE,
)


class FormulaParser:
    def __init__(self, text: str):
        self.tokens = self._tokenize(text)
        self.pos = 0

    @staticmethod
    def _tokenize(text: str) -> List[Tuple[str, str]]:
        tokens: List[Tuple[str, str]] = []
        idx = 0
        n = len(text)
        while idx < n:
            match = TOKEN_RE.match(text, idx)
            if not match:
                bad = text[idx : idx + 20]
                raise ValueError(f"无法解析公式，位置 {idx} 附近: {bad!r}")
            idx = match.end()
            for kind in ("number", "ident", "lpar", "rpar", "comma"):
                value = match.group(kind)
                if value is not None:
                    tokens.append((kind, value))
                    break
        return tokens

    def parse(self) -> Any:
        node = self._parse_expr()
        if self.pos != len(self.tokens):
            kind, value = self.tokens[self.pos]
            raise ValueError(f"多余的 token: {kind}={value!r}")
        return node

    def _peek(self) -> Optional[Tuple[str, str]]:
        if self.pos >= len(self.tokens):
            return None
        return self.tokens[self.pos]

    def _consume(self, expected_kind: str) -> str:
        token = self._peek()
        if token is None:
            raise ValueError(f"期望 {expected_kind}，但遇到结尾")
        kind, value = token
        if kind != expected_kind:
            raise ValueError(f"期望 {expected_kind}，但得到 {kind}={value!r}")
        self.pos += 1
        return value

    def _parse_expr(self) -> Any:
        token = self._peek()
        if token is None:
            raise ValueError("空公式")
        kind, value = token
        if kind == "number":
            self.pos += 1
            return NumberNode(raw=value)
        if kind == "ident":
            self.pos += 1
            ident_name = value
            next_tok = self._peek()
            if next_tok and next_tok[0] == "lpar":
                self._consume("lpar")
                args: List[Any] = []
                if self._peek() and self._peek()[0] != "rpar":
                    args.append(self._parse_expr())
                    while self._peek() and self._peek()[0] == "comma":
                        self._consume("comma")
                        args.append(self._parse_expr())
                self._consume("rpar")
                return CallNode(name=ident_name, args=args)
            return IdentifierNode(name=ident_name)
        raise ValueError(f"不支持的 token: {kind}={value!r}")


def _norm_name(name: str) -> str:
    return name.strip().lower()


def _field_from_identifier(name: str) -> Optional[str]:
    return FIELD_ALIASES.get(_norm_name(name))


def _field_from_param_name(name: str) -> Optional[str]:
    return FIELD_ALIASES.get(_norm_name(name))


def _get_method_signature(my_cls: Any, func_name: str) -> Optional[Sequence[inspect.Parameter]]:
    candidates = [func_name, func_name.upper(), func_name.lower()]
    method = None
    for cand in candidates:
        if hasattr(my_cls, cand):
            method = getattr(my_cls, cand)
            break
    if method is None:
        return None
    try:
        sig = inspect.signature(method)
    except (TypeError, ValueError):
        return None
    params = [
        p
        for p in sig.parameters.values()
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    return params


class FormulaConverter:
    def __init__(self, my_cls: Any, my_name: str = "My", x_dict_name: str = "X_dict"):
        self.my_cls = my_cls
        self.my_name = my_name
        self.x_dict_name = x_dict_name
        self._base_ref = f'{x_dict_name}["close"]'

    def convert(self, formula: str) -> str:
        ast = FormulaParser(formula).parse()
        return self._node_to_code(ast)

    def _field_code(self, field: str) -> str:
        return f'{self.x_dict_name}["{field}"]'

    def _const_code(self, raw: str) -> str:
        return f"np.full_like({self._base_ref}, {raw}, dtype=np.float64)"

    def _node_to_code(self, node: Any) -> str:
        if isinstance(node, NumberNode):
            return self._const_code(node.raw)

        if isinstance(node, IdentifierNode):
            maybe_field = _field_from_identifier(node.name)
            if maybe_field is not None:
                return self._field_code(maybe_field)
            return node.name

        if isinstance(node, CallNode):
            return self._call_to_code(node)

        raise TypeError(f"未知节点类型: {type(node)!r}")

    def _call_to_code(self, node: CallNode) -> str:
        method_name = node.name.upper()
        params = _get_method_signature(self.my_cls, method_name)
        converted_args = [self._node_to_code(arg) for arg in node.args]

        if not params:
            return f"{self.my_name}.{method_name}({', '.join(converted_args)})"

        market_positions: List[int] = []
        for i, p in enumerate(params):
            if _field_from_param_name(p.name) is not None:
                market_positions.append(i)

        non_market_positions = [i for i in range(len(params)) if i not in market_positions]
        use_non_market_only_mode = len(market_positions) > 0 and len(converted_args) <= len(non_market_positions)

        final_args: List[str] = []
        if use_non_market_only_mode:
            user_idx = 0
            for i, p in enumerate(params):
                mapped = _field_from_param_name(p.name)
                if mapped is not None:
                    final_args.append(self._field_code(mapped))
                else:
                    if user_idx < len(converted_args):
                        final_args.append(converted_args[user_idx])
                        user_idx += 1
                    elif p.default is inspect._empty:
                        raise ValueError(f"{method_name} 缺少必要参数: {p.name}")
                    else:
                        final_args.append(self._const_code(repr(p.default)))
        else:
            user_idx = 0
            for p in params:
                if user_idx < len(converted_args):
                    final_args.append(converted_args[user_idx])
                    user_idx += 1
                else:
                    mapped = _field_from_param_name(p.name)
                    if mapped is not None:
                        final_args.append(self._field_code(mapped))
                    elif p.default is inspect._empty:
                        raise ValueError(f"{method_name} 缺少必要参数: {p.name}")
                    else:
                        final_args.append(self._const_code(repr(p.default)))

            # 允许多传参数（例如可变函数或未来扩展），按原样追加
            if user_idx < len(converted_args):
                final_args.extend(converted_args[user_idx:])

        return f"{self.my_name}.{method_name}({', '.join(final_args)})"


def convert_formula(formula: str, my_cls: Any, x_dict_name: str = "X_dict", my_name: str = "My") -> str:
    return FormulaConverter(my_cls=my_cls, my_name=my_name, x_dict_name=x_dict_name).convert(formula)


def build_factor_expressions(formula, my_cls: Any, my_name: str = "My"):
    if isinstance(formula, list):
        factor_exprs = [FormulaConverter(my_cls=my_cls, my_name=my_name, x_dict_name="X_dict").convert(f) for f in formula]
        factor_now_exprs = [FormulaConverter(my_cls=my_cls, my_name=my_name, x_dict_name="X_dict_now").convert(f) for f in formula]
        factor_test_exprs = [FormulaConverter(my_cls=my_cls, my_name=my_name, x_dict_name="X_dict_test").convert(f) for f in formula]
        return {
            "factor": factor_exprs,
            "factor_now": factor_now_exprs,
            "factor_test": factor_test_exprs,
        }
    else:
        return {
            "factor": FormulaConverter(my_cls=my_cls, my_name=my_name, x_dict_name="X_dict").convert(formula),
            "factor_now": FormulaConverter(my_cls=my_cls, my_name=my_name, x_dict_name="X_dict_now").convert(formula),
            "factor_test": FormulaConverter(my_cls=my_cls, my_name=my_name, x_dict_name="X_dict_test").convert(formula),
        }


def eval_factors(
    formula,
    my_cls: Any,
    X_dict: Dict[str, np.ndarray],
    X_dict_now: Dict[str, np.ndarray],
    X_dict_test: Dict[str, np.ndarray],
):
    exprs = build_factor_expressions(formula=formula, my_cls=my_cls, my_name="My")
    env = {
        "np": np,
        "My": my_cls,
        "X_dict": X_dict,
        "X_dict_now": X_dict_now,
        "X_dict_test": X_dict_test,
    }
    if isinstance(formula, list):
        return {name: [eval(expr, env, {}) for expr in expr_list] for name, expr_list in exprs.items()}
    else:
        return {name: eval(expr, env, {}) for name, expr in exprs.items()}# #### 使用转换器计算因子
# formula = "CORR(TS_ZSCORE(high, LV(high, 5)), EMV(85), open_interest)"
# formula = "ADD(WR(1.0), SIN(MUL(LLV(HV(MIN(EMV(22), RANK(BIAS(23), EMV(5.048163479178479))), 11), KDJ_J(4, 63, 11)), SLOPE(RSI(36), 88.67576321773176))))"
# formula = "RANK(WR(3), FORCAST(EMA(MA(BRAR_BR(39.67400187583232), PSY(61)), SCALE(RANK(BOLL_LOWER(51.92058608481411, 105.33912028012675), CCI(89)), MUL(TS_ZSCORE(TOPRANGE(TS_RANK(TR(), ATR(95))), MAX(CR(18), KDJ_J(10, 74, 79))), RANK(DIFMA(101, 105.70288284157309, 54), DIFMA(67, 119.59247708006433, 110))))), RANK(TS_ZSCORE(SCALE(RANK_SUB(MACD_MACD(103.1367287638528, 49, 71), TS_ZSCORE(ATR(74), DPO(5, 63)), TS_ZSCORE(LLV(CR(73.59636181482315), 60.49431526716149), SIGNEDPOWER(KDJ_D(95.0270735386948, 111.14688602129436, 106.9185473465797)))), SCALE(MAX(ASI(41.10188909712552), TS_ZSCORE(MACD_DIF(90, 11), MFI(27))), RANK_SUB(BBI(100.80402744834933, 17, 12.732093098580961, 29), BIAS(54), KDJ_D(49, 80.24928039078695, 90.62123976587635)))), SCALE(SMA(SIN(SAR(63, 46.70113766183343, 62.152669951821125)), 4, 5.0), SCALE(VR(15), ADX(40.069828069434735, 90.18823201047609)))), INV(RANK(SCALE(RANK(CORR(KDJ_J(89, 57.906667999650715, 41.60920604942375), RSI(22.009126046643384), MACD_DEA(69, 4, 65)), ABS(CR(90.02718123376458))), EMV(65)), CCI(78.69447786554845))))))"
# formula = "CORR(CORR(SIN(TOPRANGE(PSY(75))), TAN(MASS(188.91227410655506, 50)), ROC(5.943201414914385)), ATR(64.67582157232408), BIAS(15.493019300191346))"
# 1) 先看转换结果字符串
exprs = build_factor_expressions(formula, My)
if isinstance(formula, list):
    print(f"共 {len(formula)} 个因子")
    # for i, expr in enumerate(exprs["factor"]):
        # print(f"因子{i+1}: {expr}") # 已注释
else:
    print(exprs["factor"])
# print(exprs["factor_now"])
# print(exprs["factor_test"])

# 2) 直接计算三套因子
factors = eval_factors(formula, My, X_dict, X_dict_now, X_dict_test)
factor = factors["factor"]
factor_now = factors["factor_now"]
factor_test = factors["factor_test"]

# ### 手动计算因子
# CLOSE = X_dict["close"]
# OPEN = X_dict["open"]
# HIGH = X_dict["high"]
# LOW = X_dict["low"]
# VOL = X_dict["volume"]
# OP = X_dict["open_interest"]
# My.OBV(CLOSE,VOL)


# My.TRIX(CLOSE)
# My.SMA(My.TRIX(46, 9.588828746845888), 18, 1.0)
# My.MACD_DEA(CLOSE)
# My.SAR(HIGH,LOW,23, 19.96131810922959, 37)
# My.DMA(97.21232422178215, 0.18313079833257395)
# My.DMA(My.BOLL_UPPER(CLOSE,104, 93.80594845698839), 0.31409153787133304)
# My.LOWRANGE(CLOSE)
# print(CLOSE)
# CLOSE/My.REF(CLOSE,1)
# CLOSE*2
# CLOSE*CLOSE
# My.RANK(CLOSE,2)
# My.LN(My.SMA(My.WR(CLOSE,HIGH,LOW,52), 14, 1.0))
# My.LV(My.DMA(OPEN, 0.3759937090202221), 5)
# My.EMV(64, 54)
# My.RSI(CLOSE,117)
# My.SMA(CLOSE, 12, 1.0)
# My.CORR(VOL, np.full_like(VOL, 48.84140077805907, dtype=np.float64), My.TS_ZSCORE(My.MTM(CLOSE,64), My.HHVBARS(OP, 6)))
# My.DIV(LOW, My.AVEDEV(My.LOWRANGE(My.ADD(My.RANK_SUB(My.TS_ZSCORE(VOL, HIGH), My.SAR(HIGH,LOW,56, 79.02856415970575, 86), My.REF(CLOSE, 14)), My.ADX(CLOSE,HIGH,LOW,4, 2))), 1))
# My.AVEDEV(My.LOWRANGE(My.ADD(My.RANK_SUB(My.TS_ZSCORE(VOL, HIGH), My.SAR(HIGH,LOW,56, 79.02856415970575, 86), My.REF(CLOSE, 14)), My.ADX(CLOSE,HIGH,LOW,4, 2))), 1)
# My.ADX(CLOSE,HIGH,LOW,4, 2).shape
# My.ADD(My.RANK_SUB(My.TS_ZSCORE(VOL, HIGH), My.SAR(HIGH,LOW,56, 79.02856415970575, 86), My.REF(CLOSE, 14)), My.ADX(CLOSE,HIGH,LOW,4, 2)).shape
# My.LOWRANGE(My.ADD(My.RANK_SUB(My.TS_ZSCORE(VOL, HIGH), My.SAR(HIGH,LOW,56, 79.02856415970575, 86), My.REF(CLOSE, 14)), My.ADX(CLOSE,HIGH,LOW,4, 2))).shape
# factor = My.ADX(CLOSE,HIGH,LOW,2, 95.94944786765016)
# factor = My.TR(CLOSE,HIGH,LOW)
# My.CORR(My.MASS(HIGH,LOW,29, 48), np.full_like(VOL, 29.75129871880469, dtype=np.float64), 88.72440404389046)
# My.DMA(np.full_like(VOL, 7.62933260259474, dtype=np.float64), 0.4289059471044482)
# My.SIN(My.INV(My.TAN(My.EMA(My.INV(np.full_like(VOL, 43.36675801249388)), 18))))
# factor
# factor = My.MACD_DEA(CLOSE,38.25130077271199, 37.66879219628618, 83)
AUTO_ON = True
if not AUTO_ON:
    # 因子表达式
    formula = "WR(close, high, low, 2)"
    factor = My.WR(X_dict["close"],X_dict["high"],X_dict["low"],2)
    factor_now = My.WR(X_dict_now["close"],X_dict_now["high"],X_dict_now["low"],2)
    factor_test = My.WR(X_dict_test["close"],X_dict_test["high"],X_dict_test["low"],2)# ## 测试计算IC、IR值

def panel_to_long_factor_df(factor_arr, y_arr, pivot_dict, split_name):
    factor_df = pd.DataFrame(
        np.asarray(factor_arr, dtype=np.float64),
        index=pivot_dict['close'].index,
        columns=pivot_dict['close'].columns,
    )
    ret_df = pd.DataFrame(
        np.asarray(y_arr, dtype=np.float64),
        index=pivot_dict['future_return'].index,
        columns=pivot_dict['future_return'].columns,
    )

    factor_long = factor_df.stack(future_stack=True).rename('factor')
    ret_long = ret_df.stack(future_stack=True).rename('future_return')
    out = pd.concat([factor_long, ret_long], axis=1).reset_index()
    out.columns = ['time', 'code', 'factor', 'future_return']
    out['split'] = split_name
    out = out.sort_values(['time', 'code']).reset_index(drop=True)
    return out


def build_quantile_report(long_df, quantiles=5):
    dfq = long_df.copy()
    dfq = dfq[np.isfinite(dfq['factor']) & np.isfinite(dfq['future_return'])].copy()
    if dfq.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    def _quantile_by_time(s):
        if s.notna().sum() < quantiles:
            return pd.Series(np.nan, index=s.index)
        try:
            return pd.qcut(s.rank(method='first'), quantiles, labels=False) + 1
        except Exception:
            return pd.Series(np.nan, index=s.index)

    dfq['quantile'] = dfq.groupby('time')['factor'].transform(_quantile_by_time)
    dfq = dfq.dropna(subset=['quantile']).copy()
    if dfq.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    dfq['quantile'] = dfq['quantile'].astype(int)
    quantile_ret = dfq.groupby(['time', 'quantile'])['future_return'].mean().unstack()
    long_short = quantile_ret[quantile_ret.columns.max()] - quantile_ret[quantile_ret.columns.min()]
    return quantile_ret, long_short


def ic_curve_from_long(long_df):
    rows = []
    for t, grp in long_df.groupby('time'):
        grp = grp[np.isfinite(grp['factor']) & np.isfinite(grp['future_return'])]
        if len(grp) < int(globals().get("MIN_CROSS_SECTION_COUNT", 2)):
            continue
        rows.append((t, grp['factor'].rank().corr(grp['future_return'].rank())))
    if not rows:
        return pd.Series(dtype=float)
    s = pd.Series(dict(rows)).sort_index()
    s.name = 'rank_ic'
    return s


def calc_ic_stats(factor, y):
    """计算单个样本区间的 rankIC / ICIR 统计量"""
    pred = np.asarray(factor, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if pred.shape != y.shape:
        return {
            "fitness": -np.inf,
            "mean_ic": -np.inf,
            "icir": -np.inf,
            "valid_ts": 0,
            "total_ts": int(y.shape[0]) if y.ndim >= 1 else 0,
            "ic_series": np.array([], dtype=np.float64),
        }

    ic_values = []
    min_count = int(globals().get("MIN_CROSS_SECTION_COUNT", 2))

    for t in range(y.shape[0]):
        pred_t = pred[t].ravel()
        y_t = y[t].ravel()
        mask = np.isfinite(pred_t) & np.isfinite(y_t)

        if np.sum(mask) < min_count:
            continue

        pred_rank = _rankdata(pred_t[mask], method="average")
        y_rank = _rankdata(y_t[mask], method="average")
        corr = np.corrcoef(pred_rank, y_rank)[0, 1]

        if np.isfinite(corr):
            ic_values.append(corr)

    if len(ic_values) < 1:
        return {
            "fitness": -np.inf,
            "mean_ic": -np.inf,
            "icir": -np.inf,
            "valid_ts": 0,
            "total_ts": y.shape[0],
            "ic_series": np.array([], dtype=np.float64),
        }

    ic_values = np.asarray(ic_values, dtype=np.float64)
    mean_ic = float(np.mean(ic_values))
    icir = float(mean_ic / (np.std(ic_values) + 1e-8))
    a = float(globals().get("IC_WEIGHT_A", 0.4))
    b = float(globals().get("IC_WEIGHT_B", 0.6))
    fitness = a * mean_ic + b * np.tanh(icir)

    return {
        "fitness": fitness,
        "mean_ic": mean_ic,
        "icir": icir,
        "valid_ts": int(len(ic_values)),
        "total_ts": int(y.shape[0]),
        "ic_series": ic_values,
    }


def fitness_func(factor, y, factor_test=None, y_test=None, return_details=False):
    """训练集/测试集综合适应度；分析展示时建议配合 calc_ic_stats 单独看各分区结果。"""
    train_stats = calc_ic_stats(factor, y)

    if factor_test is None or y_test is None:
        return train_stats if return_details else train_stats["fitness"]

    test_stats = calc_ic_stats(factor_test, y_test)

    base_train = train_stats["fitness"]
    base_test = test_stats["fitness"]
    w_train = float(globals().get("FITNESS_W_TRAIN", 0.3))
    w_test = float(globals().get("FITNESS_W_TEST", 0.7))
    fitness_scheme = globals().get("FITNESS_SCHEME", "B")
    overfit_lambda = float(globals().get("FITNESS_OVERFIT_LAMBDA", 0.2))

    if not np.isfinite(base_train) or not np.isfinite(base_test):
        final_fitness = -np.inf
    elif fitness_scheme == "A":
        final_fitness = w_train * base_train + w_test * base_test
    else:
        final_fitness = (
            w_train * base_train
            + w_test * base_test
            - overfit_lambda * abs(base_train - base_test)
        )

    if return_details:
        return {
            "fitness": final_fitness,
            "mean_ic": train_stats["mean_ic"],
            "icir": train_stats["icir"],
            "valid_ts": train_stats["valid_ts"],
            "total_ts": train_stats["total_ts"],
            "mean_ic_test": test_stats["mean_ic"],
            "icir_test": test_stats["icir"],
            "valid_ts_test": test_stats["valid_ts"],
            "total_ts_test": test_stats["total_ts"],
            "ic_series_train": train_stats["ic_series"],
            "ic_series_test": test_stats["ic_series"],
        }

    return final_fitness


def analyze_single_factor(factor_val, factor_now_val, factor_test_val, y, y_now, y_test, formula_expr, pivoted, pivoted_now, pivoted_test):
    temp1 = fitness_func(factor_val, y, factor_test=factor_test_val, y_test=y_test, return_details=True)
    temp_train = calc_ic_stats(factor_val, y)
    temp_test = calc_ic_stats(factor_test_val, y_test)
    temp_now = calc_ic_stats(factor_now_val, y_now)

    print(f"\n因子表达式: {formula_expr}")
    print(f"品种合约: {SYMBOLS}")
    print(f"预测周期: {Y_PERIOD}")
    summary_df = pd.DataFrame([
        [temp_train['mean_ic'], temp_train['icir'], temp_train['valid_ts']],
        [temp_test['mean_ic'], temp_test['icir'], temp_test['valid_ts']],
        [temp_now['mean_ic'], temp_now['icir'], temp_now['valid_ts']],
    ], columns=['IC', 'IR', '有效时间点'], index=[
        f'训练集({BEGIN_TIME}~{END_TIME})',
        f'测试集({BEGIN_TIME_TEST}~{END_TIME_TEST})',
        f'真实集({BEGIN_TIME_NOW}~{END_TIME_NOW})',
    ])
    print(summary_df)

    print(f"综合适应度（训练/测试）: {temp1['fitness']:.6f}")

    return {
        "formula": formula_expr,
        "temp_train": temp_train,
        "temp_test": temp_test,
        "temp_now": temp_now,
        "temp1": temp1,
    }


if isinstance(formula, list):
    all_results = []
    all_analysis = []
    for i, (f_expr, f_val, f_now_val, f_test_val) in enumerate(zip(formula, factor, factor_now, factor_test)):
        print(f"\n{'='*50}")
        print(f"计算第 {i+1}/{len(formula)} 个因子")
        print(f"{'='*50}")
        result = analyze_single_factor(f_val, f_now_val, f_test_val, y, y_now, y_test, f_expr, pivoted, pivoted_now, pivoted_test)
        all_results.append(result)

        analysis_train = panel_to_long_factor_df(f_val, y, pivoted, 'train')
        analysis_test = panel_to_long_factor_df(f_test_val, y_test, pivoted_test, 'test')
        analysis_now = panel_to_long_factor_df(f_now_val, y_now, pivoted_now, 'now')
        
        all_analysis.append({
            'factor_index': i + 1,
            'formula': f_expr,
            'train': analysis_train,
            'test': analysis_test,
            'now': analysis_now
        })

        for split_name, long_df in [('训练集', analysis_train), ('测试集', analysis_test), ('真实集', analysis_now)]:
            ic_s = ic_curve_from_long(long_df)
            qret, ls = build_quantile_report(long_df, quantiles=QUANTILES)

            print(f"\n===== {split_name} =====")
            print(f"样本数: {len(long_df)}")
            print(f"有效IC时间点: {ic_s.shape[0]}")
            if not ic_s.empty:
                print(f"平均IC: {ic_s.mean():.6f}, ICIR: {ic_s.mean() / (ic_s.std() + 1e-8):.6f}")
            if not qret.empty:
                print(qret.mean().to_frame('各分位平均未来收益').T)
                print(ls.describe().to_frame('多空组合统计'))

    print(f"\n{'='*50}")
    print("多因子汇总")
    print(f"{'='*50}")
    for i, result in enumerate(all_results):
        print(f"\n因子{i+1}: {result['formula']}")
        print(f"  训练集 IC: {result['temp_train']['mean_ic']:.6f}, IR: {result['temp_train']['icir']:.6f}")
        print(f"  测试集 IC: {result['temp_test']['mean_ic']:.6f}, IR: {result['temp_test']['icir']:.6f}")
        print(f"  真实集 IC: {result['temp_now']['mean_ic']:.6f}, IR: {result['temp_now']['icir']:.6f}")
else:
    result = analyze_single_factor(factor, factor_now, factor_test, y, y_now, y_test, formula, pivoted, pivoted_now, pivoted_test)

    analysis_train = panel_to_long_factor_df(factor, y, pivoted, 'train')
    analysis_test = panel_to_long_factor_df(factor_test, y_test, pivoted_test, 'test')
    analysis_now = panel_to_long_factor_df(factor_now, y_now, pivoted_now, 'now')

    for split_name, long_df in [('训练集', analysis_train), ('测试集', analysis_test), ('真实集', analysis_now)]:
        ic_s = ic_curve_from_long(long_df)
        qret, ls = build_quantile_report(long_df, quantiles=QUANTILES)

        print(f"\n===== {split_name} =====")
        print(f"样本数: {len(long_df)}")
        print(f"有效IC时间点: {ic_s.shape[0]}")
        if not ic_s.empty:
            print(f"平均IC: {ic_s.mean():.6f}, ICIR: {ic_s.mean() / (ic_s.std() + 1e-8):.6f}")
        if not qret.empty:
            print(qret.mean().to_frame('各分位平均未来收益').T)
            print(ls.describe().to_frame('多空组合统计'))


# ## 因子加入本地库

if SAVE_FACTOR:
    output_csv = "多因子分析库.csv"

    if isinstance(formula, list):
        rows_to_save = []
        for i, result in enumerate(all_results):
            row = {
                "入库时间": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "因子表达式": result["formula"],
                "品种选择": SELECTED_SECTOR,
                "品种合约": str(SYMBOLS),
                "预测周期": Y_PERIOD,
                "训练集时间": f"{BEGIN_TIME}~{END_TIME}",
                "训练集IC": result["temp_train"]['mean_ic'],
                "训练集IR": result["temp_train"]['icir'],
                "训练集有效时间点": result["temp_train"]['valid_ts'],
                "测试集时间": f"{BEGIN_TIME_TEST}~{END_TIME_TEST}",
                "测试集IC": result["temp_test"]['mean_ic'],
                "测试集IR": result["temp_test"]['icir'],
                "测试集有效时间点": result["temp_test"]['valid_ts'],
                "真实集时间": f"{BEGIN_TIME_NOW}~{END_TIME_NOW}",
                "真实集IC": result["temp_now"]['mean_ic'],
                "真实集IR": result["temp_now"]['icir'],
                "真实集有效时间点": result["temp_now"]['valid_ts'],
                "按IC分类": "IC有效因子" if result["temp_now"]['mean_ic'] >= 0.05 else "IC无效因子",
                "按IR分类": "IR稳定因子" if result["temp_now"]['icir'] >= 0.3 else "IR不稳定因子"
            }
            rows_to_save.append(row)

        duplicate_cols = [
            "因子表达式",
            "品种合约",
            "预测周期",
            "训练集时间",
            "测试集时间",
            "真实集时间"
        ]

        if os.path.exists(output_csv):
            try:
                df_base = pd.read_csv(output_csv, dtype=str)
            except Exception:
                df_base = pd.DataFrame()
            if not df_base.empty:
                for row in rows_to_save:
                    row_df = pd.DataFrame([{col: str(row[col]) for col in duplicate_cols}])
                    df_cmp = df_base[duplicate_cols].astype(str)
                    matches = (df_cmp == row_df.iloc[0]).all(axis=1)
                    if not matches.any():
                        df_base = pd.concat([df_base, pd.DataFrame([row])], ignore_index=True)
                df_base.to_csv(output_csv, index=False, encoding="utf-8-sig")
                print(f"本次 {len(rows_to_save)} 个因子结果已追加到 {output_csv}")
            else:
                df_base = pd.DataFrame(rows_to_save)
                df_base.to_csv(output_csv, index=False, encoding="utf-8-sig")
                print(f"本次 {len(rows_to_save)} 个因子结果已保存到 {output_csv}")
        else:
            df_base = pd.DataFrame(rows_to_save)
            df_base.to_csv(output_csv, index=False, encoding="utf-8-sig")
            print(f"本次 {len(rows_to_save)} 个因子结果已保存到 {output_csv}")
    else:
        row = {
            "入库时间": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "因子表达式": formula,
            "品种选择": SELECTED_SECTOR,
            "品种合约": str(SYMBOLS),
            "预测周期": Y_PERIOD,
            "训练集时间": f"{BEGIN_TIME}~{END_TIME}",
            "训练集IC": result["temp_train"]['mean_ic'],
            "训练集IR": result["temp_train"]['icir'],
            "训练集有效时间点": result["temp_train"]['valid_ts'],
            "测试集时间": f"{BEGIN_TIME_TEST}~{END_TIME_TEST}",
            "测试集IC": result["temp_test"]['mean_ic'],
            "测试集IR": result["temp_test"]['icir'],
            "测试集有效时间点": result["temp_test"]['valid_ts'],
            "真实集时间": f"{BEGIN_TIME_NOW}~{END_TIME_NOW}",
            "真实集IC": result["temp_now"]['mean_ic'],
            "真实集IR": result["temp_now"]['icir'],
            "真实集有效时间点": result["temp_now"]['valid_ts']
        }

        duplicate_cols = [
            "因子表达式",
            "品种合约",
            "预测周期",
            "训练集时间",
            "测试集时间",
            "真实集时间"
        ]

        if os.path.exists(output_csv):
            try:
                df_base = pd.read_csv(output_csv, dtype=str)
            except Exception:
                df_base = pd.DataFrame()
            is_duplicate = False
            if not df_base.empty:
                row_df = pd.DataFrame([{col: str(row[col]) for col in duplicate_cols}])
                df_cmp = df_base[duplicate_cols].astype(str)
                matches = (df_cmp == row_df.iloc[0]).all(axis=1)
                if matches.any():
                    is_duplicate = True
            if is_duplicate:
                print(f"结果已存在于 {output_csv}, 不追加")
            else:
                df_base = pd.concat([df_base, pd.DataFrame([row])], ignore_index=True)
                df_base.to_csv(output_csv, index=False, encoding="utf-8-sig")
                print(f"本次因子结果已追加到 {output_csv}")
        else:
            df_base = pd.DataFrame([row])
            df_base.to_csv(output_csv, index=False, encoding="utf-8-sig")
            print(f"本次因子结果已保存到 {output_csv}")


for split_name, long_df in [('训练集', analysis_train), ('测试集', analysis_test), ('真实集', analysis_now)]:
    ic_s = ic_curve_from_long(long_df)
    qret, ls = build_quantile_report(long_df, quantiles=QUANTILES)

    print(f"\n===== {split_name} =====")
    print(f"样本数: {len(long_df)}")
    print(f"有效IC时间点: {ic_s.shape[0]}")
    if not ic_s.empty:
        print(f"平均IC: {ic_s.mean():.6f}, ICIR: {ic_s.mean() / (ic_s.std() + 1e-8):.6f}")
    if not qret.empty:
        print(qret.mean().to_frame('各分位平均未来收益').T)
        print(ls.describe().to_frame('多空组合统计'))


# ## 绘制 Rolling IC 与多空累计收益
output_dir = "多因子分析可视化"
os.makedirs(output_dir, exist_ok=True)

if isinstance(formula, list):
    for analysis in all_analysis:
        factor_idx = analysis['factor_index']
        factor_expr = analysis['formula']
        safe_expr = "".join(c if c.isalnum() or c in ('_', '-', '(' , ')', '+', '*', '/') else '_' for c in factor_expr)[:50]
        
        plt.figure(figsize=(12, 4))
        for label, long_df in [('训练集', analysis['train']), ('测试集', analysis['test']), ('真实集', analysis['now'])]:
            ic_s = ic_curve_from_long(long_df)
            if not ic_s.empty:
                ic_s.rolling(20, min_periods=5).mean().plot(label=f'{label} Rolling IC(20)')
        plt.axhline(0.0, linestyle='--')
        plt.title(f'因子{factor_idx} Rolling Rank IC\n{factor_expr}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/因子{factor_idx}_Rolling_IC.png", dpi=150)
        plt.close()

        plt.figure(figsize=(12, 4))
        for label, long_df in [('训练集', analysis['train']), ('测试集', analysis['test']), ('真实集', analysis['now'])]:
            _, ls = build_quantile_report(long_df, quantiles=QUANTILES)
            if not ls.empty:
                ls.fillna(0).cumsum().plot(label=f'{label} 多空累计收益')
        plt.axhline(0.0, linestyle='--')
        plt.title(f'因子{factor_idx} 多空累计收益\n{factor_expr}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/因子{factor_idx}_多空累计收益.png", dpi=150)
        plt.close()
else:
    plt.figure(figsize=(12, 4))
    for label, long_df in [('训练集', analysis_train), ('测试集', analysis_test), ('真实集', analysis_now)]:
        ic_s = ic_curve_from_long(long_df)
        if not ic_s.empty:
            ic_s.rolling(20, min_periods=5).mean().plot(label=f'{label} Rolling IC(20)')
    plt.axhline(0.0, linestyle='--')
    plt.title('Rolling Rank IC')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/单因子_Rolling_IC.png", dpi=150)
    plt.close()

    plt.figure(figsize=(12, 4))
    for label, long_df in [('训练集', analysis_train), ('测试集', analysis_test), ('真实集', analysis_now)]:
        _, ls = build_quantile_report(long_df, quantiles=QUANTILES)
        if not ls.empty:
            ls.fillna(0).cumsum().plot(label=f'{label} 多空累计收益')
    plt.axhline(0.0, linestyle='--')
    plt.title('Top-Bottom Quantile Long-Short Cumulative Return')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/单因子_多空累计收益.png", dpi=150)
    plt.close()




# ==================== LightGBM 多因子合成模型 ====================
# 仅在多因子列表（公式个数 > 1）时执行
if isinstance(formula, list) and len(formula) > 1 and USE_LIGHTGBM:
    print("\n" + "="*60)
    print("开始 LightGBM 多因子合成模型训练与评估")
    print("="*60)

    # ---------- 1. 获取已计算好的因子数组 ----------
    factor_list_train = factors["factor"]      # list of (T_train, N) arrays
    factor_list_test  = factors["factor_test"] # list of (T_test, N) arrays
    factor_list_now   = factors["factor_now"]  # list of (T_now, N) arrays
    K = len(factor_list_train)                 # 因子个数

    # 面板形状
    T_train, N = factor_list_train[0].shape
    T_test, _  = factor_list_test[0].shape
    T_now, _   = factor_list_now[0].shape

    # ---------- 2. 堆叠为三维数组 (T, N, K) ----------
    X_train_raw = np.stack(factor_list_train, axis=2)
    X_test_raw  = np.stack(factor_list_test,  axis=2)
    X_now_raw   = np.stack(factor_list_now,   axis=2)

    # ---------- 3. 横截面标准化 (每个时间点，每个因子独立标准化) ----------
    def cross_sectional_standardize(X):
        """X: (T, N, K) -> 返回相同形状，NaN保持原位"""
        X_std = np.full_like(X, np.nan, dtype=np.float64)
        for t in range(X.shape[0]):
            for k in range(X.shape[2]):
                col = X[t, :, k]
                mask = np.isfinite(col)
                if mask.sum() < 2:
                    continue
                mean = col[mask].mean()
                std  = col[mask].std()
                if std > 1e-8:
                    X_std[t, mask, k] = (col[mask] - mean) / std
                else:
                    X_std[t, mask, k] = 0.0
        return X_std

    X_train = cross_sectional_standardize(X_train_raw)
    X_test  = cross_sectional_standardize(X_test_raw)
    X_now   = cross_sectional_standardize(X_now_raw)

    # ---------- 4. 展平为 DataFrame，保留时间和资产标签 ----------
    times_train = pivoted['close'].index
    assets_train = pivoted['close'].columns
    times_test  = pivoted_test['close'].index
    assets_test  = pivoted_test['close'].columns
    times_now   = pivoted_now['close'].index
    assets_now   = pivoted_now['close'].columns

    def flatten_panel(X, y_arr, times, assets):
        """将 (T,N,K) 和 (T,N) 展平为 DataFrame，跳过目标为 NaN 的行"""
        T, N, K = X.shape
        records = []
        for t in range(T):
            for n in range(N):
                if np.isnan(y_arr[t, n]):
                    continue
                record = {
                    'time': times[t],
                    'asset': assets[n],
                    'target': y_arr[t, n]
                }
                for k in range(K):
                    record[f'factor_{k}'] = X[t, n, k]
                records.append(record)
        return pd.DataFrame(records)

    df_train = flatten_panel(X_train, y, times_train, assets_train)
    df_test  = flatten_panel(X_test,  y_test, times_test, assets_test)
    df_now   = flatten_panel(X_now,   y_now,  times_now,  assets_now)

    feature_cols = [f'factor_{k}' for k in range(K)]

    # ---------- 5. 配置 LightGBM 参数（可在此处修改） ----------
    lgb_params = {
        'n_estimators': 200,
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'min_child_samples': 20,
        'random_state': 42,
        'verbosity': -1
    }

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(df_train[feature_cols], df_train['target'])

    # ---------- 6. 预测 ----------
    df_train['pred'] = model.predict(df_train[feature_cols])
    df_test['pred']  = model.predict(df_test[feature_cols])
    df_now['pred']   = model.predict(df_now[feature_cols])

    # ---------- 7. 将预测值还原为面板格式 (T, N) ----------
    def predictions_to_panel(df_pred, times, assets, target_shape):
        pred_panel = np.full(target_shape, np.nan, dtype=np.float64)
        time_to_idx = {t: i for i, t in enumerate(times)}
        asset_to_idx = {a: j for j, a in enumerate(assets)}
        for _, row in df_pred.iterrows():
            i = time_to_idx.get(row['time'])
            j = asset_to_idx.get(row['asset'])
            if i is not None and j is not None:
                pred_panel[i, j] = row['pred']
        return pred_panel

    pred_panel_train = predictions_to_panel(df_train, times_train, assets_train, y.shape)
    pred_panel_test  = predictions_to_panel(df_test,  times_test,  assets_test,  y_test.shape)
    pred_panel_now   = predictions_to_panel(df_now,   times_now,   assets_now,   y_now.shape)

    # ---------- 8. 评估合成因子（复用已有评估函数） ----------
    def evaluate_synthetic_factor(pred_panel, y_arr, pivoted_dict, name):
        long_df = panel_to_long_factor_df(pred_panel, y_arr, pivoted_dict, name)
        ic_s = ic_curve_from_long(long_df)
        qret, ls = build_quantile_report(long_df, quantiles=QUANTILES)
        print(f"\n===== {name} =====")
        print(f"有效IC时间点: {ic_s.shape[0]}")
        if not ic_s.empty:
            mean_ic = ic_s.mean()
            icir = mean_ic / (ic_s.std() + 1e-8)
            print(f"平均IC: {mean_ic:.6f}, ICIR: {icir:.6f}")
        if not qret.empty:
            print("各分位平均未来收益:")
            print(qret.mean().to_frame('收益').T)
            print("多空组合统计:")
            print(ls.describe().to_frame('统计'))
        return ic_s, qret, ls

    print("\n--- LightGBM 合成因子评估 ---")
    ic_train, _, _ = evaluate_synthetic_factor(pred_panel_train, y, pivoted, "训练集")
    ic_test,  _, _ = evaluate_synthetic_factor(pred_panel_test,  y_test, pivoted_test, "测试集")
    ic_now,   _, _ = evaluate_synthetic_factor(pred_panel_now,   y_now,  pivoted_now,  "真实集")

    # ---------- 9. 特征重要性 ----------
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    print("\n特征重要性（按降序排列，对应原始公式列表）:")
    for i in indices:
        expr = formula[i] if i < len(formula) else f"因子{i}"
        print(f"  {expr}: {importance[i]}")

    # ---------- 10. 绘图 ----------
    # Rolling IC 曲线
    plt.figure(figsize=(12, 4))
    if not ic_train.empty:
        ic_train.rolling(20, min_periods=5).mean().plot(label='训练集 Rolling IC(20)')
    if not ic_test.empty:
        ic_test.rolling(20, min_periods=5).mean().plot(label='测试集 Rolling IC(20)')
    if not ic_now.empty:
        ic_now.rolling(20, min_periods=5).mean().plot(label='真实集 Rolling IC(20)')
    plt.axhline(0.0, linestyle='--')
    plt.title('LightGBM 合成因子 Rolling Rank IC')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/LightGBM_Rolling_IC.png", dpi=150)
    plt.close()

    # 多空累计收益曲线
    plt.figure(figsize=(12, 4))
    for name, pred_panel, y_arr, piv in [
        ('训练集', pred_panel_train, y, pivoted),
        ('测试集', pred_panel_test,  y_test, pivoted_test),
        ('真实集', pred_panel_now,   y_now,  pivoted_now)
    ]:
        long_df = panel_to_long_factor_df(pred_panel, y_arr, piv, name)
        _, ls = build_quantile_report(long_df, quantiles=QUANTILES)
        if not ls.empty:
            ls.fillna(0).cumsum().plot(label=f'{name} 多空累计收益')
    plt.axhline(0.0, linestyle='--')
    plt.title('LightGBM 合成因子 Top-Bottom 多空累计收益')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/LightGBM_多空累计收益.png", dpi=150)
    plt.close()

    # 真实集预测值与真实值散点图
    if not df_now.empty:
        plt.figure(figsize=(6, 6))
        plt.scatter(df_now['pred'], df_now['target'], alpha=0.3, s=1)
        plt.xlabel('预测值')
        plt.ylabel('真实未来收益')
        plt.title('真实集：预测 vs 真实')
        plt.grid(True)
        plt.savefig(f"{output_dir}/LightGBM_预测vs真实.png", dpi=150)
        plt.close()

    # ========== LightGBM 扩展可视化 ==========
    try:
        import shap
        SHAP_AVAILABLE = True
    except ImportError:
        SHAP_AVAILABLE = False
        print("警告: shap 库未安装，跳过 SHAP 全局解释图。")

    # ---------- 1. 特征重要性条形图 (基于 LightGBM 内置) ----------
    plt.figure(figsize=(10, 6))
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    sorted_names = [formula[i] for i in indices]
    sorted_imp = importance[indices]
    plt.barh(range(len(sorted_names)), sorted_imp[::-1], color='steelblue')
    plt.yticks(range(len(sorted_names)), sorted_names[::-1])
    plt.xlabel('特征重要性 (Gain)')
    plt.title('LightGBM 特征重要性条形图')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'LightGBM_特征重要性.png'), dpi=150)
    plt.close()

    # ---------- 2. 残差分析图 (使用真实集) ----------
    if not df_now.empty:
        residuals = df_now['target'] - df_now['pred']
        residuals = residuals.dropna()
        if len(residuals) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            # 直方图 + KDE
            axes[0].hist(residuals, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
            residuals.plot.kde(ax=axes[0], color='red', linewidth=2)
            axes[0].set_xlabel('残差')
            axes[0].set_ylabel('密度')
            axes[0].set_title('残差分布 (真实集)')
            axes[0].axvline(x=0, color='black', linestyle='--')
            # Q-Q 图
            from scipy import stats
            stats.probplot(residuals, dist="norm", plot=axes[1])
            axes[1].set_title('Q-Q 图 (真实集)')
            axes[1].get_lines()[0].set_color('steelblue')
            axes[1].get_lines()[1].set_color('red')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'LightGBM_残差分析.png'), dpi=150)
            plt.close()

    # ---------- 3. 时间序列预测误差 (按时间聚合 MAE) ----------
    if not df_now.empty:
        # 按时间计算每个时间点的 MAE
        time_mae = df_now.groupby('time').apply(
            lambda g: np.mean(np.abs(g['target'] - g['pred'])), include_groups=False
        ).reset_index(name='MAE')
        time_mae = time_mae.sort_values('time')
        plt.figure(figsize=(12, 5))
        plt.plot(time_mae['time'], time_mae['MAE'], marker='o', linestyle='-', linewidth=1, markersize=3)
        plt.axhline(y=time_mae['MAE'].mean(), color='red', linestyle='--', label=f"平均 MAE = {time_mae['MAE'].mean():.5f}")
        plt.xlabel('时间')
        plt.ylabel('平均绝对误差 (MAE)')
        plt.title('真实集上预测误差随时间变化')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'LightGBM_时间序列误差.png'), dpi=150)
        plt.close()

    # ---------- 4. 分位数预测表现 (预测值分组下的实际收益) ----------
    if not df_now.empty:
        df_q = df_now[['pred', 'target']].dropna().copy()
        if len(df_q) > 0:
            # 使用全局分位数 (例如 10 组，或沿用 QUANTILES)
            n_quantiles = QUANTILES if QUANTILES >= 2 else 5
            try:
                df_q['pred_quantile'] = pd.qcut(df_q['pred'], q=n_quantiles, labels=False) + 1
            except ValueError:
                # 分位数边缘重复时使用 rank 方法
                df_q['pred_quantile'] = pd.qcut(df_q['pred'].rank(method='first'), q=n_quantiles, labels=False) + 1
            quantile_perf = df_q.groupby('pred_quantile')['target'].mean()
            plt.figure(figsize=(8, 5))
            plt.bar(quantile_perf.index, quantile_perf.values, color='teal', alpha=0.7)
            plt.axhline(y=0, color='black', linestyle='--')
            plt.xlabel('预测值分位数')
            plt.ylabel('实际未来收益均值')
            plt.title(f'分位数预测表现 (真实集, {n_quantiles} 组)')
            plt.xticks(quantile_perf.index)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'LightGBM_分位数预测表现.png'), dpi=150)
            plt.close()

    # ---------- 5. SHAP 全局解释 (使用真实集或测试集) ----------
    if SHAP_AVAILABLE and not df_now.empty:
        try:
            # 为避免内存过大，随机采样部分样本计算 SHAP 值
            sample_size = min(5000, len(df_now))
            df_sample = df_now.sample(n=sample_size, random_state=42)
            X_sample = df_sample[feature_cols]
            # 创建 TreeExplainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            # 全局特征重要性：平均绝对 SHAP 值
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            # 条形图
            plt.figure(figsize=(10, 6))
            sorted_idx = np.argsort(mean_abs_shap)[::-1]
            sorted_names_shap = [formula[i] for i in sorted_idx]
            sorted_shap = mean_abs_shap[sorted_idx]
            plt.barh(range(len(sorted_names_shap)), sorted_shap[::-1], color='darkorange')
            plt.yticks(range(len(sorted_names_shap)), sorted_names_shap[::-1])
            plt.xlabel('平均 |SHAP 值|')
            plt.title('SHAP 全局特征重要性 (真实集)')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'LightGBM_SHAP全局特征重要性.png'), dpi=150)
            plt.close()
            # 可选：summary plot (若需要更丰富的可视化)
            # shap.summary_plot(shap_values, X_sample, feature_names=formula, show=False)
            # plt.savefig(os.path.join(output_dir, 'lgb_shap_summary.png'), dpi=150, bbox_inches='tight')
            # plt.close()
        except Exception as e:
            print(f"SHAP 可视化失败: {e}")
    elif not SHAP_AVAILABLE:
        print("跳过 SHAP 图 (未安装 shap 库)")

    print("\nLightGBM 多因子合成模型分析完成。")






# ==================== Elastic Net 多因子合成模型 ====================
# 仅在多因子列表（公式个数 > 1）时执行
if isinstance(formula, list) and len(formula) > 1 and USE_ELASTIC_NET:
    try:
        from sklearn.linear_model import ElasticNet
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        from sklearn.impute import SimpleImputer
    except ImportError:
        raise ImportError("请先安装 scikit-learn: pip install scikit-learn")

    print("\n" + "="*60)
    print("开始 Elastic Net 多因子合成模型训练与评估")
    print("="*60)

    # ---------- 1. 获取已计算好的因子数组 ----------
    factor_list_train = factors["factor"]      # list of (T_train, N) arrays
    factor_list_test  = factors["factor_test"] # list of (T_test, N) arrays
    factor_list_now   = factors["factor_now"]  # list of (T_now, N) arrays
    K = len(factor_list_train)                 # 因子个数

    # 面板形状
    T_train, N = factor_list_train[0].shape
    T_test, _  = factor_list_test[0].shape
    T_now, _   = factor_list_now[0].shape

    # ---------- 2. 堆叠为三维数组 (T, N, K) ----------
    X_train_raw = np.stack(factor_list_train, axis=2)
    X_test_raw  = np.stack(factor_list_test,  axis=2)
    X_now_raw   = np.stack(factor_list_now,   axis=2)

    # ---------- 3. 横截面标准化 (每个时间点，每个因子独立标准化) ----------
    def cross_sectional_standardize(X):
        X_std = np.full_like(X, np.nan, dtype=np.float64)
        for t in range(X.shape[0]):
            for k in range(X.shape[2]):
                col = X[t, :, k]
                mask = np.isfinite(col)
                if mask.sum() < 2:
                    continue
                mean = col[mask].mean()
                std  = col[mask].std()
                if std > 1e-8:
                    X_std[t, mask, k] = (col[mask] - mean) / std
                else:
                    X_std[t, mask, k] = 0.0
        return X_std

    X_train_std = cross_sectional_standardize(X_train_raw)
    X_test_std  = cross_sectional_standardize(X_test_raw)
    X_now_std   = cross_sectional_standardize(X_now_raw)

    # ---------- 4. 处理缺失值：将 NaN 填充为 0（标准化后 NaN 代表无有效计算值） ----------
    # 使用 SimpleImputer 可更灵活，这里直接 fillna(0) 更简单
    def fill_nan_with_zero(X):
        X_filled = X.copy()
        X_filled[np.isnan(X_filled)] = 0.0
        return X_filled

    X_train = fill_nan_with_zero(X_train_std)
    X_test  = fill_nan_with_zero(X_test_std)
    X_now   = fill_nan_with_zero(X_now_std)

    # ---------- 5. 展平为 DataFrame，保留时间和资产标签 ----------
    times_train = pivoted['close'].index
    assets_train = pivoted['close'].columns
    times_test  = pivoted_test['close'].index
    assets_test  = pivoted_test['close'].columns
    times_now   = pivoted_now['close'].index
    assets_now   = pivoted_now['close'].columns

    def flatten_panel(X, y_arr, times, assets):
        T, N, K = X.shape
        records = []
        for t in range(T):
            for n in range(N):
                if np.isnan(y_arr[t, n]):
                    continue
                record = {
                    'time': times[t],
                    'asset': assets[n],
                    'target': y_arr[t, n]
                }
                for k in range(K):
                    record[f'factor_{k}'] = X[t, n, k]
                records.append(record)
        return pd.DataFrame(records)

    df_train = flatten_panel(X_train, y, times_train, assets_train)
    df_test  = flatten_panel(X_test,  y_test, times_test, assets_test)
    df_now   = flatten_panel(X_now,   y_now,  times_now,  assets_now)

    feature_cols = [f'factor_{k}' for k in range(K)]

    # ---------- 6. 配置 Elastic Net 参数（可在此处修改） ----------
    # alpha: 正则化强度，l1_ratio: L1 比例 (0 = 岭回归, 1 = Lasso)
    enet_params = {
        'alpha': 0.05,          # 正则化强度，越大系数越稀疏
        'l1_ratio': 0.5,       # 0.5 表示 Elastic Net 混合
        'fit_intercept': True,
        'max_iter': 5000,
        'random_state': 42,
        'selection': 'cyclic'
    }
    model = ElasticNet(**enet_params)
    model.fit(df_train[feature_cols], df_train['target'])

    # ---------- 7. 预测 ----------
    df_train['pred'] = model.predict(df_train[feature_cols])
    df_test['pred']  = model.predict(df_test[feature_cols])
    df_now['pred']   = model.predict(df_now[feature_cols])

    # ---------- 8. 将预测值还原为面板格式 (T, N) ----------
    def predictions_to_panel(df_pred, times, assets, target_shape):
        pred_panel = np.full(target_shape, np.nan, dtype=np.float64)
        time_to_idx = {t: i for i, t in enumerate(times)}
        asset_to_idx = {a: j for j, a in enumerate(assets)}
        for _, row in df_pred.iterrows():
            i = time_to_idx.get(row['time'])
            j = asset_to_idx.get(row['asset'])
            if i is not None and j is not None:
                pred_panel[i, j] = row['pred']
        return pred_panel

    pred_panel_train = predictions_to_panel(df_train, times_train, assets_train, y.shape)
    pred_panel_test  = predictions_to_panel(df_test,  times_test,  assets_test,  y_test.shape)
    pred_panel_now   = predictions_to_panel(df_now,   times_now,   assets_now,   y_now.shape)

    # ---------- 9. 评估合成因子（复用已有评估函数） ----------
    def evaluate_synthetic_factor(pred_panel, y_arr, pivoted_dict, name):
        long_df = panel_to_long_factor_df(pred_panel, y_arr, pivoted_dict, name)
        ic_s = ic_curve_from_long(long_df)
        qret, ls = build_quantile_report(long_df, quantiles=QUANTILES)
        print(f"\n===== {name} =====")
        print(f"有效IC时间点: {ic_s.shape[0]}")
        if not ic_s.empty:
            mean_ic = ic_s.mean()
            icir = mean_ic / (ic_s.std() + 1e-8)
            print(f"平均IC: {mean_ic:.6f}, ICIR: {icir:.6f}")
        if not qret.empty:
            print("各分位平均未来收益:")
            print(qret.mean().to_frame('收益').T)
            print("多空组合统计:")
            print(ls.describe().to_frame('统计'))
        return ic_s, qret, ls

    print("\n--- Elastic Net 合成因子评估 ---")
    ic_train, _, _ = evaluate_synthetic_factor(pred_panel_train, y, pivoted, "训练集")
    ic_test,  _, _ = evaluate_synthetic_factor(pred_panel_test,  y_test, pivoted_test, "测试集")
    ic_now,   _, _ = evaluate_synthetic_factor(pred_panel_now,   y_now,  pivoted_now,  "真实集")

    # ---------- 10. 输出模型系数（特征重要性） ----------
    coefficients = model.coef_
    print("\nElastic Net 系数（按因子顺序，非零系数表示被选中）:")
    for i, coef in enumerate(coefficients):
        expr = formula[i] if i < len(formula) else f"因子{i}"
        print(f"  {expr}: {coef:.6f}")

    # 输出截距
    print(f"截距 (intercept): {model.intercept_:.6f}")

    # 可选：输出训练集上的 MSE/MAE
    train_mse = mean_squared_error(df_train['target'], df_train['pred'])
    train_mae = mean_absolute_error(df_train['target'], df_train['pred'])
    print(f"\n训练集 MSE: {train_mse:.6f}, MAE: {train_mae:.6f}")

    # ---------- 11. 绘图 ----------
    # Rolling IC 曲线
    plt.figure(figsize=(12, 4))
    if not ic_train.empty:
        ic_train.rolling(20, min_periods=5).mean().plot(label='训练集 Rolling IC(20)')
    if not ic_test.empty:
        ic_test.rolling(20, min_periods=5).mean().plot(label='测试集 Rolling IC(20)')
    if not ic_now.empty:
        ic_now.rolling(20, min_periods=5).mean().plot(label='真实集 Rolling IC(20)')
    plt.axhline(0.0, linestyle='--')
    plt.title('Elastic Net 合成因子 Rolling Rank IC')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ElasticNet_滚动IC.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 多空累计收益曲线
    plt.figure(figsize=(12, 4))
    for name, pred_panel, y_arr, piv in [
        ('训练集', pred_panel_train, y, pivoted),
        ('测试集', pred_panel_test,  y_test, pivoted_test),
        ('真实集', pred_panel_now,   y_now,  pivoted_now)
    ]:
        long_df = panel_to_long_factor_df(pred_panel, y_arr, piv, name)
        _, ls = build_quantile_report(long_df, quantiles=QUANTILES)
        if not ls.empty:
            ls.fillna(0).cumsum().plot(label=f'{name} 多空累计收益')
    plt.axhline(0.0, linestyle='--')
    plt.title('Elastic Net 合成因子 Top-Bottom 多空累计收益')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ElasticNet_多空累计收益.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 真实集预测值与真实值散点图
    if not df_now.empty:
        plt.figure(figsize=(6, 6))
        plt.scatter(df_now['pred'], df_now['target'], alpha=0.3, s=1)
        plt.xlabel('预测值')
        plt.ylabel('真实未来收益')
        plt.title('真实集：预测 vs 真实')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'ElasticNet_预测vs真实.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # ========== Elastic Net 扩展可视化 ==========
    output_dir = "多因子分析可视化"
    os.makedirs(output_dir, exist_ok=True)

    # ---------- 1. 系数条形图 ----------
    coefficients = model.coef_
    intercept = model.intercept_
    # 按系数绝对值排序
    coef_series = pd.Series(coefficients, index=formula)
    coef_series_sorted = coef_series.reindex(coef_series.abs().sort_values(ascending=False).index)
    
    plt.figure(figsize=(10, 6))
    colors = ['red' if c < 0 else 'green' for c in coef_series_sorted.values]
    plt.barh(range(len(coef_series_sorted)), coef_series_sorted.values, color=colors, alpha=0.7)
    plt.yticks(range(len(coef_series_sorted)), coef_series_sorted.index)
    plt.axvline(x=0, color='black', linestyle='--')
    plt.xlabel('系数值')
    plt.title('Elastic Net 系数条形图 (红色为负, 绿色为正)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ElasticNet_系数条形图.png'), dpi=150)
    plt.close()

    # ---------- 2. 正则化路径图 (不同 alpha 下的系数变化) ----------
    # 使用训练集数据，拟合一系列 alpha 值 (从大到小)
    alphas = np.logspace(-3, 1, 50)  # 0.001 到 10
    coef_path = []
    for a in alphas:
        enet = ElasticNet(alpha=a, l1_ratio=enet_params.get('l1_ratio', 0.5), 
                          fit_intercept=True, max_iter=5000, random_state=42)
        enet.fit(df_train[feature_cols], df_train['target'])
        coef_path.append(enet.coef_)
    coef_path = np.array(coef_path)  # (len(alphas), K)
    
    plt.figure(figsize=(10, 6))
    for i in range(K):
        plt.plot(alphas, coef_path[:, i], label=formula[i], linewidth=1.5)
    plt.xscale('log')
    plt.xlabel('alpha (正则化强度)')
    plt.ylabel('系数值')
    plt.title('Elastic Net 正则化路径图 (l1_ratio = {})'.format(enet_params.get('l1_ratio', 0.5)))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ElasticNet_正则化路径图.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ---------- 3. 预测残差分布直方图 (使用真实集) ----------
    if not df_now.empty:
        residuals = df_now['target'] - df_now['pred']
        residuals = residuals.dropna()
        if len(residuals) > 0:
            plt.figure(figsize=(10, 5))
            plt.hist(residuals, bins=50, density=True, alpha=0.7, color='purple', edgecolor='black')
            # 添加正态分布拟合曲线
            from scipy.stats import norm
            mu, std = norm.fit(residuals)
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            plt.plot(x, p, 'k', linewidth=2, label=f'正态拟合 (μ={mu:.4f}, σ={std:.4f})')
            plt.xlabel('残差')
            plt.ylabel('密度')
            plt.title('Elastic Net 预测残差分布 (真实集)')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'ElasticNet_残差分布直方图.png'), dpi=150)
            plt.close()

    # ---------- 4. 时间截面性能热力图 ----------
    # 构造每个时间点、每个合约的残差 (或绝对误差) 矩阵，绘制热力图展示误差分布
    if not df_now.empty:
        # 构建面板残差
        residual_panel = np.full_like(y_now, np.nan)
        time_to_idx = {t: i for i, t in enumerate(times_now)}
        asset_to_idx = {a: j for j, a in enumerate(assets_now)}
        for _, row in df_now.iterrows():
            i = time_to_idx.get(row['time'])
            j = asset_to_idx.get(row['asset'])
            if i is not None and j is not None:
                residual_panel[i, j] = row['target'] - row['pred']
        
        # 选择部分时间点避免热力图过大 (例如最多显示 50 个时间点)
        n_times_show = min(50, residual_panel.shape[0])
        if n_times_show > 1:
            # 取最后 n_times_show 个时间点 (或者均匀采样)
            step = max(1, residual_panel.shape[0] // n_times_show)
            indices = np.arange(0, residual_panel.shape[0], step)[:n_times_show]
            sub_residual = residual_panel[indices, :]
            sub_times = [times_now[i] for i in indices]
            
            # 可选：对每个时间点标准化残差以突出异常
            # 这里直接绘制原始残差
            
            plt.figure(figsize=(12, 8))
            im = plt.imshow(sub_residual, aspect='auto', cmap='RdBu', vmin=-np.nanpercentile(sub_residual, 2), 
                            vmax=np.nanpercentile(sub_residual, 98))
            plt.colorbar(im, label='残差值')
            plt.yticks(range(len(sub_times)), [str(t)[:10] for t in sub_times], fontsize=8)
            plt.xticks(range(len(assets_now)), assets_now, rotation=90, fontsize=6)
            plt.xlabel('合约')
            plt.ylabel('时间')
            plt.title('Elastic Net 残差热力图 (真实集部分时间截面)')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'ElasticNet_残差热力图.png'), dpi=150)
            plt.close()

    print("\nElastic Net 多因子合成模型分析完成。")




def integrate_instashap_to_factor_analysis(
    factor_list: List[np.ndarray],   # 每个因子形状 (T, N)
    factor_names: List[str],
    y: np.ndarray,                   # 未来收益，形状 (T, N)
    y_test: np.ndarray,
    y_now: np.ndarray,
    pivoted: Dict[str, pd.DataFrame],  # 时间×合约数据
    config: Dict = None
):
    """
    将多个因子作为输入特征，训练 InstaSHAP 模型解释因子组合
    """
    # 1. 重塑数据：将 (T, N) 转换为 (T*N, M) 格式
    T, N = y.shape
    M = len(factor_list)
    
    X_flat = np.column_stack([f.flatten() for f in factor_list])  # (T*N, M)
    y_flat = y.flatten()  # (T*N,)
    
    # 去除 NaN
    valid_mask = ~(np.isnan(X_flat).any(axis=1) | np.isnan(y_flat))
    X_clean = X_flat[valid_mask]
    y_clean = y_flat[valid_mask]
    
    print(f"InstaSHAP 训练数据准备完成: X.shape={X_clean.shape}, y.shape={y_clean.shape}")
    
    # 2. 创建 InstaSHAP 模型（将因子视为"特征"）
    # 注意：k 表示因子交互阶数，建议 k=2
    model = InstaSHAPGAM(
        n_features=M,
        k=2,  # 包含二阶因子交互
        feature_names=factor_names,
        hidden_dims=[128, 64, 32],
        activation="relu"
    )
    
    # 3. 训练
    trainer = InstaSHAPTrainer(
        model=model,
        blackbox_model=None,  # 我们直接使用 y 作为 f(x;S) 的近似
        lr=1e-3,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    
    history = trainer.train(
        X_train=X_clean,
        y_train=y_clean,
        X_val=None,  # 可添加验证集
        y_val=None,
        epochs=50,
        batch_size=256,
        verbose=True
    )
    
    # 4. 计算 SHAP 值（因子重要性）
    shapley_values = trainer.explain(X_clean)
    
    # 5. 分析结果
    mean_abs_shap = np.mean(np.abs(shapley_values), axis=0)
    importance_df = pd.DataFrame({
        'factor': factor_names,
        'mean_abs_shap': mean_abs_shap,
        'normalized_importance': mean_abs_shap / mean_abs_shap.sum()
    }).sort_values('mean_abs_shap', ascending=False)
    
    print("\n=== 因子重要性（基于 InstaSHAP） ===")
    print(importance_df)
    
    # 6. 可视化形状函数（f_T）
    shape_functions = model.get_shape_functions()
    top_factors = importance_df['factor'].head(4).tolist()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, factor_name in enumerate(top_factors):
        factor_idx = factor_names.index(factor_name)
        # 获取该因子的形状函数
        key = f"f_{factor_idx}"  # 主效应
        if key in shape_functions:
            x_vals = np.linspace(np.percentile(X_clean[:, factor_idx], 1),
                                np.percentile(X_clean[:, factor_idx], 99), 100)
            y_vals = shape_functions[key](x_vals.reshape(-1, 1))
            axes[idx].plot(x_vals, y_vals)
            axes[idx].set_title(f"{factor_name} 的形状函数 (f_{factor_idx})")
            axes[idx].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            axes[idx].set_xlabel("因子值")
            axes[idx].set_ylabel("贡献")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'InstaSHAP_形状函数.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    return model, trainer, importance_df, history


# ==================== InstaSHAP 多因子组合模型 ====================
# 仅在多因子列表（公式个数 > 1）时执行
if isinstance(formula, list) and len(formula) > 1 and USE_INSTASHAP:
    print("\n" + "="*60)
    print("开始 InstaSHAP 多因子组合模型训练与评估")
    print("="*60)

    # ---------- 1. 获取已计算好的因子数组 ----------
    factor_list_train = factors["factor"]      # list of (T_train, N) arrays
    factor_list_test  = factors["factor_test"] # list of (T_test, N) arrays
    factor_list_now   = factors["factor_now"]  # list of (T_now, N) arrays
    K = len(factor_list_train)                 # 因子个数
    factor_names = [f"F{i+1}" for i in range(K)]

    print(f"\n共 {K} 个因子参与 InstaSHAP 分析")

    # ---------- 2. 训练并评估 InstaSHAP ----------
    if len(factor_list_train) > 0:
        model, trainer, importance_df, history = integrate_instashap_to_factor_analysis(
            factor_list=factor_list_train,
            factor_names=factor_names,
            y=y,
            y_test=y_test,
            y_now=y_now,
            pivoted=pivoted
        )
        
        # 保存模型
        torch.save(model.state_dict(), os.path.join(output_dir, "instashap_model.pth"))
        print(f"InstaSHAP 模型已保存至 {output_dir}/instashap_model.pth")


    # ========== InstaSHAP 扩展可视化 ==========
    # ---------- 1. 因子重要性条形图（基于平均绝对 SHAP 值） ----------
    plt.figure(figsize=(10, 6))
    factors = importance_df['factor'].values
    imp_values = importance_df['mean_abs_shap'].values
    sorted_idx = np.argsort(imp_values)[::-1]
    plt.barh(range(len(factors)), imp_values[sorted_idx], color='steelblue')
    plt.yticks(range(len(factors)), factors[sorted_idx])
    plt.xlabel('平均 |SHAP 值|')
    plt.title('InstaSHAP 因子重要性条形图')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'InstaSHAP_因子重要性.png'), dpi=150)
    plt.close()

    # ---------- 2. 二阶交互效应热力图 ----------
    # 提取所有二阶交互项的平均绝对贡献 |φ_{i,j}|
    # InstaSHAPGAM 模型中，每个二阶交互项对应一个 subnet，key 为 "i_j"
    n_feat = model.n_features
    interaction_matrix = np.zeros((n_feat, n_feat))
    
    # 获取所有样本的 X_clean (训练时用的数据)
    # 重新整理训练数据用于计算交互贡献
    T, N = y.shape
    M = len(factor_list_train)
    X_flat = np.column_stack([f.flatten() for f in factor_list_train])
    y_flat = y.flatten()
    valid_mask = ~(np.isnan(X_flat).any(axis=1) | np.isnan(y_flat))
    X_clean = X_flat[valid_mask]
    
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(X_clean, dtype=torch.float32, device=trainer.device)
        # 计算每个二阶交互项的贡献值
        for T_subset in model.interaction_sets:
            if len(T_subset) == 2:
                i, j = T_subset
                key = model._subset_to_key(T_subset)
                subnet = model.subnets[key]
                x_T = x_tensor[:, list(T_subset)]
                phi_T = subnet(x_T).squeeze().cpu().numpy()
                mean_abs_contrib = np.mean(np.abs(phi_T))
                interaction_matrix[i, j] = mean_abs_contrib
                interaction_matrix[j, i] = mean_abs_contrib  # 对称
    
    # 绘制热力图
    plt.figure(figsize=(10, 8))
    im = plt.imshow(interaction_matrix, cmap='Reds', aspect='auto')
    plt.colorbar(im, label='平均 |φ_{i,j}|')
    plt.xticks(range(n_feat), factor_names, rotation=45, ha='right')
    plt.yticks(range(n_feat), factor_names)
    plt.title('InstaSHAP 二阶交互效应强度热力图')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'InstaSHAP_交互效应热力图.png'), dpi=150)
    plt.close()

    # ---------- 3. SHAP 值蜂群图（beeswarm） ----------
    # 使用 matplotlib 手动实现简化蜂群图（随机抖动 + 按特征分组）
    shapley_all = trainer.explain(X_clean)  # (n_samples, n_feat)
    # 将数据转换为长格式用于绘图
    n_samples_display = min(2000, shapley_all.shape[0])  # 限制样本数避免过密
    sample_idx = np.random.choice(shapley_all.shape[0], n_samples_display, replace=False)
    shap_sample = shapley_all[sample_idx, :]
    
    plt.figure(figsize=(12, 7))
    # 按特征重要性排序显示
    order = np.argsort(np.mean(np.abs(shap_sample), axis=0))[::-1]
    positions = []
    labels = []
    for i, feat_idx in enumerate(order):
        values = shap_sample[:, feat_idx]
        # 添加随机垂直抖动（正态分布，标准差 0.08）
        jitter = np.random.normal(0, 0.08, size=len(values))
        y_pos = np.ones_like(values) * i + jitter
        # 根据 SHAP 值正负着色
        colors = ['red' if v > 0 else 'blue' for v in values]
        plt.scatter(values, y_pos, c=colors, alpha=0.6, s=10, edgecolors='none')
        positions.append(i)
        labels.append(factor_names[feat_idx])
    
    plt.yticks(positions, labels)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    plt.xlabel('SHAP 值（对预测的贡献）')
    plt.title('InstaSHAP 因子贡献蜂群图（红色=正向，蓝色=负向）')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'InstaSHAP_蜂群图.png'), dpi=150)
    plt.close()

    # ---------- 4. 交互网络图（基于二阶交互强度） ----------
    try:
        import networkx as nx
        # 构建网络
        G = nx.Graph()
        # 添加节点
        for i, name in enumerate(factor_names):
            G.add_node(name, size=np.mean(np.abs(shapley_all[:, i])))
        # 添加边（二阶交互强度大于阈值）
        threshold = np.percentile(interaction_matrix[interaction_matrix > 0], 70) if np.any(interaction_matrix > 0) else 0.01
        for i in range(n_feat):
            for j in range(i+1, n_feat):
                weight = interaction_matrix[i, j]
                if weight > threshold:
                    G.add_edge(factor_names[i], factor_names[j], weight=weight)
        
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G, seed=42, k=2, iterations=50)
        # 节点大小正比于平均绝对 SHAP
        node_sizes = [G.nodes[n]['size'] * 500 for n in G.nodes]
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue', edgecolors='black')
        nx.draw_networkx_labels(G, pos, font_size=10)
        # 边宽正比于交互强度
        edges = G.edges(data=True)
        widths = [d['weight'] * 3 for (_, _, d) in edges]
        nx.draw_networkx_edges(G, pos, width=widths, alpha=0.6, edge_color='gray')
        plt.title('因子交互网络图（节点大小=重要性，边宽=二阶交互强度）')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'InstaSHAP_交互网络图.png'), dpi=150)
        plt.close()
    except ImportError:
        print("警告：未安装 networkx 库，跳过交互网络图。")

    print("InstaSHAP 扩展可视化完成。")

    print("\nInstaSHAP 多因子组合模型分析完成。")

if __name__ == "__main__":
    try:
        build_real_data_visualizations()
    except Exception as exc:
        print(f"真实数据可视化失败: {exc}")