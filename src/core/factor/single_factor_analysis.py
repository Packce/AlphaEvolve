
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

nest_asyncio.apply()
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", message=".*Glyph.*")
warnings.filterwarnings("ignore", message=".*does not have a glyph.*")
warnings.filterwarnings("ignore", message=".*has no glyph.*")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 高级配色
COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3F88C5']

# ## 需要调整的全局变量参数

# 使用转换器的因子表达式
# formula = "CORR(TS_ZSCORE(high, LV(high, 5)), EMV(85), open_interest)"
formula = "RANK(WR(2), BOLL_UPPER(24, 2.26))"

# 是否使用转换器，不使用时需修改手动计算因子中的因子表达式
AUTO_ON = True

# 是否保存因子到当前目录的因子库中
SAVE_FACTOR = True

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
USE_ALPHALENS = False           # 是否使用ALPHALENS，15分钟数据下默认关闭；日频时可开启
# USE_ALPHALENS = True

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

# ========== 板块选择配置 ==========
# 选择要使用的板块，可以设置为：
# - "all": 使用所有板块的所有合约
# - "农产品": 使用农产品板块的所有合约
# - "金属": 使用金属板块的所有合约
# - "能源化工建材航运": 使用能源化工建材航运板块的所有合约
# - "金融": 使用金融板块的所有合约
# - ["农产品", "金属"]: 使用多个板块的所有合约
# - ["农产品", "油脂油料类"]: 使用指定板块的指定子板块
# - [["农产品", "油脂油料类"], ["金属", "贵金属"]]: 使用多个子板块
# - [["农产品", "油脂油料类"], "金属"]: 混合选择（子板块和整个板块）
# - ["时间分类", "日盘商品"]: 使用日盘商品的所有合约
# - ["时间分类", "标准夜盘商品"]: 使用标准夜盘商品的所有合约
# - ["时间分类", "有色金属"]: 使用有色金属的所有合约
# - ["时间分类", "贵金属与原油"]: 使用贵金属与原油的所有合约
# - ["时间分类", "股指期货"]: 使用股指期货的所有合约
# - ["时间分类", "国债期货"]: 使用国债期货的所有合约
# - 或者设置为 None，然后手动指定 SYMBOLS 列表

# SELECTED_SECTOR = ["时间分类", "有色金属"]  # 设置为 None 表示手动指定合约，或设置为上述选项之一
# SELECTED_SECTOR = None  # 设置为 None 表示手动指定合约，或设置为上述选项之一

# ========== 手动指定合约列表（当 SELECTED_SECTOR 为 None 时使用）==========
# 合约列表（支持多个合约），格式为 "合约代码888"
# SYMBOLS = ["au888", "rb888","cu888","jd888","AP888"]  # 例如：["au888", "rb888", "cu888"]
# SYMBOLS = ["au888","ag888"]  # 例如：["au888", "rb888", "cu888"]

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


# ## 旧的获取数据
# import pandas as pd
# import numpy as np

# # 读取 CSV，前两行作为多级表头（feature, code），跳过第3行 "time"
# df = pd.read_csv(r'c:\Users\Admin\AppData\Local\qmfquant\factor\train_X_dict_all.csv', header=[0, 1], skiprows=[2])
# print(df)

# # 第一列是时间索引
# time_col = df.columns[0]
# data_df = df.drop(columns=[time_col])  # 或保留时间用于索引

# # 按 feature 分组列：保留全部 feature（open, close, high, low, volume, open_interest 等）
# data_read_csv = {}
# for key in data_df.columns.get_level_values(0).unique():
#     if key == 'feature' or key == 'code':  # 首列已去掉，这里防呆
#         continue
#     cols = [c for c in data_df.columns if c[0] == key]
#     if cols:
#         arr = data_df[cols].values.astype(np.float64)
#         data_read_csv[key] = arr

# # data_read_csv 包含所有 feature，如 open, close, high, low, volume, open_interest
# # 每个 value 为 array(T, N)，T=时间行数, N=品种数
# print({k: v.shape for k, v in data_read_csv.items()})
# print(data_read_csv)
# print(data_read_csv["open"].shape)


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


def build_factor_expressions(formula: str, my_cls: Any, my_name: str = "My") -> Dict[str, str]:
    return {
        "factor": convert_formula(formula, my_cls=my_cls, x_dict_name="X_dict", my_name=my_name),
        "factor_now": convert_formula(formula, my_cls=my_cls, x_dict_name="X_dict_now", my_name=my_name),
        "factor_test": convert_formula(formula, my_cls=my_cls, x_dict_name="X_dict_test", my_name=my_name),
    }


def eval_factors(
    formula: str,
    my_cls: Any,
    X_dict: Dict[str, np.ndarray],
    X_dict_now: Dict[str, np.ndarray],
    X_dict_test: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    exprs = build_factor_expressions(formula=formula, my_cls=my_cls, my_name="My")
    env = {
        "np": np,
        "My": my_cls,
        "X_dict": X_dict,
        "X_dict_now": X_dict_now,
        "X_dict_test": X_dict_test,
    }
    return {name: eval(expr, env, {}) for name, expr in exprs.items()}# #### 使用转换器计算因子
# formula = "CORR(TS_ZSCORE(high, LV(high, 5)), EMV(85), open_interest)"
# formula = "ADD(WR(1.0), SIN(MUL(LLV(HV(MIN(EMV(22), RANK(BIAS(23), EMV(5.048163479178479))), 11), KDJ_J(4, 63, 11)), SLOPE(RSI(36), 88.67576321773176))))"
# formula = "RANK(WR(3), FORCAST(EMA(MA(BRAR_BR(39.67400187583232), PSY(61)), SCALE(RANK(BOLL_LOWER(51.92058608481411, 105.33912028012675), CCI(89)), MUL(TS_ZSCORE(TOPRANGE(TS_RANK(TR(), ATR(95))), MAX(CR(18), KDJ_J(10, 74, 79))), RANK(DIFMA(101, 105.70288284157309, 54), DIFMA(67, 119.59247708006433, 110))))), RANK(TS_ZSCORE(SCALE(RANK_SUB(MACD_MACD(103.1367287638528, 49, 71), TS_ZSCORE(ATR(74), DPO(5, 63)), TS_ZSCORE(LLV(CR(73.59636181482315), 60.49431526716149), SIGNEDPOWER(KDJ_D(95.0270735386948, 111.14688602129436, 106.9185473465797)))), SCALE(MAX(ASI(41.10188909712552), TS_ZSCORE(MACD_DIF(90, 11), MFI(27))), RANK_SUB(BBI(100.80402744834933, 17, 12.732093098580961, 29), BIAS(54), KDJ_D(49, 80.24928039078695, 90.62123976587635)))), SCALE(SMA(SIN(SAR(63, 46.70113766183343, 62.152669951821125)), 4, 5.0), SCALE(VR(15), ADX(40.069828069434735, 90.18823201047609)))), INV(RANK(SCALE(RANK(CORR(KDJ_J(89, 57.906667999650715, 41.60920604942375), RSI(22.009126046643384), MACD_DEA(69, 4, 65)), ABS(CR(90.02718123376458))), EMV(65)), CCI(78.69447786554845))))))"
# formula = "CORR(CORR(SIN(TOPRANGE(PSY(75))), TAN(MASS(188.91227410655506, 50)), ROC(5.943201414914385)), ATR(64.67582157232408), BIAS(15.493019300191346))"
# 1) 先看转换结果字符串
exprs = build_factor_expressions(formula, My)
# print(exprs["factor"])
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


temp1 = fitness_func(factor, y, factor_test=factor_test, y_test=y_test, return_details=True)
temp_train = calc_ic_stats(factor, y)
temp_test = calc_ic_stats(factor_test, y_test)
temp_now = calc_ic_stats(factor_now, y_now)

print(f"因子表达式: {formula}")
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


# ## 因子加入本地库

if SAVE_FACTOR:
    # 将当前因子结果追加到当前目录下的csv文件中（如不存在则新建）
    output_csv = "因子库.csv"
    row = {
        "入库时间": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "因子表达式": formula,
        "品种选择": SELECTED_SECTOR,
        "品种合约": str(SYMBOLS),
        "数据周期": SYMBOL_CYCLE,
        "预测周期": Y_PERIOD,
        "训练集时间": f"{BEGIN_TIME}~{END_TIME}",
        "训练集IC": temp_train['mean_ic'],
        "训练集IR": temp_train['icir'],
        "训练集有效时间点": temp_train['valid_ts'],
        "测试集时间": f"{BEGIN_TIME_TEST}~{END_TIME_TEST}",
        "测试集IC": temp_test['mean_ic'],
        "测试集IR": temp_test['icir'],
        "测试集有效时间点": temp_test['valid_ts'],
        "真实集时间": f"{BEGIN_TIME_NOW}~{END_TIME_NOW}",
        "真实集IC": temp_now['mean_ic'],
        "真实集IR": temp_now['icir'],
        "真实集有效时间点": temp_now['valid_ts'],
        "按IC分类": "IC有效因子" if temp_now['mean_ic'] >= 0.05 else "IC无效因子",
        "按IR分类": "IR稳定因子" if temp_now['icir'] >= 0.3 else "IR不稳定因子"
    }

    # 判断重复只考虑：因子表达式、品种合约、预测周期、训练集时间、测试集时间、真实集时间
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
            df_base = pd.read_csv(output_csv, dtype=str)  # 以字符串形式读，避免类型误差
        except Exception:
            df_base = pd.DataFrame()
        is_duplicate = False
        if not df_base.empty:
            # 将当前row转为DataFrame，类型全部转str
            row_df = pd.DataFrame([{col: str(row[col]) for col in duplicate_cols}])
            df_cmp = df_base[duplicate_cols].astype(str)
            # 比较仅这几个字段是否完全一致
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
        print(f"本次因子结果已追加到 {output_csv}")


# ## 准备长表分析数据

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


analysis_train = panel_to_long_factor_df(factor, y, pivoted, 'train')
analysis_test = panel_to_long_factor_df(factor_test, y_test, pivoted_test, 'test')
analysis_now = panel_to_long_factor_df(factor_now, y_now, pivoted_now, 'now')

# print(analysis_train.head())


# ## 计算分层收益与多空收益
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


# ===================== 因子分析可视化（自动保存图片） =====================

def save_figure(fig, filename):
    output_dir = "单因子分析可视化"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_ic_ir_analysis(ic_series_dict, ir_vals, title="因子 IC / IR 分析"):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(title, fontsize=18, fontweight='bold')

    colors = {name: COLORS[i % len(COLORS)] for i, name in enumerate(ic_series_dict.keys())}

    for name, ic in ic_series_dict.items():
        ax1.plot(ic.index, ic, color=colors[name], linewidth=1.5, label=f"{name} | 均值={ic.mean():.3f}")
    ax1.axhline(0, color='black', linestyle='--', alpha=0.6)
    ax1.set_title('IC 时间序列')
    ax1.legend()

    for name, ic in ic_series_dict.items():
        ax2.plot(ic.index, ic.rolling(20).mean(), color=colors[name], linewidth=2)
    ax2.axhline(0, color='black', linestyle='--', alpha=0.6)
    ax2.set_title('滚动 IC (20期)')

    ic_all = np.concatenate([v.dropna().values for v in ic_series_dict.values()])
    ax3.hist(ic_all, bins=30, color=COLORS[0], alpha=0.7)
    ax3.axvline(ic_all.mean(), color='crimson', linewidth=2, label=f"均值={ic_all.mean():.3f}")
    ax3.set_title('IC 分布')
    ax3.legend()

    names = list(ir_vals.keys())
    irs = list(ir_vals.values())
    ax4.bar(names, irs, color=[colors[name] for name in names], alpha=0.85)
    for b, v in zip(ax4.patches, irs):
        ax4.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{v:.3f}", ha="center")
    ax4.axhline(0.2, color='orange', linestyle='--', label="合格 0.2")
    ax4.axhline(0.5, color='red', linestyle='--', label="优秀 0.5")
    ax4.set_title('信息比率 IR')
    ax4.legend()

    plt.tight_layout()
    save_figure(fig, "IC_IR分析.png")


def plot_quantile_return(quantile_ret_dict, title="因子分位数收益"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle(title, fontsize=18, fontweight='bold')

    x = np.arange(1, 6)
    w = 0.18
    for i, (name, df) in enumerate(quantile_ret_dict.items()):
        if df.empty:
            continue
        m = df.mean()
        ax1.bar(x + i * w, m, w, color=COLORS[i], alpha=0.85, label=name)
    ax1.axhline(0, color='black', linestyle='--', alpha=0.6)
    ax1.set_title('分位数平均收益')
    ax1.legend()

    for i, (name, df) in enumerate(quantile_ret_dict.items()):
        if df.empty or 1 not in df.columns or 5 not in df.columns:
            continue
        ls = (df[5] - df[1]).fillna(0).cumsum()
        ax2.plot(ls.index, ls, color=COLORS[i], linewidth=2.2, label=name)
    ax2.axhline(0, color='black', linestyle='--', alpha=0.6)
    ax2.set_title('多空累计收益 5-1')
    ax2.legend()

    plt.tight_layout()
    save_figure(fig, "分位数收益.png")


def plot_factor_return_dist(quantile_ret):
    fig, ax = plt.subplots(figsize=(12, 6))
    data = [quantile_ret[q].dropna().values for q in range(1, 6) if q in quantile_ret]
    if len(data) > 0:
        parts = ax.violinplot(data, positions=list(range(1, len(data) + 1)), showmedians=True)
        for i, b in enumerate(parts['bodies']):
            b.set_facecolor(COLORS[i % len(COLORS)])
            b.set_alpha(0.7)
        parts['cmedians'].set_color('white')
        parts['cmedians'].set_linewidth(2)
        ax.set_title('分位数收益分布')
        ax.set_xticks(list(range(1, len(data) + 1)))

    plt.tight_layout()
    save_figure(fig, "收益分布.png")


def factor_analysis_visualization(ic_series_dict, ir_vals, quantile_ret_dict, quantile_ret):
    plot_ic_ir_analysis(ic_series_dict, ir_vals)
    plot_quantile_return(quantile_ret_dict)
    plot_factor_return_dist(quantile_ret)
    print('所有图表已生成！单因子分析结束')


def build_visualization_data():
    ic_series_dict = {}
    ir_vals = {}
    quantile_ret_dict = {}

    for split_name, long_df in [('训练集', analysis_train), ('测试集', analysis_test), ('真实集', analysis_now)]:
        ic_s = ic_curve_from_long(long_df)
        qret, _ = build_quantile_report(long_df, quantiles=QUANTILES)

        ic_series_dict[split_name] = ic_s
        ir_vals[split_name] = ic_s.mean() / (ic_s.std() + 1e-8) if not ic_s.empty else 0.0
        quantile_ret_dict[split_name] = qret

    quantile_ret = {}
    for qret in quantile_ret_dict.values():
        if not qret.empty:
            quantile_ret.update({q: qret[q].dropna() for q in qret.columns if q in range(1, QUANTILES + 1)})

    return ic_series_dict, ir_vals, quantile_ret_dict, quantile_ret


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
plt.figure(figsize=(12, 4))
for label, long_df in [('训练集', analysis_train), ('测试集', analysis_test), ('真实集', analysis_now)]:
    ic_s = ic_curve_from_long(long_df)
    if not ic_s.empty:
        ic_s.rolling(20, min_periods=5).mean().plot(label=f'{label} Rolling IC(20)')
plt.axhline(0.0, linestyle='--')
plt.title('Rolling Rank IC')
plt.legend()
output_dir = "单因子分析可视化"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, '滚动IC.png'), dpi=150, bbox_inches='tight')
plt.close()

plt.figure(figsize=(12, 4))
for label, long_df in [('训练集', analysis_train), ('测试集', analysis_test), ('真实集', analysis_now)]:
    _, ls = build_quantile_report(long_df, quantiles=QUANTILES)
    if not ls.empty:
        ls.fillna(0).cumsum().plot(label=f'{label} 多空累计收益')
plt.axhline(0.0, linestyle='--')
plt.title('Top-Bottom Quantile Long-Short Cumulative Return')
plt.legend()
plt.savefig(os.path.join(output_dir, '多空累计收益.png'), dpi=150, bbox_inches='tight')
plt.close()


# ===================== 自动可视化输出 =====================
ic_series_dict, ir_vals, quantile_ret_dict, quantile_ret = build_visualization_data()
factor_analysis_visualization(ic_series_dict, ir_vals, quantile_ret_dict, quantile_ret)


# ## 可选：准备 Alphalens 数据格式
def prepare_alphalens_inputs(long_df, pivot_close):
    factor_series = (
        long_df[['time', 'code', 'factor']]
        .dropna()
        .set_index(['time', 'code'])['factor']
        .sort_index()
    )
    price_df = pivot_close.sort_index().sort_index(axis=1)
    return factor_series, price_df


if USE_ALPHALENS:
    factor_train_al, prices_train_al = prepare_alphalens_inputs(analysis_train, pivoted['close'])
    try:
        al_train = alphalens.utils.get_clean_factor_and_forward_returns(
            factor=factor_train_al,
            prices=prices_train_al,
            quantiles=QUANTILES,
            periods=PERIODS,
            max_loss=0.35,
        )
        print(al_train.head())
    except Exception as e:
        print(f"Alphalens 数据准备失败: {e}")
else:
    print("USE_ALPHALENS=False，默认跳过 Alphalens 数据准备。对于15分钟数据，建议先使用上面的自定义 IC / 分层分析结果。")


# ## 可选：使用 Alphalens 生成分析图
if USE_ALPHALENS:
    try:
        alphalens.tears.create_returns_tear_sheet(al_train)
        alphalens.tears.create_information_tear_sheet(al_train)
    except Exception as e:
        print(f"Alphalens 分析失败: {e}")
else:
    print("已跳过 Alphalens tear sheet。若切换到日频数据，可将 USE_ALPHALENS 改为 True 再运行。")

