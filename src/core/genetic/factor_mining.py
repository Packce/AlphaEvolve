"""                 
遗传编程(Genetic Programming)框架
=================================

这是一个用于金融因子挖掘的遗传编程框架，包含以下主要组件：

1. 基础函数集：包含数学运算、时间序列分析、排名等函数
2. 函数集管理类：动态管理可用的函数
3. 表达式树节点类：表示因子表达式的树形结构
4. 遗传编程主类：实现进化算法的核心逻辑

主要特性：
- 防止未来函数泄漏的滚动窗口计算
- 安全的数值计算（处理除零、溢出等异常）
- 支持多种遗传操作（交叉、变异、选择）
- 可扩展的函数集
- 树复杂度控制和简约惩罚
"""

import numpy as np
import pandas as pd
import operator
import random
import time
import logging
import sys
import os
from datetime import datetime
from typing import List, Dict, Any, Callable, Optional, Union
from scipy.stats import rankdata as _rankdata

try:
    import cupy as cp
    import cudf
    _GPU_AVAILABLE = cp.cuda.is_available()
    if _GPU_AVAILABLE:
        xp = cp
        gd = cudf
    else:
        xp = np
        gd = pd
except ImportError:
    _GPU_AVAILABLE = False
    xp = np
    gd = pd

_is_cupy = _GPU_AVAILABLE

def _to_cpu(arr):
    """将数组转换回numpy/cpu格式"""
    if _is_cupy and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return arr

def _ensure_xp_array(data):
    """确保输入数据为当前后端数组"""
    if isinstance(data, (list, tuple)):
        return xp.array(data)
    elif isinstance(data, pd.Series):
        return xp.asarray(data.values)
    elif isinstance(data, np.ndarray):
        return xp.asarray(data)
    elif _is_cupy and isinstance(data, cp.ndarray):
        return data
    else:
        return xp.asarray([data])

def _to_np_output(data):
    """确保输出为numpy格式（用于返回结果）"""
    if isinstance(data, tuple):
        return tuple(_to_np_output(item) for item in data)
    if isinstance(data, pd.Series):
        return data.values
    if isinstance(data, np.ndarray):
        return data
    if _is_cupy and isinstance(data, cp.ndarray):
        return cp.asnumpy(data)
    if isinstance(data, (list, tuple)):
        return np.asarray(data)
    return np.asarray(data)


def _safe_scalar_param(arg, default=10.0):
    """从标量/数组中鲁棒提取单个有限数值，用于高阶指标参数。"""
    try:
        if np.isscalar(arg):
            val = float(arg)
            return val if np.isfinite(val) else default

        arr = np.asarray(arg, dtype=np.float64)
        finite_vals = arr[np.isfinite(arr)]
        if finite_vals.size == 0:
            return default

        # 优先取最后一个有限值，通常比机械取首元素更稳健
        return float(finite_vals[-1])
    except Exception:
        return default

logger = logging.getLogger("gp_framework")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_handler)
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"gp_studay_{run_stamp}.log")
    _file_handler = logging.FileHandler(log_path, encoding="utf-8")
    _file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    logger.addHandler(_file_handler)
    logger.propagate = False


def log_print(*args, **kwargs):
    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    message = sep.join(str(a) for a in args)
    if end and end != "\n":
        message = f"{message}{end}"
    logger.info(message)


def _normalize_window_length(
    d, x=None, other=None, min_len=1, default=None, max_len=120, return_type="int"
):
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
            if not np.isfinite(d): # isfinite():检查是否为有限值，inf和nan都是非有限值
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
        d = float(np.round(d, 6))  # 保留小数点后6位，可修改
        d = max(float(min_len), d)
    else:
        d = int(np.round(d))
        d = max(min_len, d)

    # 步骤4：尝试根据x和other自动推断最大窗口（防止超出数据长度）
    available_len = None
    if x is not None and not np.isscalar(x):
        x_arr = np.asarray(x)
        if x_arr.size > 0:
            available_len = x_arr.shape[0]
    if other is not None and not np.isscalar(other):
        other_arr = np.asarray(other)
        if other_arr.size > 0:
            # 如果available_len已设置，则取两个序列中较小的长度
            available_len = (
                other_arr.shape[0]
                if available_len is None
                else min(available_len, other_arr.shape[0])
            )

    # 步骤5：窗口长度不得超过max_len和推断所得数据可用长度
    max_len_val = float(max_len) if return_type == "float" else max_len
    if available_len is not None:
        max_lim = float(available_len) if return_type == "float" else available_len
        d = min(d, max_len_val, max_lim)
    else:
        d = min(d, max_len_val)

    # 返回最终规范化的窗口长度
    return d


# ================= 高阶技术指标包装器函数 =================
# 这些包装器确保周期参数被正确转换为标量整数，以便在遗传编程中使用。
# 如果遗传算法误将序列传给周期参数槽位，包装器会进行安全转换。

# 全局变量：存储当前的X_dict，用于验证行情数据参数
_current_X_dict = None


def _set_current_X_dict(X_dict):
    """设置当前的X_dict，用于验证行情数据参数"""
    global _current_X_dict
    _current_X_dict = X_dict


def _get_market_data(key, default_key=None):
    """
    从_current_X_dict获取行情数据

    参数:
        key: 行情数据键名，如 'close', 'high', 'low', 'open', 'volume'
        default_key: 如果key不存在，尝试使用的默认键名

    返回:
        行情数据数组，如果不存在则返回None
    """
    global _current_X_dict

    if _current_X_dict is None or not isinstance(_current_X_dict, dict):
        return None

    # 优先使用指定的key
    if key in _current_X_dict:
        return _current_X_dict[key]

    # 如果指定了default_key，尝试使用
    if default_key and default_key in _current_X_dict:
        return _current_X_dict[default_key]

    return None


def _validate_market_data(param, allowed_keys):
    """
    验证参数是否是允许的原始行情数据

    参数:
        param: 要验证的参数（数组）
        allowed_keys: 允许的行情数据键列表，如 ['close', 'high', 'low']

    返回:
        bool: 如果参数是允许的原始行情数据，返回True；否则返回False
    """
    global _current_X_dict

    if _current_X_dict is None:
        # 如果没有设置X_dict，无法验证，返回True（向后兼容）
        return True

    if not isinstance(_current_X_dict, dict):
        return True

    # 检查参数是否直接来自X_dict中的允许键
    for key in allowed_keys:
        if key in _current_X_dict:
            # 使用numpy的array_equal来比较数组是否相同（考虑内存地址和值）
            try:
                if np.array_equal(param, _current_X_dict[key], equal_nan=True):
                    return True
                # 也检查是否是同一个对象（内存地址相同）
                if param is _current_X_dict[key]:
                    return True
            except Exception:
                pass

    log_print(f"验证行情数据参数失败: {param}, {allowed_keys}")
    return False


def _to_int_param(p, default=10, x=None, other=None, min_len=1, max_len=120):
    """安全地将参数转换为整数标量，并限制到可用长度"""
    base_x = x
    base_other = other
    if base_x is None and base_other is None:
        if (
            _current_X_dict
            and isinstance(_current_X_dict, dict)
            and len(_current_X_dict) > 0
        ):
            try:
                base_x = next(iter(_current_X_dict.values()))
            except Exception:
                base_x = None
    return _normalize_window_length(
        p, x=base_x, other=base_other, min_len=min_len, default=default, max_len=max_len
    )


def _to_float_param(p, default=2.0):
    """安全地将参数转换为浮点标量"""
    if isinstance(p, (list, np.ndarray, pd.Series)):
        try:
            val = p[-1] if len(p) > 0 else default
            if isinstance(val, (np.ndarray, pd.Series)):
                val = val.iloc[-1] if hasattr(val, "iloc") else val[0]
            return float(val)
        except:
            return default
    try:
        return float(p)
    except:
        return default


# ================= 新的包装器函数：自动从X_dict获取行情数据，只接受周期参数 =================


def macd_dif_func_auto(short=12, long=26):
    """MACD DIF指标包装器 - 自动从X_dict获取close，只接受周期参数"""
    close = _get_market_data("close")
    if close is None:
        if (
            _current_X_dict
            and isinstance(_current_X_dict, dict)
            and len(_current_X_dict) > 0
        ):
            arr = next(iter(_current_X_dict.values()))
            log_print(f"验证行情数据参数失败: {close}, ['close']")
            return np.full_like(arr, np.nan, dtype=np.float64)
        log_print(f"验证行情数据参数失败: {close}, ['close']")
        return np.nan
    s = _to_int_param(short, 12)
    l = _to_int_param(long, 26)
    return My.MACD_DIF(close, s, l)


def macd_dea_func_auto(short=12, long=26, m=9):
    """MACD DEA指标包装器 - 自动从X_dict获取close，只接受周期参数"""
    close = _get_market_data("close")
    if close is None:
        if (
            _current_X_dict
            and isinstance(_current_X_dict, dict)
            and len(_current_X_dict) > 0
        ):
            arr = next(iter(_current_X_dict.values()))
            return np.full_like(arr, np.nan, dtype=np.float64)
        return np.nan
    s = _to_int_param(short, 12)
    l = _to_int_param(long, 26)
    m_val = _to_int_param(m, 9)
    return My.MACD_DEA(close, s, l, m_val)


def macd_macd_func_auto(short=12, long=26, m=9):
    """MACD MACD指标包装器 - 自动从X_dict获取close，只接受周期参数"""
    close = _get_market_data("close")
    if close is None:
        if (
            _current_X_dict
            and isinstance(_current_X_dict, dict)
            and len(_current_X_dict) > 0
        ):
            arr = next(iter(_current_X_dict.values()))
            return np.full_like(arr, np.nan, dtype=np.float64)
        return np.nan
    s = _to_int_param(short, 12)
    l = _to_int_param(long, 26)
    m_val = _to_int_param(m, 9)
    return My.MACD_MACD(close, s, l, m_val)


def kdj_k_func_auto(n=9, m1=3):
    """KDJ K指标包装器 - 自动从X_dict获取close, high, low，只接受周期参数"""
    close = _get_market_data("close")
    high = _get_market_data("high")
    low = _get_market_data("low")
    if close is None or high is None or low is None:
        if (
            _current_X_dict
            and isinstance(_current_X_dict, dict)
            and len(_current_X_dict) > 0
        ):
            arr = next(iter(_current_X_dict.values()))
            return np.full_like(arr, np.nan, dtype=np.float64)
        return np.nan
    n_val = _to_int_param(n, 9)
    m1_val = _to_int_param(m1, 3)
    return My.KDJ_K(close, high, low, n_val, m1_val)


def kdj_d_func_auto(n=9, m1=3, m2=3):
    """KDJ D指标包装器 - 自动从X_dict获取close, high, low，只接受周期参数"""
    close = _get_market_data("close")
    high = _get_market_data("high")
    low = _get_market_data("low")
    if close is None or high is None or low is None:
        if (
            _current_X_dict
            and isinstance(_current_X_dict, dict)
            and len(_current_X_dict) > 0
        ):
            arr = next(iter(_current_X_dict.values()))
            return np.full_like(arr, np.nan, dtype=np.float64)
        return np.nan
    n_val = _to_int_param(n, 9)
    m1_val = _to_int_param(m1, 3)
    m2_val = _to_int_param(m2, 3)
    return My.KDJ_D(close, high, low, n_val, m1_val, m2_val)


def kdj_j_func_auto(n=9, m1=3, m2=3):
    """KDJ J指标包装器 - 自动从X_dict获取close, high, low，只接受周期参数"""
    close = _get_market_data("close")
    high = _get_market_data("high")
    low = _get_market_data("low")
    if close is None or high is None or low is None:
        if (
            _current_X_dict
            and isinstance(_current_X_dict, dict)
            and len(_current_X_dict) > 0
        ):
            arr = next(iter(_current_X_dict.values()))
            return np.full_like(arr, np.nan, dtype=np.float64)
        return np.nan
    n_val = _to_int_param(n, 9)
    m1_val = _to_int_param(m1, 3)
    m2_val = _to_int_param(m2, 3)
    return My.KDJ_J(close, high, low, n_val, m1_val, m2_val)


def rsi_func_auto(n=14):
    """RSI指标包装器 - 自动从X_dict获取close，只接受周期参数"""
    close = _get_market_data("close")
    if close is None:
        if (
            _current_X_dict
            and isinstance(_current_X_dict, dict)
            and len(_current_X_dict) > 0
        ):
            arr = next(iter(_current_X_dict.values()))
            return np.full_like(arr, np.nan, dtype=np.float64)
        return np.nan
    n_val = _to_int_param(n, 14)
    return My.RSI(close, n_val)


def sar_func_auto(n=10, s=2, m=20):
    """SAR指标包装器 - 自动从X_dict获取high, low，只接受周期参数"""
    high = _get_market_data("high")
    low = _get_market_data("low")
    if high is None or low is None:
        if (
            _current_X_dict
            and isinstance(_current_X_dict, dict)
            and len(_current_X_dict) > 0
        ):
            arr = next(iter(_current_X_dict.values()))
            return np.full_like(arr, np.nan, dtype=np.float64)
        return np.nan
    n_val = _to_int_param(n, 10)
    s_val = _to_float_param(s, 2.0)
    m_val = _to_float_param(m, 20.0)
    return My.SAR(high, low, n_val, s_val, m_val)


def wr_func_auto(n=10):
    """WR指标包装器 - 自动从X_dict获取close, high, low，只接受周期参数"""
    close = _get_market_data("close")
    high = _get_market_data("high")
    low = _get_market_data("low")
    if close is None or high is None or low is None:
        if (
            _current_X_dict
            and isinstance(_current_X_dict, dict)
            and len(_current_X_dict) > 0
        ):
            arr = next(iter(_current_X_dict.values()))
            return np.full_like(arr, np.nan, dtype=np.float64)
        return np.nan
    n_val = _to_int_param(n, 10)
    return My.WR(close, high, low, n_val)


def bias_func_auto(period=6):
    """BIAS指标包装器 - 自动从X_dict获取close，只接受周期参数"""
    close = _get_market_data("close")
    if close is None:
        if (
            _current_X_dict
            and isinstance(_current_X_dict, dict)
            and len(_current_X_dict) > 0
        ):
            arr = next(iter(_current_X_dict.values()))
            return np.full_like(arr, np.nan, dtype=np.float64)
        return np.nan
    p_val = _to_int_param(period, 6)
    return My.BIAS(close, p_val)


def boll_upper_func_auto(n=20, p=2):
    """BOLL上轨指标包装器 - 自动从X_dict获取close，只接受周期参数"""
    close = _get_market_data("close")
    if close is None:
        if (
            _current_X_dict
            and isinstance(_current_X_dict, dict)
            and len(_current_X_dict) > 0
        ):
            arr = next(iter(_current_X_dict.values()))
            return np.full_like(arr, np.nan, dtype=np.float64)
        return np.nan
    n_val = _to_int_param(n, 20)
    p_val = _to_float_param(p, 2.0)
    return My.BOLL_UPPER(close, n_val, p_val)


def boll_lower_func_auto(n=20, p=2):
    """BOLL下轨指标包装器 - 自动从X_dict获取close，只接受周期参数"""
    close = _get_market_data("close")
    if close is None:
        if (
            _current_X_dict
            and isinstance(_current_X_dict, dict)
            and len(_current_X_dict) > 0
        ):
            arr = next(iter(_current_X_dict.values()))
            return np.full_like(arr, np.nan, dtype=np.float64)
        return np.nan
    n_val = _to_int_param(n, 20)
    p_val = _to_float_param(p, 2.0)
    return My.BOLL_LOWER(close, n_val, p_val)


def psy_func_auto(n=12):
    """PSY指标包装器 - 自动从X_dict获取close，只接受周期参数"""
    close = _get_market_data("close")
    if close is None:
        if (
            _current_X_dict
            and isinstance(_current_X_dict, dict)
            and len(_current_X_dict) > 0
        ):
            arr = next(iter(_current_X_dict.values()))
            return np.full_like(arr, np.nan, dtype=np.float64)
        return np.nan
    n_val = _to_int_param(n, 12)
    return My.PSY(close, n_val)


def cci_func_auto(n=14):
    """CCI指标包装器 - 自动从X_dict获取close, high, low，只接受周期参数"""
    close = _get_market_data("close")
    high = _get_market_data("high")
    low = _get_market_data("low")
    if close is None or high is None or low is None:
        if (
            _current_X_dict
            and isinstance(_current_X_dict, dict)
            and len(_current_X_dict) > 0
        ):
            arr = next(iter(_current_X_dict.values()))
            return np.full_like(arr, np.nan, dtype=np.float64)
        return np.nan
    n_val = _to_int_param(n, 14)
    return My.CCI(close, high, low, n_val)


def tr_func_auto():
    """TR指标包装器 - 自动从X_dict获取close, high, low，无需参数"""
    close = _get_market_data("close")
    high = _get_market_data("high")
    low = _get_market_data("low")
    if close is None or high is None or low is None:
        if (
            _current_X_dict
            and isinstance(_current_X_dict, dict)
            and len(_current_X_dict) > 0
        ):
            arr = next(iter(_current_X_dict.values()))
            return np.full_like(arr, np.nan, dtype=np.float64)
        return np.nan
    return My.TR(close, high, low)


def atr_func_auto(n=20):
    """ATR指标包装器 - 自动从X_dict获取close, high, low，只接受周期参数"""
    close = _get_market_data("close")
    high = _get_market_data("high")
    low = _get_market_data("low")
    if close is None or high is None or low is None:
        if (
            _current_X_dict
            and isinstance(_current_X_dict, dict)
            and len(_current_X_dict) > 0
        ):
            arr = next(iter(_current_X_dict.values()))
            return np.full_like(arr, np.nan, dtype=np.float64)
        return np.nan
    n_val = _to_int_param(n, 20)
    return My.ATR(close, high, low, n_val)


def bbi_func_auto(m1=3, m2=6, m3=12, m4=20):
    """BBI指标包装器 - 自动从X_dict获取close，只接受周期参数"""
    close = _get_market_data("close")
    if close is None:
        if (
            _current_X_dict
            and isinstance(_current_X_dict, dict)
            and len(_current_X_dict) > 0
        ):
            arr = next(iter(_current_X_dict.values()))
            return np.full_like(arr, np.nan, dtype=np.float64)
        return np.nan
    v1 = _to_int_param(m1, 3)
    v2 = _to_int_param(m2, 6)
    v3 = _to_int_param(m3, 12)
    v4 = _to_int_param(m4, 20)
    return My.BBI(close, v1, v2, v3, v4)


def adx_func_auto(m1=14, m2=6):
    """ADX指标包装器 - 自动从X_dict获取close, high, low，只接受周期参数"""
    close = _get_market_data("close")
    high = _get_market_data("high")
    low = _get_market_data("low")
    if close is None or high is None or low is None:
        if (
            _current_X_dict
            and isinstance(_current_X_dict, dict)
            and len(_current_X_dict) > 0
        ):
            arr = next(iter(_current_X_dict.values()))
            return np.full_like(arr, np.nan, dtype=np.float64)
        return np.nan
    v1 = _to_int_param(m1, 14)
    v2 = _to_int_param(m2, 6)
    return My.ADX(close, high, low, v1, v2)


def trix_func_auto(m=12):
    """TRIX指标包装器 - 自动从X_dict获取close，只接受周期参数"""
    close = _get_market_data("close")
    if close is None:
        if (
            _current_X_dict
            and isinstance(_current_X_dict, dict)
            and len(_current_X_dict) > 0
        ):
            arr = next(iter(_current_X_dict.values()))
            return np.full_like(arr, np.nan, dtype=np.float64)
        return np.nan
    v1 = _to_int_param(m, 12)
    return My.TRIX(close, v1)


def vr_func_auto(m1=26):
    """VR指标包装器 - 自动从X_dict获取close, vol，只接受周期参数"""
    close = _get_market_data("close")
    vol = _get_market_data("volume", "vol")
    if close is None or vol is None:
        if (
            _current_X_dict
            and isinstance(_current_X_dict, dict)
            and len(_current_X_dict) > 0
        ):
            arr = next(iter(_current_X_dict.values()))
            return np.full_like(arr, np.nan, dtype=np.float64)
        return np.nan
    v1 = _to_int_param(m1, 26)
    return My.VR(close, vol, v1)


def cr_func_auto(n=20):
    """CR指标包装器 - 自动从X_dict获取close, high, low，只接受周期参数"""
    close = _get_market_data("close")
    high = _get_market_data("high")
    low = _get_market_data("low")
    if close is None or high is None or low is None:
        if (
            _current_X_dict
            and isinstance(_current_X_dict, dict)
            and len(_current_X_dict) > 0
        ):
            arr = next(iter(_current_X_dict.values()))
            return np.full_like(arr, np.nan, dtype=np.float64)
        return np.nan
    n_val = _to_int_param(n, 20)
    return My.CR(close, high, low, n_val)


def emv_func_auto(n=14):
    """EMV指标包装器 - 自动从X_dict获取high, low, vol，只接受周期参数"""
    high = _get_market_data("high")
    low = _get_market_data("low")
    vol = _get_market_data("volume", "vol")
    if high is None or low is None or vol is None:
        if (
            _current_X_dict
            and isinstance(_current_X_dict, dict)
            and len(_current_X_dict) > 0
        ):
            arr = next(iter(_current_X_dict.values()))
            return np.full_like(arr, np.nan, dtype=np.float64)
        return np.nan
    n_val = _to_int_param(n, 14)
    return My.EMV(high, low, vol, n_val)


def dpo_func_auto(m1=20, m2=10):
    """DPO指标包装器 - 自动从X_dict获取close，只接受周期参数"""
    close = _get_market_data("close")
    if close is None:
        if (
            _current_X_dict
            and isinstance(_current_X_dict, dict)
            and len(_current_X_dict) > 0
        ):
            arr = next(iter(_current_X_dict.values()))
            return np.full_like(arr, np.nan, dtype=np.float64)
        return np.nan
    v1 = _to_int_param(m1, 20)
    v2 = _to_int_param(m2, 10)
    return My.DPO(close, v1, v2)


def brar_ar_func_auto(m1=26):
    """BRAR_AR指标包装器 - 自动从X_dict获取open, high, low，只接受周期参数"""
    open_price = _get_market_data("open")
    high = _get_market_data("high")
    low = _get_market_data("low")
    if open_price is None or high is None or low is None:
        if (
            _current_X_dict
            and isinstance(_current_X_dict, dict)
            and len(_current_X_dict) > 0
        ):
            arr = next(iter(_current_X_dict.values()))
            return np.full_like(arr, np.nan, dtype=np.float64)
        return np.nan
    v1 = _to_int_param(m1, 26)
    return My.BRAR_AR(open_price, high, low, v1)


def brar_br_func_auto(m1=26):
    """BRAR_BR指标包装器 - 自动从X_dict获取open, close, high, low，只接受周期参数"""
    open_price = _get_market_data("open")
    close = _get_market_data("close")
    high = _get_market_data("high")
    low = _get_market_data("low")
    if open_price is None or close is None or high is None or low is None:
        if (
            _current_X_dict
            and isinstance(_current_X_dict, dict)
            and len(_current_X_dict) > 0
        ):
            arr = next(iter(_current_X_dict.values()))
            return np.full_like(arr, np.nan, dtype=np.float64)
        return np.nan
    v1 = _to_int_param(m1, 26)
    return My.BRAR_BR(open_price, close, high, low, v1)


def difma_func_auto(n1=10, n2=50, m=10):
    """DIFMA指标包装器 - 自动从X_dict获取close，只接受周期参数"""
    close = _get_market_data("close")
    if close is None:
        if (
            _current_X_dict
            and isinstance(_current_X_dict, dict)
            and len(_current_X_dict) > 0
        ):
            arr = next(iter(_current_X_dict.values()))
            return np.full_like(arr, np.nan, dtype=np.float64)
        return np.nan
    v1 = _to_int_param(n1, 10)
    v2 = _to_int_param(n2, 50)
    v3 = _to_int_param(m, 10)
    return My.DIFMA(close, v1, v2, v3)


def mtm_func_auto(n=12):
    """MTM指标包装器 - 自动从X_dict获取close，只接受周期参数"""
    close = _get_market_data("close")
    if close is None:
        if (
            _current_X_dict
            and isinstance(_current_X_dict, dict)
            and len(_current_X_dict) > 0
        ):
            arr = next(iter(_current_X_dict.values()))
            return np.full_like(arr, np.nan, dtype=np.float64)
        return np.nan
    n_val = _to_int_param(n, 12)
    return My.MTM(close, n_val)


def mass_func_auto(n1=9, n2=25):
    """MASS指标包装器 - 自动从X_dict获取high, low，只接受周期参数"""
    high = _get_market_data("high")
    low = _get_market_data("low")
    if high is None or low is None:
        if (
            _current_X_dict
            and isinstance(_current_X_dict, dict)
            and len(_current_X_dict) > 0
        ):
            arr = next(iter(_current_X_dict.values()))
            return np.full_like(arr, np.nan, dtype=np.float64)
        return np.nan
    v1 = _to_int_param(n1, 9)
    v2 = _to_int_param(n2, 25)
    return My.MASS(high, low, v1, v2)


def roc_func_auto(n=12):
    """ROC指标包装器 - 自动从X_dict获取close，只接受周期参数"""
    close = _get_market_data("close")
    if close is None:
        if (
            _current_X_dict
            and isinstance(_current_X_dict, dict)
            and len(_current_X_dict) > 0
        ):
            arr = next(iter(_current_X_dict.values()))
            return np.full_like(arr, np.nan, dtype=np.float64)
        return np.nan
    n_val = _to_int_param(n, 12)
    return My.ROC(close, n_val)


def obv_func_auto():
    """OBV指标包装器 - 自动从X_dict获取close, vol，无需参数"""
    close = _get_market_data("close")
    vol = _get_market_data("volume", "vol")
    if close is None or vol is None:
        if (
            _current_X_dict
            and isinstance(_current_X_dict, dict)
            and len(_current_X_dict) > 0
        ):
            arr = next(iter(_current_X_dict.values()))
            return np.full_like(arr, np.nan, dtype=np.float64)
        return np.nan
    return My.OBV(close, vol)


def mfi_func_auto(n=14):
    """MFI指标包装器 - 自动从X_dict获取close, high, low, vol，只接受周期参数"""
    close = _get_market_data("close")
    high = _get_market_data("high")
    low = _get_market_data("low")
    vol = _get_market_data("volume", "vol")
    if close is None or high is None or low is None or vol is None:
        if (
            _current_X_dict
            and isinstance(_current_X_dict, dict)
            and len(_current_X_dict) > 0
        ):
            arr = next(iter(_current_X_dict.values()))
            return np.full_like(arr, np.nan, dtype=np.float64)
        return np.nan
    n_val = _to_int_param(n, 14)
    return My.MFI(close, high, low, vol, n_val)


def asi_func_auto(m1=26):
    """ASI指标包装器 - 自动从X_dict获取open, close, high, low，只接受周期参数"""
    open_price = _get_market_data("open")
    close = _get_market_data("close")
    high = _get_market_data("high")
    low = _get_market_data("low")
    if open_price is None or close is None or high is None or low is None:
        if (
            _current_X_dict
            and isinstance(_current_X_dict, dict)
            and len(_current_X_dict) > 0
        ):
            arr = next(iter(_current_X_dict.values()))
            return np.full_like(arr, np.nan, dtype=np.float64)
        return np.nan
    v1 = _to_int_param(m1, 26)
    return My.ASI(open_price, close, high, low, v1)


class FunctionSet:
    """
    函数集管理类

    管理遗传编程中可用的函数集合，支持动态增删函数

    属性:
        functions: 字典，存储函数名到(函数对象, 参数个数)的映射

    功能:
        - 注册新函数
        - 删除函数
        - 查询函数信息
        - 获取所有可用函数
    """

    def __init__(self):
        """
        初始化函数集，注册所有基础函数

        每个函数条目格式：'函数名': (函数对象, 参数个数)
        """
        self.functions = {
            # 基础数学运算（2个参数）
            "ADD": (My.ADD, 2),
            "SUB": (My.SUB, 2),
            "MUL": (My.MUL, 2),
            "DIV": (My.DIV, 2),
            # 单元函数（1个参数）
            "INV": (My.INV, 1),
            "SIGNEDPOWER": (My.SIGNEDPOWER, 1),
            'SIGMOID': (My.SIGMOID, 1),
            # 新增算子函数（1个参数）
            "ABS": (My.ABS, 1),  # 绝对值
            "LN": (My.LN, 1),  # 自然对数
            "SQRT": (My.SQRT, 1),  # 平方根
            "SIN": (My.SIN, 1),  # 正弦
            "COS": (My.COS, 1),  # 余弦
            "TAN": (My.TAN, 1),  # 正切
            # 新增算子函数（2个参数）
            "MAX": (My.MAX, 2),  # 最大值
            "MIN": (My.MIN, 2),  # 最小值
            "REF": (My.REF, 2),  # 引用
            "DIFF": (My.DIFF, 2),  # 差分
            "HHV": (My.HHV, 2),  # 最高值
            "HV": (My.HV, 2),  # 最高值（不包括当前K线）
            "LLV": (My.LLV, 2),  # 最低值
            "LV": (My.LV, 2),  # 最低值（不包括当前K线）
            "HHVBARS": (My.HHVBARS, 2),  # 最高值到当前周期数
            "LLVBARS": (My.LLVBARS, 2),  # 最低值到当前周期数
            "MA": (My.MA, 2),  # 简单移动平均
            "EMA": (My.EMA, 2),  # 指数移动平均
            "SMA": (My.SMA, 3),  # 中国式SMA（3个参数：x, d, m）
            "WMA": (My.WMA, 2),  # 加权移动平均
            "DMA": (My.DMA, 2),  # 动态移动平均
            "AVEDEV": (My.AVEDEV, 2),  # 平均绝对偏差
            "SLOPE": (My.SLOPE, 2),  # 线性回归斜率
            "FORCAST": (My.FORCAST, 2),  # 线性回归预测值
            "TOPRANGE": (My.TOPRANGE, 1),  # 最高价周期数
            "LOWRANGE": (My.LOWRANGE, 1),  # 最低价周期数
            # 时间序列函数（2个参数：数据+窗口）
            "RANK": (My.RANK, 2),
            "SCALE": (My.SCALE, 2),
            "TS_RANK": (My.TS_RANK, 2),
            "TS_ZSCORE": (My.TS_ZSCORE, 2),
            # 双变量时间序列函数（3个参数：数据1+数据2+窗口）
            "CORR": (My.CORR, 3),
            "COVA": (My.COVA, 3),
            "RANK_SUB": (My.RANK_SUB, 3),
            "RANK_DIV": (My.RANK_DIV, 3),
            # 高阶技术指标函数（自动从X_dict获取行情数据，只接受周期参数）
            "MACD_DIF": (macd_dif_func_auto, 2),  # short, long (close自动获取)
            "MACD_DEA": (macd_dea_func_auto, 3),  # short, long, m (close自动获取)
            "MACD_MACD": (macd_macd_func_auto, 3),  # short, long, m (close自动获取)
            "KDJ_K": (kdj_k_func_auto, 2),  # n, m1 (close, high, low自动获取)
            "KDJ_D": (kdj_d_func_auto, 3),  # n, m1, m2 (close, high, low自动获取)
            "KDJ_J": (kdj_j_func_auto, 3),  # n, m1, m2 (close, high, low自动获取)
            "RSI": (rsi_func_auto, 1),  # n (close自动获取)
            "SAR": (sar_func_auto, 3),  # n, s, m (high, low自动获取)
            "WR": (wr_func_auto, 1),  # n (close, high, low自动获取)，已注释
            "BIAS": (bias_func_auto, 1),  # period (close自动获取)
            "BOLL_UPPER": (boll_upper_func_auto, 2),  # n, p (close自动获取)
            "BOLL_LOWER": (boll_lower_func_auto, 2),  # n, p (close自动获取)
            "PSY": (psy_func_auto, 1),  # n (close自动获取)
            "CCI": (cci_func_auto, 1),  # n (close, high, low自动获取)
            "TR": (tr_func_auto, 0),  # 无参数 (close, high, low自动获取)
            "ATR": (atr_func_auto, 1),  # n (close, high, low自动获取)
            "BBI": (bbi_func_auto, 4),  # m1, m2, m3, m4 (close自动获取)
            "ADX": (adx_func_auto, 2),  # m1, m2 (close, high, low自动获取)
            "TRIX": (trix_func_auto, 1),  # m (close自动获取)
            "VR": (vr_func_auto, 1),  # m1 (close, vol自动获取)
            "CR": (cr_func_auto, 1),  # n (close, high, low自动获取)
            "EMV": (emv_func_auto, 1),  # n (high, low, vol自动获取)
            "DPO": (dpo_func_auto, 2),  # m1, m2 (close自动获取)
            "BRAR_AR": (brar_ar_func_auto, 1),  # m1 (open, high, low自动获取)
            "BRAR_BR": (brar_br_func_auto, 1),  # m1 (open, close, high, low自动获取)
            "DIFMA": (difma_func_auto, 3),  # n1, n2, m (close自动获取)
            "MTM": (mtm_func_auto, 1),  # n (close自动获取)
            "MASS": (mass_func_auto, 2),  # n1, n2 (high, low自动获取)
            "ROC": (roc_func_auto, 1),  # n (close自动获取)
            "OBV": (obv_func_auto, 0),  # 无参数 (close, vol自动获取)
            "MFI": (mfi_func_auto, 1),  # n (close, high, low, vol自动获取)
            "ASI": (asi_func_auto, 1),  # m1 (open, close, high, low自动获取)
        }

    def add_function(self, name: str, func: Callable, arity: int):
        """
        添加新函数到函数集

        参数:
            name: 函数名称
            func: 函数对象
            arity: 函数参数个数
        """
        self.functions[name] = (func, arity)

    def get(self, name: str):
        """
        获取函数信息

        参数:
            name: 函数名称

        返回:
            (函数对象, 参数个数) 元组
        """
        return self.functions[name]

    def all(self):
        """
        获取所有函数的副本

        返回:
            包含所有函数信息的字典副本
        """
        return self.functions.copy()

    def remove_function(self, name: str):
        """
        从函数集中删除函数

        参数:
            name: 要删除的函数名称
        """
        if name in self.functions:
            del self.functions[name]


def _is_all_nan_or_inf(value) -> bool:
    try:
        if value is None:
            return False
        if np.isscalar(value):
            return bool(np.isnan(value) or np.isinf(value))
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return False
            return bool(np.all(np.isnan(value)) or np.all(np.isinf(value)))
    except Exception:
        return False
    return False

# 描述性统计
def _summarize_value(value) -> str:
    try:
        if value is None:
            return "None"
        if np.isscalar(value):
            return f"scalar={value}"
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return f"array(shape={value.shape}, dtype={value.dtype}, size=0)"
            arr = value
            size = arr.size
            nan_cnt = np.isnan(arr).sum()
            inf_cnt = np.isinf(arr).sum()
            finite_mask = np.isfinite(arr)
            if np.any(finite_mask):
                finite_vals = arr[finite_mask]
                min_val = float(np.min(finite_vals))
                max_val = float(np.max(finite_vals))
            else:
                min_val = np.nan
                max_val = np.nan
            return (
                f"array(shape={arr.shape}, dtype={arr.dtype}, "
                f"nan={nan_cnt}/{size}({nan_cnt / size:.2%}), "
                f"inf={inf_cnt}/{size}({inf_cnt / size:.2%}), "
                f"min={min_val}, max={max_val})"
            )
        return f"type={type(value).__name__}"
    except Exception as e:
        return f"summary_error={e}"


def _trace_all_nan_inf(node, result, args, trace_state):
    if trace_state is None:
        return
    if trace_state.get("mode") == "first" and trace_state.get("reported"):
        return
    if not _is_all_nan_or_inf(result):
        return
    expr_str = (
        node.to_str() if hasattr(node, "to_str") else str(getattr(node, "name", "N/A"))
    )
    path = trace_state.get("path", [])
    path_str = " > ".join(path) if path else ""
    log_print("警告: pred全为NaN或inf时的溯源:")
    log_print(f"  触发节点: {getattr(node, 'name', 'N/A')}")
    log_print(f"  表达式: {expr_str}")
    if path_str:
        log_print(f"  路径: {path_str}")
    if args is not None:
        children = getattr(node, "children", []) or []
        for idx, arg in enumerate(args):
            child_name = "N/A"
            child_expr = "N/A"
            if idx < len(children):
                child = children[idx]
                if getattr(child, "value", None) is not None:
                    child_name = str(child.value)
                else:
                    child_name = str(getattr(child, "name", "N/A"))
                if hasattr(child, "to_str"):
                    child_expr = child.to_str()
            log_print(f"  子项[{idx}] 名称: {child_name}")
            log_print(f"  子项[{idx}] 表达式: {child_expr}")
            log_print(f"  子项[{idx}] 参数: {_summarize_value(arg)}")
    log_print(f"  节点输出: {_summarize_value(result)}")
    trace_state["reported"] = True


class Node:
    """
    表达式树节点类

    表示因子表达式的树形结构，每个节点可以是：
    1. 叶子节点：变量或常数
    2. 内部节点：函数调用

    属性:
        name: 节点名称（函数名或变量名）
        children: 子节点列表
        value: 节点值（变量名或常数值，仅叶子节点使用）
    """

    def __init__(self, name: str, children: List["Node"] = None, value: Any = None):
        """
        初始化节点

        参数:
            name: 节点名称
            children: 子节点列表
            value: 节点值（变量名或常数）
        """
        self.name = name
        self.children = children or []
        self.value = value  # 变量名（str）或常数（number）

    def evaluate(self, X, function_set, trace_state=None):
        """
        递归评估表达式树

        参数:
            X: 输入数据，可以是字典{变量名: 数据}或其他格式
            function_set: 函数集对象

        返回:
            result：表达式在当前输入数据下的计算结果

        评估逻辑:
            1. 叶子节点：返回变量值或常数
            2. 内部节点：递归计算子节点，然后调用函数
        """
        # 设置全局X_dict，用于验证行情数据参数
        if isinstance(X, dict):
            _set_current_X_dict(X)

        # 处理叶子节点（变量或常数）
        if self.value is not None:
            if trace_state is not None:
                trace_path = trace_state.setdefault("path", [])
                trace_path.append(str(self.value))
            try:
                if isinstance(self.value, str):  # 变量节点
                    if isinstance(X, dict):
                        result = X.get(self.value, 0)
                    elif hasattr(X, "__getitem__"):
                        try:
                            result = X[self.value]
                        except Exception:
                            result = 0
                    else:
                        result = 0
                else:  # 常数节点
                    if isinstance(X, dict) and len(X) > 0:
                        # 返回与数据同形状的常数数组，方便维度匹配
                        arr = next(iter(X.values()))
                        # 创建一个与self.value数据类型一样的全是arr的数组
                        result = np.full_like(arr, self.value, dtype=np.float64)
                    else:
                        result = self.value
                _trace_all_nan_inf(self, result, None, trace_state)
                return result
            finally:
                if trace_state is not None:
                    trace_path.pop()

        # 处理内部节点（函数调用）
        if trace_state is not None:
            trace_path = trace_state.setdefault("path", [])
            # setdefault() 是 Python 字典（dict）的一个方法，用于安全地获取指定键的值，如果键不存在，则插入一个默认值并返回该默认值。
            trace_path.append(str(self.name))
        try:
            func, arity = function_set.get(self.name)
            # 递归计算所有子节点，args用于存储当前节点所有子节点的计算结果。
            args = [
                child.evaluate(X, function_set, trace_state=trace_state)
                for child in self.children
            ]

            # 高阶技术指标函数列表（这些函数的参数应该是标量，不需要扩展为二维数组）
            high_level_indicators = [
                "MACD_DIF",
                "MACD_DEA",
                "MACD_MACD",
                "KDJ_K",
                "KDJ_D",
                "KDJ_J",
                "RSI",
                "SAR",
                "WR",
                "BIAS",
                "BOLL_UPPER",
                "BOLL_LOWER",
                "PSY",
                "CCI",
                "TR",
                "ATR",
                "BBI",
                "ADX",
                "TRIX",
                "VR",
                "CR",
                "EMV",
                "DPO",
                "BRAR_AR",
                "BRAR_BR",
                "DIFMA",
                "MTM",
                "MASS",
                "ROC",
                "MFI",
                "ASI",
                "BARSLAST",
            ]

            # 如果是高阶技术指标函数，将参数鲁棒转换为标量
            if self.name in high_level_indicators:
                args = [_safe_scalar_param(arg, default=10.0) for arg in args]

            try:
                # 调用函数
                result = func(*args) # *的作用是将这个可迭代对象解包成独立的位置参数，依次传递给函数。

                # 计算结果后处理：确保返回合法的数值
                if result is None or (isinstance(result, float) and np.isnan(result)):
                    if isinstance(X, dict) and len(X) > 0:
                        arr = next(iter(X.values()))
                        result = np.full_like(arr, np.nan, dtype=np.float64)
                    else:
                        result = np.nan
                elif np.isscalar(result):
                    result = result
                elif isinstance(result, np.ndarray) and result.shape == ():
                    result = float(result)
                elif isinstance(result, np.ndarray) and result.dtype == object:
                    try:
                        result = result.astype(np.float64)
                    except Exception:
                        arr = (
                            next(iter(X.values()))
                            if isinstance(X, dict) and len(X) > 0
                            else 0
                        )
                        result = np.full_like(arr, np.nan, dtype=np.float64)

                _trace_all_nan_inf(self, result, args, trace_state)
                return result
            except Exception as e:
                # 函数调用失败时返回NaN
                # 记录错误信息以便调试（使用debug级别，避免过多日志输出）
                log_print(
                    f"Node.evaluate函数执行异常: {function_set}{self.name}, 错误: {str(e)}, 参数数量: {len(self.children)}"
                )
                logger.debug(
                    f"函数执行异常: {self.name}, 错误: {str(e)}, 参数数量: {len(self.children)}"
                )

                if isinstance(X, dict) and len(X) > 0:
                    arr = next(iter(X.values()))
                    result = np.full_like(arr, np.nan, dtype=np.float64)
                else:
                    result = np.nan
                _trace_all_nan_inf(self, result, args, trace_state)
                return result
        finally:
            if trace_state is not None:
                trace_path.pop()

    def to_str(self) -> str:
        """
        将表达式树转换为字符串表示

        返回:
            表达式的字符串形式

        示例:
            add(mul(x, 2), y) -> "add(mul(x, 2), y)"
        """
        try:
            if self.value is not None:
                return str(self.value)
            return f"{self.name}({', '.join([c.to_str() for c in self.children])})"
        except Exception as e:
            log_print(f"Node.to_str函数异常: {e}")
            raise

    def depth(self) -> int:
        """
        计算表达式树的深度

        返回:
            树的最大深度

        用途:
            控制表达式复杂度，避免过深的树结构
        """
        try:
            if not self.children:
                return 1
            return 1 + max(child.depth() for child in self.children)
        except Exception as e:
            log_print(f"Node.depth函数异常: {e}")
            raise

    def size(self) -> int:
        """
        计算表达式树的节点总数

        返回:
            树中节点的总数量

        用途:
            控制表达式复杂度，用于简约惩罚
        """
        try:
            return 1 + sum(child.size() for child in self.children)
        except Exception as e:
            log_print(f"Node.size函数异常: {e}")
            raise

    def get_variables(self) -> set:
        """
        提取表达式中使用的所有变量（特征）名称

        返回:
            变量名称的集合
        """
        try:
            variables = set()
            if self.value is not None and isinstance(self.value, str):
                # 检查是否是变量（在variable_names中）
                # 这里假设所有字符串值都是变量名
                variables.add(self.value)
            for child in self.children:
                variables.update(child.get_variables())
            return variables
        except Exception as e:
            log_print(f"Node.get_variables函数异常: {e}")
            raise


class GeneticProgrammer:
    """
    遗传编程主类

    实现遗传编程算法用于自动化因子挖掘

    核心流程:
    1. 初始化随机种群
    2. 评估适应度
    3. 选择、交叉、变异产生新一代
    4. 重复进化直到收敛

    主要特性:
    - 多种遗传操作（交叉、子树变异、点变异等）
    - 锦标赛选择
    - 精英保留策略
    - 简约惩罚（控制树复杂度）
    - 进度监控和回调
    """

    def __init__(
        self,
        generations: int = 30,  # 进化代数
        population_size: int = 200,  # 种群规模
        n_components: int = 5,  # 保留的最优个体数量，整个进化过程完成后
        hall_of_fame: int = 6,  # 精英保留数量，每次进化过程中的优秀个体
        function_set: FunctionSet = None,  # 函数集
        parsimony_coefficient: float = 0.001,  # 简约系数（惩罚复杂树）
        tournament_size: int = 4,  # 锦标赛规模
        random_state: Optional[int] = None,  # 随机数种子（None 表示随机）
        init_depth: tuple = (4, 6),  # 初始树深度范围
        suit_size:  tuple = (6, 12),   # 合适的表达树的节点数上下界
        const_range: Optional[tuple] = (-2, 2),  # 常数范围
        ts_window: int = 30,  # 时间窗口范围
        # 遗传操作概率
        p_crossover: float = 0.45,  # 交叉概率
        # p_subtree_mutation: float = 0.25,  # 子树变异概率
        p_subtree_mutation: float = 0.45,  # 子树变异概率
        # p_hoist_mutation: float = 0.05,  # 提升变异概率
        p_hoist_mutation: float = 0.18,  # 提升变异概率
        # p_point_mutation: float = 0.08,  # 点变异概率
        p_point_mutation: float = 0.18,  # 点变异概率
        # p_point_replace: float = 0.3,  # 点替换概率
        p_point_replace: float = 0.5,  # 点替换概率
        immigration_rate: float = 0.20,  # 每代注入随机个体比例
        variable_names: List[str] = None,  # 变量名列表
        max_program_size: Optional[int] = None,  # 进化过程最大节点数限制
        max_best_program_size: Optional[int] = None,  # 最终最优个体最大节点数限制
        ic_objective: str = "max",  # IC优化方向: "max" 或 "min"
        allow_const_in_function: Optional[
            Dict[str, Union[bool, List[bool]]]
        ] = None,  # 按函数+参数控制是否允许常数
        stagnation_threshold: int = 10,  # 停滞检测阈值，最近多少代没有显著提升则认为停滞
        min_improvement: float = 0.001,  # 最小显著提升阈值，用于检测停滞
        max_restarts: int = 3,  # 最大自动重启次数
    ):
        """
        初始化遗传编程器

        参数说明详见类属性注释
        """
        self.generations = generations
        self.population_size = population_size
        self.n_components = n_components
        self.hall_of_fame = hall_of_fame
        self.function_set = function_set or FunctionSet()
        self.parsimony_coefficient = parsimony_coefficient
        self.tournament_size = tournament_size
        if isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
            # RandomState 不保存“原始输入seed”，这里记录当前内部状态对应的整数值
            self.random_seed = int(self.random_state.get_state()[1][0])
        elif random_state is None:
            # 显式生成整数seed，便于复现与日志追踪
            self.random_seed = int(np.random.randint(0, 2**32 - 1, dtype=np.uint32))
            self.random_state = np.random.RandomState(self.random_seed)
        else:
            self.random_seed = int(random_state)
            self.random_state = np.random.RandomState(self.random_seed)
        log_print(f"本次使用的随机数种子为: {self.random_seed}")
        self.init_depth = init_depth
        self.suit_size = suit_size
        self.const_range = const_range
        self.ts_window = ts_window
        self.p_crossover = p_crossover
        self.p_subtree_mutation = p_subtree_mutation
        self.p_hoist_mutation = p_hoist_mutation
        self.p_point_mutation = p_point_mutation
        self.p_point_replace = p_point_replace
        self.variable_names = variable_names or ["X"]
        self.best_programs_ = []  # 存储最优程序
        self.max_program_size = max_program_size
        self.max_best_program_size = max_best_program_size
        if ic_objective not in ("max", "min"):
            raise ValueError("ic_objective 必须是 'max' 或 'min'")
        self.ic_objective = ic_objective
        self.allow_const_in_function = allow_const_in_function
        self._resample_count = 0
        self.immigration_rate = immigration_rate
        self.stagnation_threshold = stagnation_threshold
        self.min_improvement = min_improvement
        self.max_restarts = max_restarts

        # 初始化时函数权重选择
        self._weighted_functions = {}
        self._cumulative_weights = {}
        self._validate_probabilities()

    def _validate_probabilities(self):
        total = (
            float(self.p_crossover)
            + float(self.p_subtree_mutation)
            + float(self.p_hoist_mutation)
            + float(self.p_point_mutation)
        )
        if total > 1.0 + 1e-12:
            raise ValueError(f"遗传操作概率之和不能大于1，当前为 {total:.4f}。")
        if min(
            self.p_crossover,
            self.p_subtree_mutation,
            self.p_hoist_mutation,
            self.p_point_mutation,
        ) < 0:
            raise ValueError("遗传操作概率不能为负数。")

    def fit(
        self,
        fitness_func: Callable,  # 适应度函数
        fitness_args: tuple = (),  # 适应度函数额外参数
        fitness_kwargs: dict = None,  # 适应度函数关键字参数
        progress_callback: Callable = None,  # 进度回调函数
    ):
        """
        执行遗传编程进化过程

        参数:
            fitness_func: 适应度评估函数，接收个体并返回适应度值
            fitness_args: 适应度函数的额外位置参数
            fitness_kwargs: 适应度函数的关键字参数
            progress_callback: 进度回调函数，接收(代数, 平均长度, 平均适应度, 最优长度, 最优适应度)

        返回:
            self: 训练后的遗传编程器对象
        """
        fitness_kwargs = fitness_kwargs or {}
        if "function_set" not in fitness_kwargs:
            fitness_kwargs["function_set"] = self.function_set

        # 初始化种群：生成随机表达式树
        population = [self._random_program() for _ in range(self.population_size)]

        gen = 0
        fitness_history = []
        stagnation_threshold = self.stagnation_threshold
        max_restarts = self.max_restarts
        restart_count = 0
        best_fitness_overall = -np.inf
        best_program_info = None
        all_best_programs = []



        # 如果没有传入进度回调，使用默认的进度显示，回调不止是可以显示，还可以执行其他传入的函数
        if progress_callback is None:
            start_time = time.time()

            def default_progress_callback(
                gen,
                avg_len,
                avg_fit,
                best_len,
                best_fit,
                avg_ic=None,
                avg_ir=None,
                best_ic=None,
                best_ir=None,
                best_expr=None,
                restart_count=None,
                avg_ic_train=None,
                avg_ir_train=None,
                avg_ic_test=None,
                avg_ir_test=None,
                best_ic_train=None,
                best_ir_train=None,
                best_ic_test=None,
                best_ir_test=None,
                best_valid_ts=None,
                best_valid_ts_test=None,
            ):
                """
                默认进度显示函数

                显示训练进度，包括代数、平均/最优适应度、剩余时间等信息
                """
                if gen == 0 :
                    table_width = 206
                    if restart_count is not None and restart_count > 0:
                        log_print(
                            "\n" + "=" * ((table_width-14)//2) + f" 第 {restart_count} 轮重启 " + "=" * ((table_width-10)//2)
                        )
                    log_print("开始遗传规划因子挖掘...")
                    log_print("=" * table_width)


                    log_print(
                        "|       种群信息       |               最优个体               |                 因子质量                 |"
                    )
                    log_print("\n" + "-" * table_width)
                    log_print(
                        "轮次  代数    平均长度    平均适应度    最优长度    最优适应度      平均IC(训/测)       平均IR(训/测)       最优IC(训/测)       最优IR(训/测)   最优有效时点(训/测)   剩余时间  最优表达式"
                    )
                    log_print("\n" + "-" * table_width)


                # 处理-inf/nan显示
                def _fmt_metric(val, width=7):
                    if val is None:
                        return " " * (width - 3) + "nan"
                    if np.isfinite(val):
                        return f"{val:{width}.4f}"
                    return " " * (width - 3) + "nan"

                # avg_fit总宽度为7，保留4位小数（浮点数）
                avg_fit_str = f"{avg_fit:7.4f}" if avg_fit != -np.inf else "   -inf"
                best_fit_str = f"{best_fit:7.4f}" if best_fit != -np.inf else "   -inf"
                avg_ic_train = avg_ic if avg_ic_train is None else avg_ic_train
                avg_ir_train = avg_ir if avg_ir_train is None else avg_ir_train
                best_ic_train = best_ic if best_ic_train is None else best_ic_train
                best_ir_train = best_ir if best_ir_train is None else best_ir_train
                avg_ic_pair_str = f"{_fmt_metric(avg_ic_train)}/{_fmt_metric(avg_ic_test)}"
                avg_ir_pair_str = f"{_fmt_metric(avg_ir_train)}/{_fmt_metric(avg_ir_test)}"
                best_ic_pair_str = f"{_fmt_metric(best_ic_train)}/{_fmt_metric(best_ic_test)}"
                best_ir_pair_str = f"{_fmt_metric(best_ir_train)}/{_fmt_metric(best_ir_test)}"
                best_valid_ts_pair_str = f"{int(best_valid_ts or 0):>4d}/{int(best_valid_ts_test or 0):>4d}"

                # 计算剩余时间
                elapsed = time.time() - start_time
                if gen > 0:
                    time_per_gen = elapsed / (gen + 1)
                    time_left = (self.generations - gen - 1) * time_per_gen
                    time_str = (
                        f"{time_left / 60:.2f}m"
                        if time_left > 60
                        else f"{time_left:.2f}s"
                    )
                else:
                    time_str = "?"

                expr_str = best_expr if best_expr else ""
                iter_str = f"{restart_count + 1:3d}" if restart_count is not None else "  -"
                log_print(
                    f"{iter_str:>4} {gen + 1:>4d} {avg_len:>11.1f} {avg_fit_str:>13} "
                    f"{best_len:>11d} {best_fit_str:>13} {avg_ic_pair_str:>17} {avg_ir_pair_str:>17} "
                    f"{best_ic_pair_str:>17} {best_ir_pair_str:>17} {best_valid_ts_pair_str:>18} {time_str:>9}  {expr_str}"
                )

                if gen == self.generations - 1 or restart_count >= max_restarts:
                    log_print("-" * 206)

            progress_callback = default_progress_callback

        # 主进化循环
        # details 是一个包含个体详细绩效信息的字典
        while gen < self.generations and restart_count < max_restarts + 1:
            # 评估种群适应度
            fitness_details = self._evaluate_population(
                population,
                fitness_func,
                fitness_args,
                fitness_kwargs,
                return_details=True,
            )
            raw_fitness = [detail.get("fitness", -np.inf) for detail in fitness_details]

            # 应用简约惩罚：不符合预期复杂度的树会被惩罚
            fitnesses = [
                raw_fitness[i] - self._calc_parsimony(population[i])
                for i in range(len(population))
            ]

            # 统计当前代信息
            # 计算有效适应度的平均值（排除 -inf）
            valid_fitnesses = [f for f in fitnesses if f != -np.inf]
            # 列表表达式最终输出一个列表，以上代码等价于:
            # valid_fitnesses = []
            # for f in fitnesses:
            #     if f != -np.inf:          # 如果f不是负无穷
            #         valid_fitnesses.append(f)  # 添加到新列表

            if len(valid_fitnesses) > 0:
                avg_fit = np.mean(valid_fitnesses)
            else:
                avg_fit = -np.inf

            best_idx = np.argmax(fitnesses)
            best_fit = fitnesses[best_idx]
            avg_size = np.mean([p.size() for p in population])
            best_size = population[best_idx].size()
            best_expr = population[best_idx].to_str()

            fitness_history.append(best_fit)

            if self._check_fitness_stagnation(
                fitness_history,
                stagnation_threshold=stagnation_threshold,
                min_improvement=self.min_improvement,
            ):
                log_print(f"检测到适应度停滞 (代数: {gen + 1})，当前最优适应度: {fitness_history[-1]:.6f}")
                if best_program_info is not None:
                    all_best_programs.append(best_program_info) # 第n代的进化过程加的是第n-1代最优的
                if restart_count < max_restarts:
                    restart_count += 1
                    log_print(f"开始第 {restart_count} 次自动重启...")
                    population, fitness_history, gen, = self._restart_evolution()
                    fitness_details = self._evaluate_population(
                        population,
                        fitness_func,
                        fitness_args,
                        fitness_kwargs,
                        return_details=True,
                    )
                    raw_fitness = [detail.get("fitness", -np.inf) for detail in fitness_details]

                    # 应用简约惩罚：不符合预期复杂度的树会被惩罚
                    fitnesses = [
                        raw_fitness[i] - self._calc_parsimony(population[i])
                        for i in range(len(population))
                    ]
                    
                    # 重启后需要重新计算 best_idx 和相关变量
                    best_idx = np.argmax(fitnesses)
                    best_fit = fitnesses[best_idx]
                    best_size = population[best_idx].size()
                    best_expr = population[best_idx].to_str()
                    
                    log_print(f"重启完成，继续主循环...")
                else:
                    log_print("已达到最大重启次数，停止重启")
                    break

            # 计算IC/IR统计
            def _mean_metric(details, key):
                values = [d.get(key, np.nan) for d in details]
                values = [v for v in values if np.isfinite(v)]
                return np.mean(values) if len(values) > 0 else np.nan

            avg_ic = _mean_metric(fitness_details, "mean_ic")
            avg_ir = _mean_metric(fitness_details, "icir")
            avg_ic_test = _mean_metric(fitness_details, "mean_ic_test")
            avg_ir_test = _mean_metric(fitness_details, "icir_test")
            best_detail = (
                fitness_details[best_idx]
                if 0 <= best_idx < len(fitness_details)
                else {}
            )
            best_ic = best_detail.get("mean_ic", np.nan)
            best_ir = best_detail.get("icir", np.nan)
            best_ic_test = best_detail.get("mean_ic_test", np.nan)
            best_ir_test = best_detail.get("icir_test", np.nan)
            best_valid_ts = best_detail.get("valid_ts", 0)
            best_valid_ts_test = best_detail.get("valid_ts_test", 0)

            if best_fit > best_fitness_overall:
                best_fitness_overall = best_fit
                best_program_info = {
                    "program": population[best_idx],
                    "fitness": best_fit,
                    "expr": best_expr,
                    "ic": best_ic,
                    "ir": best_ir,
                    "ic_test": best_ic_test,
                    "ir_test": best_ir_test,
                    "valid_ts": best_valid_ts,
                    "valid_ts_test": best_valid_ts_test,
                }

            # 调用进度回调
            if progress_callback:
                try:
                    progress_callback(
                        gen,
                        avg_size,
                        avg_fit,
                        best_size,
                        best_fit,
                        avg_ic,
                        avg_ir,
                        best_ic,
                        best_ir,
                        best_expr,
                        restart_count=restart_count,
                        avg_ic_train=avg_ic,
                        avg_ir_train=avg_ir,
                        avg_ic_test=avg_ic_test,
                        avg_ir_test=avg_ir_test,
                        best_ic_train=best_ic,
                        best_ir_train=best_ir,
                        best_ic_test=best_ic_test,
                        best_ir_test=best_ir_test,
                        best_valid_ts=best_valid_ts,
                        best_valid_ts_test=best_valid_ts_test,
                    )
                except TypeError:
                    try:
                        progress_callback(
                            gen,
                            avg_size,
                            avg_fit,
                            best_size,
                            best_fit,
                            avg_ic,
                            avg_ir,
                            best_ic,
                            best_ir,
                            best_expr,
                        )
                    except TypeError:
                        progress_callback(gen, avg_size, avg_fit, best_size, best_fit)

            # 精英保留：保留最优个体到下一代（降低精英比例以增强探索）
            elite_count = min(self.hall_of_fame, self.population_size)
            elite_indices = np.argsort(fitnesses)[-elite_count:]
            elites = [population[i] for i in elite_indices]

            # 生成下一代种群
            new_population = elites.copy()
            # 移民机制：每代注入随机个体
            if self.immigration_rate and self.immigration_rate > 0:
                max_immigrants = self.population_size - len(new_population)
                num_immigrants = int(self.population_size * self.immigration_rate)
                if num_immigrants > max_immigrants:
                    num_immigrants = max_immigrants
                if num_immigrants > 0:
                    new_population.extend(
                        self._random_program() for _ in range(num_immigrants)
                    )
            while len(new_population) < self.population_size:
                # 锦标赛选择父代
                parent = self._tournament(population, fitnesses)
                # 应用遗传操作产生子代
                child = self._mutate_or_crossover(parent, population)
                # 控制树深度：过深则剪枝
                if child.depth() > 10:
                    child = self._hoist_mutation(child)
                new_population.append(child)

            population = new_population

            gen += 1

        if best_program_info is not None:
            all_best_programs.append(best_program_info)

        log_print(f"进化完成，共发现 {len(all_best_programs)} 个候选因子，开始最终评估...")

        all_programs = [info["program"] for info in all_best_programs]
        all_fitness = self._evaluate_population(
            all_programs, fitness_func, fitness_args, fitness_kwargs
        )

        combined_fitness = []
        for i, info in enumerate(all_best_programs):
            combined_fitness.append(all_fitness[i])

        best_indices = np.argsort(combined_fitness)[-self.n_components:]
        self.best_programs_ = [all_programs[i] for i in best_indices]

        log_print(f"最终筛选完成，选择了 {len(self.best_programs_)} 个最优因子")
        return self

    def _resample_if_oversize(self, program: Node) -> Node:
        """
        超限则直接丢弃并重采样（不做hoist）
        """
        if self.max_program_size is None:
            return program
        if program.size() <= self.max_program_size:
            return program
        self._resample_count += 1
        for _ in range(30):
            candidate = self._random_program()
            if candidate.size() <= self.max_program_size:
                return candidate
        logger.warning(
            "重采样失败，保留超限个体 size=%s > max=%s",
            program.size(),
            self.max_program_size,
        )
        return program

    def _calc_parsimony(self, prog: Node) -> float:
        """
        计算解析度惩罚项

        参数:
            prog: 待评估的程序树

        返回:
            float: 解析度惩罚项值
        """
        suit_lb, suit_ub = self.suit_size
        parsimony_coefficient = self.parsimony_coefficient
        if prog.size() < suit_lb:
            return parsimony_coefficient * (prog.size() - suit_lb) ** 2
        elif suit_lb <= prog.size() <= suit_ub:
            return 0
        elif suit_ub < prog.size():
            return parsimony_coefficient * (prog.size() - suit_ub) ** 2
        return 0

    def _check_fitness_stagnation(
        self,
        fitness_history: List[float],
        stagnation_threshold: int = 10,
        min_improvement: float = 1e-6,
    ) -> bool:
        """
        检测最近若干代最优适应度是否没有显著改善
        """
        if len(fitness_history) < stagnation_threshold + 1:
            return False

        recent = fitness_history[-(stagnation_threshold + 1):]

        # 如果最近窗口内全是无效值，视为停滞
        finite_recent = [v for v in recent if np.isfinite(v)]
        if len(finite_recent) == 0:
            return True

        if self.ic_objective == "max":
            improvement = max(recent[1:]) - recent[0]
            return improvement < min_improvement
        else:
            improvement = recent[0] - min(recent[1:])
            return improvement < min_improvement

    # 重新开始进化函数
    def _restart_evolution(self):
        """
        重新初始化种群和相关变量（自动重启用）

        参数:

        返回:
            tuple: (新种群, 空适应度历史, 最低适应度, None)
        """
        log_print("检测到适应度长期停滞，准备重新初始化种群...")
        self._resample_count = 0
        new_population = [self._random_program() for _ in range(self.population_size)]
        return new_population, [], 0

    def _select_function_weighted(self, target_arity=None):
        """
        加权选择函数，优先选择 sigmoid、rank、scale、ts_zscore 和高阶技术指标
        
        参数:
            target_arity: 目标参数个数（None表示不限制）
        
        返回:
            (func_name, func, arity) 元组
        """
        # 初始化缓存（在类初始化时执行一次）
        if not hasattr(self, '_weight_cache'):
            self._weight_cache = {}
        
        # 构建缓存键
        cache_key = str(target_arity) if target_arity is not None else "all"
        
        # 检查缓存
        if cache_key not in self._weight_cache:
            items = list(self.function_set.all().items())
            weighted_items = [] # 用于存储获取各函数权重后的 (func_name, func, arity, weight) 元组
            
            # 优先使用的函数及其权重
            # preferred_functions = {}  # 不加权
            preferred_functions = {
                "RANK": 5.0,
                "SCALE": 5.0,
                "TS_ZSCORE": 5.0,
                "MACD_DIF": 3.0,
                "MACD_DEA": 3.0,
                "MACD_MACD": 3.0,
                "KDJ_K": 3.0,
                "KDJ_D": 3.0,
                "KDJ_J": 3.0,
                "RSI": 3.0,
                "SAR": 3.0,
                "WR": 3.0,
                "BIAS": 3.0,
                "BOLL_UPPER": 3.0,
                "BOLL_LOWER": 3.0,
                "PSY": 3.0,
                "CCI": 3.0,
                "TR": 3.0,
                "ATR": 3.0,
                "BBI": 3.0,
                "ADX": 3.0,
                "TRIX": 3.0,
                "VR": 3.0,
                "CR": 3.0,
                "EMV": 3.0,
                "DPO": 3.0,
                "BRAR_AR": 3.0,
                "BRAR_BR": 3.0,
                "DIFMA": 3.0,
                "MTM": 3.0,
                "MASS": 3.0,
                "ROC": 3.0,
                "OBV": 3.0,
                "MFI": 3.0,
                "ASI": 3.0,
                "ADD": 2.0,
                "SUB": 2.0,
                "MUL": 2.0,
                "DIV": 2.0,
            }
            
            # 构建加权列表
            for func_name, (func, arity) in items:
                if target_arity is not None and arity != target_arity:
                    continue
                weight = preferred_functions.get(func_name, 1.0)
                weighted_items.append((func_name, func, arity, weight))
            
            if not weighted_items:
                # 如果没有匹配的函数，返回随机函数
                idx = self.random_state.randint(0, len(items))
                func_name, (func, arity) = items[idx]
                self._weight_cache[cache_key] = {
                    'items': [(func_name, func, arity, 1.0)],
                    'cumulative': [1.0]
                }
            else:
                # 计算总权重和累积权重
                total_weight = sum(weight for _, _, _, weight in weighted_items)
                cumulative = []
                current = 0.0
                for item in weighted_items:
                    current += item[3]
                    cumulative.append(current / total_weight)
                
                self._weight_cache[cache_key] = {
                    'items': weighted_items,
                    'cumulative': cumulative
                }
        
        # 使用缓存进行快速选择
        cache_entry = self._weight_cache[cache_key]
        weighted_items = cache_entry['items']
        cumulative = cache_entry['cumulative']
        
        # 二分查找优化选择过程
        r = self.random_state.rand()
        left, right = 0, len(cumulative)
        
        while left < right:
            mid = (left + right) // 2
            if cumulative[mid] < r:
                left = mid + 1
            else:
                right = mid
        
        idx = left if left < len(cumulative) else len(cumulative) - 1
        return weighted_items[idx][0], weighted_items[idx][1], weighted_items[idx][2]

    def _random_program(self, method="full"):
        """
        生成随机表达式树程序

        参数:
            method: 生成方法
                - 'grow': 生长法（随机深度）
                - 'full': 完全法（固定深度）
                - 'half_and_half': 混合法（随机选择上述两种）

        返回:
            随机生成的表达式树根节点

        生成策略:
            - 叶子节点：变量或常数
            - 内部节点：函数调用
            - 特殊处理时间序列函数的窗口参数
        """
        variables = self.variable_names
        min_depth, max_depth = self.init_depth
        depth = self.random_state.randint(min_depth, max_depth + 1)

        if method == "grow":

            def grow(d, parent_func_name=None, arg_index=None):
                # 到达最大深度则，或随机决定生成叶子节点
                if d == 0 or (d > 0 and self.random_state.rand() < 0.5):
                    # 根节点禁止常数
                    if parent_func_name is None:
                        var_name = self.random_state.choice(variables)
                        return Node(name=var_name, value=var_name)
                    # 检查在父函数的指定参数位置是否允许生成常数
                    if not self._allow_const_at(parent_func_name, arg_index):
                        var_name = self.random_state.choice(variables)
                        return Node(name=var_name, value=var_name)
                    if self.const_range is not None and self.random_state.rand() < 0.5:
                        # 生成变量节点
                        var_name = self.random_state.choice(variables)
                        return Node(name=var_name, value=var_name)
                    elif self.const_range is not None:
                        # 生成常数节点
                        const_val = self.random_state.uniform(*self.const_range)
                        return Node(name="const", value=const_val)
                    else:
                        # 只允许变量节点
                        var_name = self.random_state.choice(variables)
                        return Node(name=var_name, value=var_name)
                else:
                    # 生成函数节点（使用加权选择，优先选择 sigmoid、rank、scale、ts_zscore）
                    func_name, func, arity = self._select_function_weighted()

                    def force_const_or_child(idx, const_builder):
                        if self._allow_const_at(func_name, idx):
                            return const_builder()
                        return grow(d - 1, func_name, idx)

                    # 特殊处理时间序列函数的窗口参数
                    if arity == 2 and func_name in [
                        "REF",
                        "DIFF",
                        "HHV",
                        "HV",
                        "LLV",
                        "LV",
                        "HHVBARS",
                        "LLVBARS",
                        "MA",
                        "EMA",
                        "WMA",
                        "DMA",
                        "AVEDEV",
                        "SLOPE",
                        "FORCAST",
                        "STD",
                        "SUM",
                    ]:
                        if func_name == "DMA":
                            children = [grow(d - 1, func_name, 0)]
                            children.append(
                                force_const_or_child(
                                    1,
                                    lambda: Node(
                                        name="const", value=self.random_state.random()
                                    ),
                                )
                            )
                        else:
                            children = [grow(d - 1, func_name, 0)]
                            children.append(
                                force_const_or_child(
                                    1,
                                    lambda: Node(
                                        name="const",
                                        value=self.random_state.randint(
                                            1, self.ts_window + 1
                                        ),
                                    ),
                                )
                            )
                    elif arity == 3 and func_name == "SMA":
                        # SMA(x, d, m): 3个参数，m通常为1
                        children = [grow(d - 1, func_name, 0)]
                        children.append(
                            force_const_or_child(
                                1,
                                lambda: Node(
                                    name="const",
                                    value=self.random_state.randint(
                                        1, self.ts_window + 1
                                    ),
                                ),
                            )
                        )
                        children.append(
                            force_const_or_child(
                                2, lambda: Node(name="const", value=1.0)
                            )
                        )
                    elif arity == 3 and func_name == "BETWEEN":
                        # BETWEEN(x, a, b): 3个参数，a和b可以是变量或常数
                        children = [
                            grow(d - 1, func_name, 0),
                            grow(d - 1, func_name, 1),
                            grow(d - 1, func_name, 2),
                        ]
                    elif func_name in [
                        "MACD_DIF",
                        "MACD_DEA",
                        "MACD_MACD",
                        "KDJ_K",
                        "KDJ_D",
                        "KDJ_J",
                        "RSI",
                        "SAR",
                        "WR",
                        "BIAS",
                        "BOLL_UPPER",
                        "BOLL_LOWER",
                        "PSY",
                        "CCI",
                        "TR",
                        "ATR",
                        "BBI",
                        "ADX",
                        "TRIX",
                        "VR",
                        "CR",
                        "EMV",
                        "DPO",
                        "BRAR_AR",
                        "BRAR_BR",
                        "DIFMA",
                        "MTM",
                        "MASS",
                        "ROC",
                        "OBV",
                        "MFI",
                        "ASI",
                    ]:
                        # 高阶技术指标函数：所有参数都是周期参数（常数），统一范围为2到120
                        # 三个参数: MACD_DEA, MACD_MACD,KDJ_D, KDJ_J,DIFMA
                        if func_name in [
                            "MACD_DEA",
                            "MACD_MACD",
                            "KDJ_D",
                            "KDJ_J",
                            "DIFMA",
                        ]:
                            children = [
                                force_const_or_child(
                                    0,
                                    lambda: Node(
                                        name="const",
                                        value=self.random_state.randint(2, 120),
                                    ),
                                ),
                                force_const_or_child(
                                    1,
                                    lambda: Node(
                                        name="const",
                                        value=self.random_state.randint(2, 120),
                                    ),
                                ),
                                force_const_or_child(
                                    2,
                                    lambda: Node(
                                        name="const",
                                        value=self.random_state.randint(2, 120),
                                    ),
                                ),
                            ]
                        # 两个参数: MACD_DIF,KDJ_K,ADX,DPO,MASS
                        elif func_name in ["MACD_DIF", "KDJ_K", "ADX", "DPO", "MASS"]:
                            children = [
                                force_const_or_child(
                                    0,
                                    lambda: Node(
                                        name="const",
                                        value=self.random_state.randint(2, 120),
                                    ),
                                ),
                                force_const_or_child(
                                    1,
                                    lambda: Node(
                                        name="const",
                                        value=self.random_state.randint(2, 120),
                                    ),
                                ),
                            ]
                        # 一个参数: RSI, WR, BIAS, PSY, CCI, ATR, CR, MTM, ROC, VR, TRIX, BRAR_AR, BRAR_BR, ASI, MFI, EMV
                        elif func_name in [
                            "RSI",
                            "WR",
                            "BIAS",
                            "PSY",
                            "CCI",
                            "ATR",
                            "CR",
                            "MTM",
                            "ROC",
                            "VR",
                            "TRIX",
                            "BRAR_AR",
                            "BRAR_BR",
                            "ASI",
                            "MFI",
                            "EMV",
                        ]:
                            children = [
                                force_const_or_child(
                                    0,
                                    lambda: Node(
                                        name="const",
                                        value=self.random_state.randint(2, 120),
                                    ),
                                )
                            ]
                        # SAR: n, s, m (s是浮点数，n和m是整数)
                        elif func_name == "SAR":
                            children = [
                                force_const_or_child(
                                    0,
                                    lambda: Node(
                                        name="const",
                                        value=self.random_state.randint(2, 120),
                                    ),
                                ),
                                force_const_or_child(
                                    1,
                                    lambda: Node(
                                        name="const",
                                        value=self.random_state.uniform(2.0, 120.0),
                                    ),
                                ),
                                force_const_or_child(
                                    2,
                                    lambda: Node(
                                        name="const",
                                        value=self.random_state.randint(2, 120),
                                    ),
                                ),
                            ]
                        # BOLL_UPPER, BOLL_LOWER: n, p (p是浮点数)
                        elif func_name in ["BOLL_UPPER", "BOLL_LOWER"]:
                            children = [
                                force_const_or_child(
                                    0,
                                    lambda: Node(
                                        name="const",
                                        value=self.random_state.randint(2, 120),
                                    ),
                                ),
                                force_const_or_child(
                                    1,
                                    lambda: Node(
                                        name="const",
                                        value=self.random_state.uniform(0.01, 5.0),
                                    ),
                                ),
                            ]
                        # TR, OBV: 无参数
                        elif func_name in ["TR", "OBV"]:
                            children = []
                        # BBI: m1, m2, m3, m4
                        elif func_name == "BBI":
                            children = [
                                force_const_or_child(
                                    0,
                                    lambda: Node(
                                        name="const",
                                        value=self.random_state.randint(2, 120),
                                    ),
                                ),
                                force_const_or_child(
                                    1,
                                    lambda: Node(
                                        name="const",
                                        value=self.random_state.randint(2, 120),
                                    ),
                                ),
                                force_const_or_child(
                                    2,
                                    lambda: Node(
                                        name="const",
                                        value=self.random_state.randint(2, 120),
                                    ),
                                ),
                                force_const_or_child(
                                    3,
                                    lambda: Node(
                                        name="const",
                                        value=self.random_state.randint(2, 120),
                                    ),
                                ),
                            ]
                        else:
                            # 默认：生成arity个随机周期参数
                            children = [
                                force_const_or_child(
                                    i,
                                    lambda: Node(
                                        name="const",
                                        value=self.random_state.randint(2, 120),
                                    ),
                                )
                                for i in range(arity)
                            ]
                    else:
                        # 普通函数：生成对应数量的子节点
                        children = [grow(d - 1, func_name, i) for i in range(arity)]

                    return Node(name=func_name, children=children)

            return grow(depth)

        elif method == "full":

            def full(d, parent_func_name=None, arg_index=None):
                # 完全法：只在最大深度生成叶子节点
                if d == 0:
                    if not self._allow_const_at(parent_func_name, arg_index):
                        var_name = self.random_state.choice(variables)
                        return Node(name=var_name, value=var_name)
                    if self.const_range is not None and self.random_state.rand() < 0.5:
                        var_name = self.random_state.choice(variables)
                        return Node(name=var_name, value=var_name)
                    elif self.const_range is not None:
                        const_val = self.random_state.uniform(*self.const_range)
                        return Node(name="const", value=const_val)
                    else:
                        var_name = self.random_state.choice(variables)
                        return Node(name=var_name, value=var_name)
                else:
                    # 与grow方法类似的函数节点生成逻辑（使用加权选择）
                    func_name, func, arity = self._select_function_weighted()

                    def force_const_or_child(idx, const_builder):
                        if self._allow_const_at(func_name, idx):
                            return const_builder()
                        return full(d - 1, func_name, idx)

                    # 特殊处理时间序列函数（代码逻辑同grow方法）
                    if arity == 2 and func_name in [
                        "REF",
                        "DIFF",
                        "HHV",
                        "HV",
                        "LLV",
                        "LV",
                        "HHVBARS",
                        "LLVBARS",
                        "MA",
                        "EMA",
                        "WMA",
                        "DMA",
                        "AVEDEV",
                        "SLOPE",
                        "FORCAST",
                        "STD",
                        "SUM",
                    ]:
                        if func_name == "DMA":
                            children = [full(d - 1, func_name, 0)]
                            children.append(
                                force_const_or_child(
                                    1,
                                    lambda: Node(
                                        name="const", value=self.random_state.random()
                                    ),
                                )
                            )
                        else:
                            children = [full(d - 1, func_name, 0)]
                            children.append(
                                force_const_or_child(
                                    1,
                                    lambda: Node(
                                        name="const",
                                        value=self.random_state.randint(
                                            1, self.ts_window + 1
                                        ),
                                    ),
                                )
                            )
                    elif arity == 3 and func_name == "SMA":
                        # SMA(x, d, m): 3个参数，m通常为1
                        children = [full(d - 1, func_name, 0)]
                        children.append(
                            force_const_or_child(
                                1,
                                lambda: Node(
                                    name="const",
                                    value=self.random_state.randint(
                                        1, self.ts_window + 1
                                    ),
                                ),
                            )
                        )
                        children.append(
                            force_const_or_child(
                                2, lambda: Node(name="const", value=1.0)
                            )
                        )
                    elif arity == 3 and func_name == "BETWEEN":
                        # BETWEEN(x, a, b): 3个参数，a和b可以是变量或常数
                        children = [
                            full(d - 1, func_name, 0),
                            full(d - 1, func_name, 1),
                            full(d - 1, func_name, 2),
                        ]
                    elif func_name in [
                        "MACD_DIF",
                        "MACD_DEA",
                        "MACD_MACD",
                        "KDJ_K",
                        "KDJ_D",
                        "KDJ_J",
                        "RSI",
                        "SAR",
                        "WR",
                        "BIAS",
                        "BOLL_UPPER",
                        "BOLL_LOWER",
                        "PSY",
                        "CCI",
                        "TR",
                        "ATR",
                        "BBI",
                        "ADX",
                        "TRIX",
                        "VR",
                        "CR",
                        "EMV",
                        "DPO",
                        "BRAR_AR",
                        "BRAR_BR",
                        "DIFMA",
                        "MTM",
                        "MASS",
                        "ROC",
                        "OBV",
                        "MFI",
                        "ASI",
                    ]:
                        # 高阶技术指标函数：所有参数都是周期参数（常数），统一范围为2到120
                        # 三个参数: MACD_DEA, MACD_MACD,KDJ_D, KDJ_J,DIFMA
                        if func_name in [
                            "MACD_DEA",
                            "MACD_MACD",
                            "KDJ_D",
                            "KDJ_J",
                            "DIFMA",
                        ]:
                            children = [
                                force_const_or_child(
                                    0,
                                    lambda: Node(
                                        name="const",
                                        value=self.random_state.randint(2, 120),
                                    ),
                                ),
                                force_const_or_child(
                                    1,
                                    lambda: Node(
                                        name="const",
                                        value=self.random_state.randint(2, 120),
                                    ),
                                ),
                                force_const_or_child(
                                    2,
                                    lambda: Node(
                                        name="const",
                                        value=self.random_state.randint(2, 120),
                                    ),
                                ),
                            ]
                        # 两个参数: MACD_DIF,KDJ_K,ADX,DPO,MASS
                        elif func_name in ["MACD_DIF", "KDJ_K", "ADX", "DPO", "MASS"]:
                            children = [
                                force_const_or_child(
                                    0,
                                    lambda: Node(
                                        name="const",
                                        value=self.random_state.randint(2, 120),
                                    ),
                                ),
                                force_const_or_child(
                                    1,
                                    lambda: Node(
                                        name="const",
                                        value=self.random_state.randint(2, 120),
                                    ),
                                ),
                            ]
                        # 一个参数: RSI, WR, BIAS, PSY, CCI, ATR, CR, MTM, ROC, VR, TRIX, BRAR_AR, BRAR_BR, ASI, MFI, EMV
                        elif func_name in [
                            "RSI",
                            "WR",
                            "BIAS",
                            "PSY",
                            "CCI",
                            "ATR",
                            "CR",
                            "MTM",
                            "ROC",
                            "VR",
                            "TRIX",
                            "BRAR_AR",
                            "BRAR_BR",
                            "ASI",
                            "MFI",
                            "EMV",
                        ]:
                            children = [
                                force_const_or_child(
                                    0,
                                    lambda: Node(
                                        name="const",
                                        value=self.random_state.randint(2, 120),
                                    ),
                                )
                            ]
                        # SAR: n, s, m (s是浮点数，n和m是整数)
                        elif func_name == "SAR":
                            children = [
                                force_const_or_child(
                                    0,
                                    lambda: Node(
                                        name="const",
                                        value=self.random_state.randint(2, 120),
                                    ),
                                ),
                                force_const_or_child(
                                    1,
                                    lambda: Node(
                                        name="const",
                                        value=self.random_state.uniform(2.0, 120.0),
                                    ),
                                ),
                                force_const_or_child(
                                    2,
                                    lambda: Node(
                                        name="const",
                                        value=self.random_state.randint(2, 120),
                                    ),
                                ),
                            ]
                        # BOLL_UPPER, BOLL_LOWER: n, p (p是浮点数)
                        elif func_name in ["BOLL_UPPER", "BOLL_LOWER"]:
                            children = [
                                force_const_or_child(
                                    0,
                                    lambda: Node(
                                        name="const",
                                        value=self.random_state.randint(2, 120),
                                    ),
                                ),
                                force_const_or_child(
                                    1,
                                    lambda: Node(
                                        name="const",
                                        value=self.random_state.uniform(0.01, 5.0),
                                    ),
                                ),
                            ]
                        # TR, OBV: 无参数
                        elif func_name in ["TR", "OBV"]:
                            children = []
                        # BBI: m1, m2, m3, m4
                        elif func_name == "BBI":
                            children = [
                                force_const_or_child(
                                    0,
                                    lambda: Node(
                                        name="const",
                                        value=self.random_state.randint(2, 120),
                                    ),
                                ),
                                force_const_or_child(
                                    1,
                                    lambda: Node(
                                        name="const",
                                        value=self.random_state.randint(2, 120),
                                    ),
                                ),
                                force_const_or_child(
                                    2,
                                    lambda: Node(
                                        name="const",
                                        value=self.random_state.randint(2, 120),
                                    ),
                                ),
                                force_const_or_child(
                                    3,
                                    lambda: Node(
                                        name="const",
                                        value=self.random_state.randint(2, 120),
                                    ),
                                ),
                            ]
                        else:
                            # 默认：生成arity个随机周期参数
                            children = [
                                force_const_or_child(
                                    i,
                                    lambda: Node(
                                        name="const",
                                        value=self.random_state.randint(2, 120),
                                    ),
                                )
                                for i in range(arity)
                            ]
                    else:
                        children = [full(d - 1, func_name, i) for i in range(arity)]

                    return Node(name=func_name, children=children)

            return full(depth)
        else:  # half_and_half
            # 随机选择grow或full方法
            if self.random_state.rand() < 0.5:
                return self._random_program(method="grow")
            else:
                return self._random_program(method="full")

    def _evaluate_population(
        self, population, fitness_func, args, kwargs, return_details=False
    ):
        """
        评估种群中所有个体的适应度

        参数:
            population: 种群列表
            fitness_func: 适应度函数
            args: 适应度函数参数
            kwargs: 适应度函数关键字参数

        返回:
            适应度值列表，或包含详细信息的列表

        注意:
            此处为串行评估，可扩展为并行评估以提高性能
        """
        try:
            if return_details:
                return [
                    fitness_func(prog, *args, return_details=True, **kwargs)
                    for prog in population
                ]
            return [fitness_func(prog, *args, **kwargs) for prog in population]
        except Exception as e:
            log_print(f"GeneticProgrammer._evaluate_population函数异常: {e}")
            raise

    def _tournament(self, population, fitnesses):
        """
        锦标赛选择

        从种群中随机选择若干个体进行比较，返回适应度最高的个体

        参数:
            population: 种群列表
            fitnesses: 适应度列表

        返回:
            选中的个体

        优势:
            - 选择压力可调（通过锦标赛规模控制）
            - 不需要适应度排序
            - 支持负适应度
        """
        try:
            idxs = self.random_state.choice(
                len(population), self.tournament_size, replace=False
            )
            best = idxs[np.argmax([fitnesses[i] for i in idxs])]
            return population[best]
        except Exception as e:
            log_print(
                f"GeneticProgrammer._tournament函数异常: {e}, population_size={len(population)}, tournament_size={self.tournament_size}"
            )
            raise

    def _mutate_or_crossover(self, parent, population):
        """
        应用遗传操作：根据概率选择交叉或变异

        参数:
            parent: 父代个体
            population: 当前种群

        返回:
            经过遗传操作的子代个体

        操作类型:
            1. 交叉：与随机个体交换子树
            2. 子树变异：替换随机子树
            3. 提升变异：将子节点提升为根
            4. 点变异：替换单个节点
            5. 复制：直接复制（保持多样性）
        """
        try:
            op = self.random_state.rand()

            if op < self.p_crossover:
                # 交叉操作
                mate_idx = self.random_state.randint(0, len(population))
                mate = population[mate_idx]
                return self._crossover(parent, mate)
            elif op < self.p_crossover + self.p_subtree_mutation:
                # 子树变异
                return self._subtree_mutation(parent)
            elif (
                op < self.p_crossover + self.p_subtree_mutation + self.p_hoist_mutation
            ):
                # 提升变异
                return self._hoist_mutation(parent)
            elif (
                op
                < self.p_crossover
                + self.p_subtree_mutation
                + self.p_hoist_mutation
                + self.p_point_mutation
            ):
                # 点变异
                return self._point_mutation(parent)
            else:
                # 直接复制
                return self._copy_tree(parent)
        except Exception as e:
            log_print(f"GeneticProgrammer._mutate_or_crossover函数异常: {e}")
            raise

    def _copy_tree(self, node: Node) -> Node:
        """
        深拷贝表达式树

        参数:
            node: 要复制的节点

        返回:
            复制后的新节点
        """
        try:
            return Node(
                name=node.name,
                value=node.value,
                children=[self._copy_tree(c) for c in node.children],
            )
        except Exception as e:
            log_print(f"GeneticProgrammer._copy_tree函数异常: {e}")
            raise

    def _all_nodes(self, node: Node) -> List[Node]:
        """
        获取表达式树中所有节点的列表

        参数:
            node: 根节点

        返回:
            包含所有节点的列表（前序遍历）
        """
        try:
            nodes = [node]
            for child in node.children:
                nodes.extend(self._all_nodes(child))
            return nodes
        except Exception as e:
            log_print(f"GeneticProgrammer._all_nodes函数异常: {e}")
            raise

    def _is_high_level_func(self, func_name: str) -> bool:
        """判断是否为高阶技术指标函数"""
        return func_name in {
            "MACD_DIF",
            "MACD_DEA",
            "MACD_MACD",
            "KDJ_K",
            "KDJ_D",
            "KDJ_J",
            "RSI",
            "SAR",
            "WR",
            "BIAS",
            "BOLL_UPPER",
            "BOLL_LOWER",
            "PSY",
            "CCI",
            "TR",
            "ATR",
            "BBI",
            "ADX",
            "TRIX",
            "VR",
            "CR",
            "EMV",
            "DPO",
            "BRAR_AR",
            "BRAR_BR",
            "DIFMA",
            "MTM",
            "MASS",
            "ROC",
            "OBV",
            "MFI",
            "ASI",
        }

    def _allow_const_at(
        self, parent_func: Optional[str], arg_index: Optional[int]
    ) -> bool:
        """
        检查指定函数和参数位置是否允许常数

        参数:
            parent_func: 父函数名，None 表示根节点或无父节点
            arg_index: 参数索引，None 表示根节点或无父节点

        返回:
            True: 允许常数（默认或显式配置为True）
            False: 不允许常数（显式配置为False）
        """
        # 如果未启用配置或没有父函数信息，默认允许常数
        if self.allow_const_in_function is None or parent_func is None:
            return True

        # 查找该函数的配置
        if parent_func not in self.allow_const_in_function:
            return True  # 未配置，默认允许

        config = self.allow_const_in_function[parent_func]

        # 如果配置是单个 bool，所有参数使用该值
        if isinstance(config, bool):
            return config

        # 如果配置是列表，根据参数索引查找
        if isinstance(config, list):
            if arg_index is None:
                return True  # 无参数索引信息，默认允许
            if arg_index < len(config):
                return config[arg_index]
            else:
                return True  # 越界，默认允许

        # 其他情况默认允许
        return True

    def _gen_high_level_children(
        self, func_name: str, existing_children: List[Node]
    ) -> List[Node]:
        """生成高阶技术指标的常数参数节点（优先保留已有常数）"""

        def keep_or_new(idx: int, value_type: str):
            if existing_children and idx < len(existing_children):
                c = existing_children[idx]
                if c is not None and c.name == "const" and c.value is not None:
                    return Node(name="const", value=c.value)
            if value_type == "float_boll":
                return Node(name="const", value=self.random_state.uniform(0.01, 5.0))
            if value_type == "float_sar":
                return Node(name="const", value=self.random_state.uniform(2.0, 120.0))
            return Node(name="const", value=self.random_state.randint(2, 120))

        # 无参数: TR, OBV
        if func_name in ["TR", "OBV"]:
            return []
        # 两个参数: MACD_DIF,KDJ_K,ADX,DPO,MASS
        if func_name in ["MACD_DIF", "KDJ_K", "ADX", "DPO", "MASS"]:
            return [keep_or_new(0, "int"), keep_or_new(1, "int")]
        # 三个参数: MACD_DEA, MACD_MACD,KDJ_D, KDJ_J,DIFMA
        if func_name in ["MACD_DEA", "MACD_MACD", "KDJ_D", "KDJ_J", "DIFMA"]:
            return [keep_or_new(0, "int"), keep_or_new(1, "int"), keep_or_new(2, "int")]
        # 一个参数: RSI, WR, BIAS, PSY, CCI, ATR, CR, MTM, ROC, VR, TRIX, BRAR_AR, BRAR_BR, ASI, MFI, EMV
        if func_name in [
            "RSI",
            "WR",
            "BIAS",
            "PSY",
            "CCI",
            "ATR",
            "CR",
            "MTM",
            "ROC",
            "VR",
            "TRIX",
            "BRAR_AR",
            "BRAR_BR",
            "ASI",
            "MFI",
            "EMV",
        ]:
            return [keep_or_new(0, "int")]
        if func_name == "SAR":
            return [
                keep_or_new(0, "int"),
                keep_or_new(1, "float_sar"),
                keep_or_new(2, "int"),
            ]
        if func_name in ["BOLL_UPPER", "BOLL_LOWER"]:
            return [keep_or_new(0, "int"), keep_or_new(1, "float_boll")]
        if func_name == "BBI":
            return [
                keep_or_new(0, "int"),
                keep_or_new(1, "int"),
                keep_or_new(2, "int"),
                keep_or_new(3, "int"),
            ]
        return [keep_or_new(0, "int")]

    def _normalize_high_level_params(self, node: Node):
        """将高阶指标的参数子节点强制归一为常数节点"""
        try:
            if node is None:
                return
            if node.value is None and self._is_high_level_func(node.name):
                new_children = self._gen_high_level_children(node.name, node.children)
                for i in range(len(new_children)):
                    if not self._allow_const_at(node.name, i):
                        existing = (
                            node.children[i]
                            if (node.children and i < len(node.children))
                            else None
                        )
                        if (
                            existing is not None
                            and existing.name != "const"
                            and existing.value is not None
                        ):
                            new_children[i] = Node(
                                name=existing.name, value=existing.value
                            )
                        else:
                            var_name = self.random_state.choice(self.variable_names)
                            new_children[i] = Node(name=var_name, value=var_name)
                node.children = new_children
            # 限制SMA最后一个参数m，避免交叉/变异生成过大值
            if node.value is None and node.name == "SMA" and len(node.children) >= 3:
                if self._allow_const_at(node.name, 2):
                    m_child = node.children[2]
                    m_raw = (
                        m_child.value
                        if (m_child is not None and m_child.name == "const")
                        else 1
                    )
                    m_val = _normalize_window_length(
                        m_raw, x=None, min_len=1, default=1, max_len=5
                    )
                    node.children[2] = Node(name="const", value=float(m_val))
                else:
                    m_child = node.children[2]
                    if m_child is not None and m_child.name == "const":
                        var_name = self.random_state.choice(self.variable_names)
                        node.children[2] = Node(name=var_name, value=var_name)
            for child in node.children:
                self._normalize_high_level_params(child)
        except Exception as e:
            log_print(f"GeneticProgrammer._normalize_high_level_params函数异常: {e}")
            raise

    def _enforce_const_constraints(self, node: Node):
        """强制执行函数参数位置的常数约束（交叉/变异后校正）"""
        try:
            if node is None:
                return
            if node.value is None and node.children:
                for idx, child in enumerate(node.children):
                    if child is not None and child.name == "const":
                        if not self._allow_const_at(node.name, idx):
                            var_name = self.random_state.choice(self.variable_names)
                            node.children[idx] = Node(name=var_name, value=var_name)
                            continue
                    self._enforce_const_constraints(child)
        except Exception as e:
            log_print(f"GeneticProgrammer._enforce_const_constraints函数异常: {e}")
            raise

    def _crossover(self, parent1, parent2):
        """
        交叉操作：交换两个个体的随机子树

        参数:
            parent1, parent2: 两个父代个体

        返回:
            交叉后的子代个体

        过程:
            1. 复制两个父代
            2. 随机选择交叉点
            3. 交换子树
            4. 对常数节点进行随机变异以保持多样性
        """
        try:
            t1 = self._copy_tree(parent1)
            t2 = self._copy_tree(parent2)

            nodes1 = self._all_nodes(t1)
            nodes2 = self._all_nodes(t2)

            if not nodes1 or not nodes2:
                return t1

            # 随机选择交叉点
            n1_idx = self.random_state.randint(0, len(nodes1))
            n1 = nodes1[n1_idx]
            if n1 is t1:
                nodes2_non_const = [n for n in nodes2 if n.name != "const"]
                if not nodes2_non_const:
                    return t1
                n2 = nodes2_non_const[
                    self.random_state.randint(0, len(nodes2_non_const))
                ]
            else:
                n2_idx = self.random_state.randint(0, len(nodes2))
                n2 = nodes2[n2_idx]

            # 交换子树
            n1.name, n1.value, n1.children = (
                n2.name,
                n2.value,
                [self._copy_tree(c) for c in n2.children],
            )

            # 对结果树中的常数节点进行随机变异（30%概率重新生成新常数）
            # 这样可以避免所有常数都相同的问题
            def mutate_constants(node):
                if node.value is not None and node.name == "const":
                    # 30%概率重新生成新的随机常数
                    if self.random_state.rand() < 0.3 and self.const_range is not None:
                        node.value = self.random_state.uniform(*self.const_range)
                for child in node.children:
                    mutate_constants(child)

            mutate_constants(t1)
            self._normalize_high_level_params(t1)
            self._enforce_const_constraints(t1)
            if t1.name == "const" and t1.value is not None:
                var_name = self.random_state.choice(self.variable_names)
                t1.name, t1.value, t1.children = var_name, var_name, []
            return t1
        except Exception as e:
            log_print(f"GeneticProgrammer._crossover函数异常: {e}")
            raise

    def _subtree_mutation(self, node: Node) -> Node:
        """
        子树变异：用随机生成的子树替换原有子树

        参数:
            node: 要变异的个体

        返回:
            变异后的个体
        """
        try:
            t = self._copy_tree(node)
            nodes = self._all_nodes(t)

            # 随机选择变异点
            n_idx = self.random_state.randint(0, len(nodes))
            n = nodes[n_idx]

            # 生成新子树替换
            depth = self.random_state.randint(*self.init_depth)
            new_subtree = self._random_program_with_depth(depth)

            if new_subtree is not None:
                # 根节点禁止常数，避免出现纯常数表达式
                if (
                    n is t
                    and new_subtree.name == "const"
                    and new_subtree.value is not None
                ):
                    var_name = self.random_state.choice(self.variable_names)
                    new_subtree = Node(name=var_name, value=var_name)
                n.name, n.value, n.children = (
                    new_subtree.name,
                    new_subtree.value,
                    new_subtree.children,
                )
            self._normalize_high_level_params(t)
            self._enforce_const_constraints(t)
            return t
        except Exception as e:
            log_print(f"GeneticProgrammer._subtree_mutation函数异常: {e}")
            raise

    def _random_program_with_depth(self, depth: int) -> Node:
        """
        生成指定深度的随机程序

        参数:
            depth: 目标深度

        返回:
            生成的表达式树根节点
        """
        try:
            variables = self.variable_names

            def build(current_depth, parent_func_name=None, arg_index=None):
                if current_depth >= depth:
                    # 到达目标深度，生成叶子节点
                    # 根节点禁止常数，避免出现纯常数表达式
                    if parent_func_name is None:
                        var_name = self.random_state.choice(variables)
                        return Node(name=var_name, value=var_name)
                    if not self._allow_const_at(parent_func_name, arg_index):
                        var_name = self.random_state.choice(variables)
                        return Node(name=var_name, value=var_name)
                    if self.random_state.rand() < 0.6:
                        var_name = self.random_state.choice(variables)
                        return Node(name=var_name, value=var_name)
                    else:
                        if self.const_range is not None:
                            const_val = self.random_state.uniform(*self.const_range)
                            return Node(name="const", value=const_val)
                        else:
                            var_name = self.random_state.choice(variables)
                            return Node(name=var_name, value=var_name)

                # 生成函数节点（使用加权选择，优先选择 sigmoid、rank、scale、ts_zscore）
                func_name, _, arity = self._select_function_weighted()

                # 特殊处理时间序列函数（与_random_program方法逻辑相同）
                def force_const_or_child(idx, const_builder):
                    if self._allow_const_at(func_name, idx):
                        return const_builder()
                    return build(current_depth + 1, func_name, idx)

                if arity == 2 and func_name in [
                    "REF",
                    "DIFF",
                    "HHV",
                    "HV",
                    "LLV",
                    "LV",
                    "HHVBARS",
                    "LLVBARS",
                    "MA",
                    "EMA",
                    "WMA",
                    "DMA",
                    "AVEDEV",
                    "SLOPE",
                    "FORCAST",
                    "STD",
                    "SUM",
                    "RANK",
                    "SCALE",
                    "TS_RANK",
                    "TS_ZSCORE",
                ]:
                    if func_name == "DMA":
                        children = [build(current_depth + 1, func_name, 0)]
                        children.append(
                            force_const_or_child(
                                1,
                                lambda: Node(
                                    name="const", value=self.random_state.random()
                                ),
                            )
                        )
                    else:
                        children = [build(current_depth + 1, func_name, 0)]
                        children.append(
                            force_const_or_child(
                                1,
                                lambda: Node(
                                    name="const",
                                    value=self.random_state.randint(
                                        1, self.ts_window + 1
                                    ),
                                ),
                            )
                        )
                    return Node(name=func_name, children=children)
                elif arity == 3 and func_name in [
                    "CORR",
                    "COVA",
                    "RANK_SUB",
                    "RANK_DIV",
                ]:
                    children = [
                        build(current_depth + 1, func_name, 0),
                        build(current_depth + 1, func_name, 1),
                    ]
                    children.append(
                        force_const_or_child(
                            2,
                            lambda: Node(
                                name="const",
                                value=self.random_state.randint(1, self.ts_window + 1),
                            ),
                        )
                    )
                    return Node(name=func_name, children=children)
                elif arity == 3 and func_name == "SMA":
                    # SMA(x, d, m): 3个参数，m通常为1
                    children = [build(current_depth + 1, func_name, 0)]
                    children.append(
                        force_const_or_child(
                            1,
                            lambda: Node(
                                name="const",
                                value=self.random_state.randint(1, self.ts_window + 1),
                            ),
                        )
                    )
                    children.append(
                        force_const_or_child(2, lambda: Node(name="const", value=1.0))
                    )
                    return Node(name=func_name, children=children)
                else:
                    children = [
                        build(current_depth + 1, func_name, i) for i in range(arity)
                    ]
                    return Node(name=func_name, children=children)

            result = build(0)

            # 确保返回有效节点
            if result is None:
                var_name = self.random_state.choice(variables)
                return Node(name=var_name, value=var_name)
            return result
        except Exception as e:
            log_print(
                f"GeneticProgrammer._random_program_with_depth函数异常: {e}, depth={depth}"
            )
            raise

    def _hoist_mutation(self, node: Node) -> Node:
        """
        提升变异：随机选择一个子节点提升为根节点

        参数:
            node: 要变异的个体

        返回:
            变异后的个体

        作用:
            减少树的复杂度，实现自动剪枝
        """
        try:
            t = self._copy_tree(node)
            nodes = self._all_nodes(t)

            if len(nodes) <= 1:
                return t

            # 选择非根节点进行提升
            sub_nodes = nodes[1:]
            # 跳过常数子节点，避免提升为常数根
            candidates = [
                n for n in sub_nodes if not (n.name == "const" and n.value is not None)
            ]
            if not candidates:
                return t
            n_idx = self.random_state.randint(0, len(candidates))
            n = candidates[n_idx]

            result = self._copy_tree(n)
            self._normalize_high_level_params(result)
            self._enforce_const_constraints(result)
            return result
        except Exception as e:
            log_print(f"GeneticProgrammer._hoist_mutation函数异常: {e}")
            raise

    def _point_mutation(self, node: Node) -> Node:
        """
        点变异：随机替换树中的单个节点

        参数:
            node: 要变异的个体

        返回:
            变异后的个体

        变异类型:
            - 变量/常数节点：替换为其他变量或常数
            - 函数节点：替换为同参数数量的其他函数
        """
        try:
            t = self._copy_tree(node)

            def mutate(n: Node, parent_name=None, arg_index=None):
                if self.random_state.rand() < self.p_point_replace:
                    if n.value is not None:
                        # 叶子节点变异
                        if n.name == "const":
                            # 如果当前是常数节点，优先重新生成新的随机常数（80%概率）
                            # 这样可以确保常数多样性
                            if parent_name is None:
                                # 根节点禁止常数
                                var_name = self.random_state.choice(self.variable_names)
                                n.name = var_name
                                n.value = var_name
                            elif (
                                self.random_state.rand() < 0.5
                                and self.const_range is not None
                            ):
                                n.value = self.random_state.uniform(*self.const_range)
                            else:
                                # 20%概率变为变量
                                var_name = self.random_state.choice(self.variable_names)
                                n.name = var_name
                                n.value = var_name
                        else:
                            # 变量节点变异：50%概率变为其他变量，50%概率变为常数
                            if self.random_state.rand() < 0.5:
                                # 变为其他变量
                                var_name = self.random_state.choice(self.variable_names)
                                n.name = var_name
                                n.value = var_name
                            else:
                                # 变为新常数
                                if parent_name is None:
                                    # 根节点禁止常数
                                    var_name = self.random_state.choice(
                                        self.variable_names
                                    )
                                    n.name = var_name
                                    n.value = var_name
                                elif (
                                    self.const_range is not None
                                    and self._allow_const_at(parent_name, arg_index)
                                ):
                                    n.name = "const"
                                    n.value = self.random_state.uniform(
                                        *self.const_range
                                    )
                                else:
                                    # 如果禁止常数，变为变量
                                    var_name = self.random_state.choice(
                                        self.variable_names
                                    )
                                    n.name = var_name
                                    n.value = var_name
                    else:
                        # 函数节点变异：保持参数个数不变（使用加权选择）
                        current_arity = len(n.children)
                        # 检查是否有匹配的函数
                        matching_funcs = [
                            (name, func, arity)
                            for name, (func, arity) in self.function_set.all().items()
                            if arity == current_arity
                        ]
                        if matching_funcs:
                            # 使用加权选择，优先选择 sigmoid、rank、scale、ts_zscore
                            func_name, _, selected_arity = (
                                self._select_function_weighted(
                                    target_arity=current_arity
                                )
                            )
                            # 确保选择的函数arity匹配
                            if selected_arity == current_arity:
                                n.name = func_name
                        # 如果没有匹配的函数，保持原函数不变

                # 递归变异子节点
                for idx, c in enumerate(n.children):
                    mutate(c, parent_name=n.name, arg_index=idx)

            mutate(t)
            self._normalize_high_level_params(t)
            self._enforce_const_constraints(t)
            return t
        except Exception as e:
            log_print(f"GeneticProgrammer._point_mutation函数异常: {e}")
            raise


class My:
    @staticmethod
    def _ensure_array(data):
        """确保输入数据为numpy数组"""
        return _ensure_xp_array(data)

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
        return _to_np_output(data)

    # ------------------ 0级:核心工具函数 ------------------
    @staticmethod
    def ADD(A, B):
        """返回A+B"""
        A = My._ensure_array(A)
        B = My._ensure_array(B)
        return My._ensure_np_output(A + B)

    @staticmethod
    def SUB(A, B):
        """返回A-B"""
        A = My._ensure_array(A)
        B = My._ensure_array(B)
        return My._ensure_np_output(A - B)

    @staticmethod
    def MUL(A, B):
        """返回A*B"""
        A = My._ensure_array(A)
        B = My._ensure_array(B)
        return My._ensure_np_output(A * B)

    @staticmethod
    def DIV(A, B):
        """返回A/B，带安全除零保护"""
        A = My._ensure_array(A)
        B = My._ensure_array(B)
        B_safe = xp.where(xp.abs(B) < 1e-10, 1e-10, B)
        return My._ensure_np_output(A / B_safe)

    @staticmethod
    def ABS(S):
        """返回N的绝对值"""
        S = My._ensure_array(S)
        return My._ensure_np_output(xp.abs(S))

    @staticmethod
    def LN(S):
        """求底是e的自然对数"""
        S = My._ensure_array(S)
        return My._ensure_np_output(xp.log(xp.abs(S) + 1e-8))

    @staticmethod
    def INV(S):
        """求S的倒数"""
        S = My._ensure_array(S)
        S_safe = xp.where(xp.abs(S) < 1e-10, 1e-10, S)
        return My._ensure_np_output(1 / S_safe)

    @staticmethod
    def POW(S, N):
        """求S的N次方"""
        S = My._ensure_array(S)
        return My._ensure_np_output(xp.power(S, N))

    @staticmethod
    def SQRT(S):
        """求S的平方根"""
        S = My._ensure_array(S)
        return My._ensure_np_output(xp.sqrt(S))

    @staticmethod
    def SIN(S):
        """求S的正弦值(弧度)"""
        S = My._ensure_array(S)
        return My._ensure_np_output(xp.sin(S))

    @staticmethod
    def COS(S):
        """求S的余弦值(弧度)"""
        S = My._ensure_array(S)
        return My._ensure_np_output(xp.cos(S))

    @staticmethod
    def TAN(S):
        """求S的正切值(弧度)"""
        S = My._ensure_array(S)
        return My._ensure_np_output(xp.tan(S))

    @staticmethod
    def MAX(S1, S2):
        """序列max"""
        S1 = My._ensure_array(S1)
        S2 = My._ensure_array(S2)
        return My._ensure_np_output(xp.maximum(S1, S2))

    @staticmethod
    def MIN(S1, S2):
        """序列min"""
        S1 = My._ensure_array(S1)
        S2 = My._ensure_array(S2)
        return My._ensure_np_output(xp.minimum(S1, S2))

    @staticmethod
    def IF(S, A, B):
        """序列布尔判断 return=A if S==True else B"""
        S = My._ensure_array(S)
        A = My._ensure_array(A)
        B = My._ensure_array(B)
        return My._ensure_np_output(xp.where(S, A, B))

    @staticmethod
    def AND(S1, S2):
        """逻辑与运算"""
        S1 = My._ensure_array(S1)
        S2 = My._ensure_array(S2)
        return My._ensure_np_output(xp.logical_and(S1, S2))

    @staticmethod
    def OR(S1, S2):
        """逻辑或运算"""
        S1 = My._ensure_array(S1)
        S2 = My._ensure_array(S2)
        return My._ensure_np_output(xp.logical_or(S1, S2))

    @staticmethod
    def NOT(S):
        """逻辑非运算"""
        S = My._ensure_array(S)
        return My._ensure_np_output(xp.logical_not(S))

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
        if xp.isscalar(x):
            return x
        x = xp.asarray(x)
        if x.size == 0:
            return x
        d = _normalize_window_length(d, x=x, min_len=4, default=10, max_len=120)
        if x.ndim == 1:
            res = xp.full_like(x, xp.nan, dtype=xp.float64)
            for i in range(d - 1, len(x)):
                window = x[i - d + 1 : i + 1]
                ranks = window.argsort().argsort() / (len(window) - 1 + 1e-8)
                res[i] = ranks[-1]
            return _to_np_output(res)
        else:
            res = xp.full_like(x, xp.nan, dtype=xp.float64)
            for j in range(x.shape[1]):
                col = x[:, j]
                for i in range(d - 1, len(col)):
                    window = col[i - d + 1 : i + 1]
                    ranks = window.argsort().argsort() / (len(window) - 1 + 1e-8)
                    res[i, j] = ranks[-1]
            return _to_np_output(res)

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
        x = xp.asarray(x)
        if xp.isscalar(x):
            return x
        d = _normalize_window_length(d, x=x, min_len=4, default=10, max_len=120)
        if x.ndim == 1:
            res = xp.full_like(x, xp.nan, dtype=xp.float64)
            for i in range(d - 1, len(x)):
                window = x[i - d + 1 : i + 1]
                current_value = x[i]
                rank_position = xp.sum(window < current_value)
                normalized_rank = rank_position / (len(window) + 1e-8)
                res[i] = normalized_rank
            return _to_np_output(res)
        else:
            res = xp.full_like(x, xp.nan, dtype=xp.float64)
            for j in range(x.shape[1]):
                for i in range(d - 1, x.shape[0]):
                    window = x[i - d + 1 : i + 1, j]
                    current_value = x[i, j]
                    rank_position = xp.sum(window < current_value)
                    normalized_rank = rank_position / (len(window) + 1e-8)
                    res[i, j] = normalized_rank
            return _to_np_output(res)

    @staticmethod
    def TS_ZSCORE(x, d):
        """
        滚动窗口Z-score标准化

        计算 (当前值 - 窗口均值) / 窗口标准差
        """
        x = xp.asarray(x)
        if xp.isscalar(x):
            return 0.0

        d = _normalize_window_length(d, x=x, min_len=4, default=10, max_len=120)

        if x.ndim == 1:
            res = xp.full_like(x, xp.nan, dtype=xp.float64)
            for i in range(d - 1, len(x)):
                window = x[i - d + 1 : i + 1]
                mean = xp.mean(window)
                std = xp.std(window, ddof=1)
                res[i] = (x[i] - mean) / (std + 1e-8)
            return _to_np_output(res)
        else:
            res = xp.full_like(x, xp.nan, dtype=xp.float64)
            for j in range(x.shape[1]):
                for i in range(d - 1, x.shape[0]):
                    window = x[i - d + 1 : i + 1, j]
                    mean = xp.mean(window)
                    std = xp.std(window, ddof=1)
                    res[i, j] = (x[i, j] - mean) / (std + 1e-8)
            return _to_np_output(res)

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
        d = _normalize_window_length(
            d, x=x, other=y, min_len=1, default=10, max_len=120
        )
        rx = My.RANK(x, d)
        ry = My.RANK(y, d)
        try:
            return rx - ry
        except:
            return np.zeros_like(rx) if hasattr(rx, "size") and rx.size > 0 else 0

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
        d = _normalize_window_length(
            d, x=x, other=y, min_len=1, default=10, max_len=120
        )
        rx = My.RANK(x, d)
        ry = My.RANK(y, d)
        try:
            ry_safe = xp.where(ry < 1e-10, 1e-10, ry)
            return rx / ry_safe
        except:
            return np.ones_like(rx) if hasattr(rx, "size") and rx.size > 0 else 1

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
        x = xp.asarray(x)
        return 1.0 / (1.0 + xp.exp(-xp.clip(x, -50, 50)))

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
        try:
            x = xp.asarray(x)
            y = xp.asarray(y)
            if x.size == 0 or y.size == 0 or xp.isscalar(x) or xp.isscalar(y):
                return np.nan
            if xp.nanstd(x) < 1e-6 or xp.nanstd(y) < 1e-6:
                return np.full_like(x, 0, dtype=np.float64)
            d = _normalize_window_length(
                d, x=x, other=y, min_len=4, default=2, max_len=120
            )

            if x.ndim == 1:
                res = xp.full_like(x, xp.nan, dtype=xp.float64)
                for i in range(d - 1, len(x)):
                    x_window = x[i - d + 1 : i + 1]
                    y_window = y[i - d + 1 : i + 1]
                    if len(x_window) >= 2:
                        try:
                            corr = np.corrcoef(x_window, y_window)[0, 1]
                            res[i] = corr if not np.isnan(corr) else np.nan
                        except:
                            res[i] = np.nan
                return _to_np_output(res)
            else:
                res = xp.full_like(x, xp.nan, dtype=xp.float64)
                for j in range(x.shape[1]):
                    for i in range(d - 1, x.shape[0]):
                        x_window = x[i - d + 1 : i + 1, j]
                        y_window = y[i - d + 1 : i + 1, j]
                        if len(x_window) >= 2:
                            try:
                                corr = np.corrcoef(x_window, y_window)[0, 1]
                                res[i, j] = corr if not np.isnan(corr) else np.nan
                            except:
                                res[i, j] = np.nan
                return _to_np_output(res)
        except Exception as e:
            log_print(
                str(e)
                + f"代码行数: {e.__traceback__.tb_lineno}"
                + f"x: {x}"
                + f"y: {y}"
                + f"d: {d}"
            )

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
        try:
            x = xp.asarray(x)
            y = xp.asarray(y)
            if x.size == 0 or y.size == 0 or xp.isscalar(x) or xp.isscalar(y):
                return np.nan
            if xp.nanstd(x) < 1e-6 or xp.nanstd(y) < 1e-6:
                return np.full_like(x, 0, dtype=np.float64)
            d = _normalize_window_length(
                d, x=x, other=y, min_len=4, default=10, max_len=120
            )

            if x.ndim == 1:
                res = xp.full_like(x, xp.nan, dtype=xp.float64)
                for i in range(d - 1, len(x)):
                    x_window = x[i - d + 1 : i + 1]
                    y_window = y[i - d + 1 : i + 1]
                    if len(x_window) >= 2:
                        try:
                            cov = np.cov(x_window, y_window)[0, 1]
                            res[i] = cov if not np.isnan(cov) else np.nan
                        except:
                            res[i] = np.nan
                return _to_np_output(res)
            else:
                res = xp.full_like(x, xp.nan, dtype=xp.float64)
                for j in range(x.shape[1]):
                    for i in range(d - 1, x.shape[0]):
                        x_window = x[i - d + 1 : i + 1, j]
                        y_window = y[i - d + 1 : i + 1, j]
                        if len(x_window) >= 2:
                            try:
                                cov = np.cov(x_window, y_window)[0, 1]
                                res[i, j] = cov if not np.isnan(cov) else np.nan
                            except:
                                res[i, j] = np.nan
                return _to_np_output(res)
        except Exception as e:
            log_print(
                str(e)
                + f"代码行数: {e.__traceback__.tb_lineno}"
                + f"x: {x}"
                + f"y: {y}"
                + f"d: {d}"
            )

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
        if xp.isscalar(x):
            return x
        x = xp.asarray(x)
        if x.size == 0:
            return x
        d = _normalize_window_length(d, x=x, min_len=4, default=10, max_len=120)

        if x.ndim == 1:
            res = xp.full_like(x, xp.nan, dtype=xp.float64)
            for i in range(d - 1, len(x)):
                window = x[i - d + 1 : i + 1]
                s = xp.sum(xp.abs(window))
                s = s if s > 1e-10 else 1e-10
                res[i] = x[i] / s
            return _to_np_output(res)
        else:
            res = xp.full_like(x, xp.nan, dtype=xp.float64)
            for j in range(x.shape[1]):
                col = x[:, j]
                for i in range(d - 1, len(col)):
                    window = col[i - d + 1 : i + 1]
                    s = xp.sum(xp.abs(window))
                    s = s if s > 1e-10 else 1e-10
                    res[i, j] = col[i] / s
            return _to_np_output(res)

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
        return xp.sign(x) * (xp.abs(x) ** 2)

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
        S = My._ensure_array(S)

        if isinstance(S, np.ndarray) and S.ndim == 2:
            T, N_cols = S.shape
            result = np.full((T, N_cols), np.nan, dtype=np.float64)
            for col_idx in range(N_cols):
                s_col = S[:, col_idx]
                try:
                    result[:, col_idx] = My.REF(s_col, N)
                except Exception:
                    pass
            return My._ensure_np_output(result)

        S = My._ensure_series(S)
        N = _normalize_window_length(N, x=S.values, min_len=1, default=1, max_len=120)

        if np.isnan(N) or N is None:
            return My._ensure_np_output(np.full(len(S), np.nan))
        elif N == 0:
            return My._ensure_np_output(S.values)
        else:
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
        N = _normalize_window_length(
            N, x=S_series.values, min_len=2, default=2, max_len=120
        )
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
                return My._ensure_np_output(
                    S.rolling(window=int(N), min_periods=1).max().values
                )
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
            S.rolling(N)
            .apply(lambda x: np.argmax(x[::-1]) if len(x) > 0 else 0, raw=True)
            .values
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
            S.rolling(N)
            .apply(lambda x: np.argmin(x[::-1]) if len(x) > 0 else 0, raw=True)
            .values
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
                # 提取M的对应列，并确保是标量
                if isinstance(M, np.ndarray) and M.ndim == 2:
                    M_col = M[:, col_idx]
                    # 如果M_col是数组，使用_normalize_window_length转换为标量
                    M_col = _normalize_window_length(
                        M_col,
                        x=s_col,
                        min_len=1e-6,
                        default=1.0,
                        max_len=120,
                        return_type="float",
                    )
                elif isinstance(M, np.ndarray) and M.ndim == 1:
                    # 如果M是一维数组，使用_normalize_window_length转换为标量
                    M_col = _normalize_window_length(
                        M,
                        x=s_col,
                        min_len=1e-6,
                        default=1.0,
                        max_len=120,
                        return_type="float",
                    )
                else:
                    M_col = M
                try:
                    result[:, col_idx] = My.SMA(s_col, N, M_col)
                except Exception as e:
                    # 如果某一列计算失败，保持NaN
                    log_print(
                        str(e) + f"代码行数: {e.__traceback__.tb_lineno}" + f"M: {M}"
                    )
            return My._ensure_np_output(result)

        # 一维数组的处理逻辑
        S = My._ensure_series(S)
        N = _normalize_window_length(N, x=S.values, min_len=2, default=2, max_len=120)
        M = _normalize_window_length(
            M, x=S.values, min_len=1e-6, default=1e-6, max_len=N, return_type="float"
        )
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
                lambda x: (
                    x[::-1].cumsum().sum() * 2 / N / (N + 1) if len(x) > 0 else np.nan
                ),
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
                try:
                    result[:, col_idx] = My.DMA(s_col, alpha_param)
                except Exception:
                    # 如果某一列计算失败，保持NaN
                    pass
            return My._ensure_np_output(result)

        # 一维数组的处理逻辑
        S = My._ensure_series(S)
        alpha_param = _normalize_window_length(
            alpha_param,
            x=S.values,
            min_len=1e-6,
            default=1e-6,
            max_len=1 - 1e-6,
            return_type="float",
        )
        if isinstance(alpha_param, (int, float)):
            return My._ensure_np_output(
                S.ewm(alpha=alpha_param, adjust=False).mean().values
            )
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
            S.rolling(N)
            .apply(lambda x: (np.abs(x - x.mean())).mean() if len(x) > 0 else np.nan)
            .values
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
                lambda x: (
                    np.polyval(np.polyfit(range(N), x, deg=1), N - 1)
                    if len(x) == N
                    else np.nan
                ),
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
        return My._ensure_np_output(
            np.concatenate(([False], (S1 <= S2)[:-1] & (S1 > S2)[1:]))
        )

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
        RSV = (
            (CLOSE - My.LLV(LOW, N)) / (My.HHV(HIGH, N) - My.LLV(LOW, N) + 1e-10) * 100
        )
        K = My.EMA(RSV, (M1 * 2 - 1))
        return My._ensure_np_output(K)

    @staticmethod
    def KDJ_D(CLOSE, HIGH, LOW, N=9, M1=3, M2=3):
        """KDJ指标"""
        RSV = (
            (CLOSE - My.LLV(LOW, N)) / (My.HHV(HIGH, N) - My.LLV(LOW, N) + 1e-10) * 100
        )
        K = My.EMA(RSV, (M1 * 2 - 1))
        D = My.EMA(K, (M2 * 2 - 1))
        return My._ensure_np_output(D)

    @staticmethod
    def KDJ_J(CLOSE, HIGH, LOW, N=9, M1=3, M2=3):
        """KDJ指标"""
        RSV = (
            (CLOSE - My.LLV(LOW, N)) / (My.HHV(HIGH, N) - My.LLV(LOW, N) + 1e-10) * 100
        )
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
        RS = EMA_UP / (EMA_DOWN + 1e-10)

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
        WR = (
            (My.HHV(HIGH, N) - CLOSE) / (My.HHV(HIGH, N) - My.LLV(LOW, N) + 1e-10) * 100
        )
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
        return My._ensure_np_output(
            (TP - My.MA(TP, N)) / (0.015 * My.AVEDEV(TP, N) + 1e-10)
        )

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
        return My._ensure_np_output(
            (My.MA(CLOSE, M1) + My.MA(CLOSE, M2) + My.MA(CLOSE, M3) + My.MA(CLOSE, M4))
            / 4
        )

    @staticmethod
    def DMI(CLOSE, HIGH, LOW, M1=14, M2=6):
        """动向指标，返回 (PDI, MDI, ADX, ADXR) 元组"""
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
        PDI = DMP * 100 / (TR + 1e-10)
        MDI = DMM * 100 / (TR + 1e-10)
        ADX = My.MA(My.ABS(MDI - PDI) / (PDI + MDI + 1e-10) * 100, M2)
        ADXR = (ADX + My.REF(ADX, M2)) / 2
        return _to_np_output((PDI, MDI, ADX, ADXR))

    @staticmethod
    def ADX(CLOSE, HIGH, LOW, M1=14, M2=6):
        """ADX指标:结果和同花顺,通达信完全一致"""
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
            My.SUM(My.IF(CLOSE > LC, VOL, 0), M1)
            / My.SUM(My.IF(CLOSE <= LC, VOL, 0), M1)
            * 100
        )

    @staticmethod
    def CR(CLOSE, HIGH, LOW, N=20):
        """CR价格动量指标"""
        HIGH = My._ensure_array(HIGH)
        LOW = My._ensure_array(LOW)
        CLOSE = My._ensure_array(CLOSE)
        MID = My.REF(HIGH + LOW + CLOSE, 1) / 3
        return My._ensure_np_output(
            My.SUM(My.MAX(0, HIGH - MID), N) / My.SUM(My.MAX(0, MID - LOW), N) * 100
        )

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
        BR = (
            My.SUM(My.MAX(0, HIGH - My.REF(CLOSE, 1)), M1)
            / My.SUM(My.MAX(0, My.REF(CLOSE, 1) - LOW), M1)
            * 100
        )
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
        OBV = My.SUM(
            My.IF(
                CLOSE > My.REF(CLOSE, 1), VOL, My.IF(CLOSE < My.REF(CLOSE, 1), -VOL, 0)
            ),
            0,
        )
        return My._ensure_np_output(OBV)

    @staticmethod
    def MFI(CLOSE, HIGH, LOW, VOL, N=14):
        """MFI指标是成交量的RSI指标"""
        HIGH = My._ensure_array(HIGH)
        LOW = My._ensure_array(LOW)
        CLOSE = My._ensure_array(CLOSE)
        VOL = My._ensure_array(VOL)
        TYP = (HIGH + LOW + CLOSE) / 3
        V1 = My.SUM(My.IF(TYP > My.REF(TYP, 1), TYP * VOL, 0), N) / My.SUM(
            My.IF(TYP < My.REF(TYP, 1), TYP * VOL, 0), N
        )
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


import warnings
import asyncio
import nest_asyncio
from qmf_data import load_kline
import qmf_model_sdk
import alphalens

nest_asyncio.apply()
warnings.filterwarnings("ignore", category=RuntimeWarning)

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
    "时间分类": {
        # 日盘商品:无夜盘，标准日盘时间（09:00-15:00），主要为农产品和部分其他商品
        "日盘商品": [
            "wr",
            "AP",
            "CJ",
            "PK",
            "RS",
            "UR",
            "fb",
            "jd",
            "lh",
            "lg",
            "ec",
            "lc",
            "si",
            "ps",
        ],
        # 标准夜盘商品:夜盘21:00-23:00，品种最多，涵盖化工、农产品、黑色金属等
        "标准夜盘商品": [
            "sp",
            "ru",
            "fu",
            "hc",
            "rb",
            "op",
            "CF",
            "CY",
            "fg",
            "MA",
            "OI",
            "PF",
            "PR",
            "PL",
            "PX",
            "RM",
            "SA",
            "SH",
            "SR",
            "TA",
            "bz",
            "a",
            "b",
            "c",
            "cs",
            "eb",
            "eg",
            "i",
            "j",
            "jm",
            "l",
            "m",
            "p",
            "pg",
            "pp",
            "v",
            "y",
            "lu",
            "nr",
        ],
        # 有色金属:夜盘21:00-01:00（跨日），主要为有色金属品种
        "有色金属": ["ao", "ni", "pb", "sn", "ss", "zn", "al", "cu", "bc"],
        # 贵金属与原油:夜盘21:00-02:30（跨日，最长）
        "贵金属与原油": ["au", "ag", "sc"],
        # 股指期货:无夜盘，日盘时间09:30-15:00，股指期货品种
        "股指期货": ["IC", "IF", "IH", "IM"],
        # 国债期货:无夜盘，日盘时间09:30-15:15，国债期货品种
        "国债期货": ["T", "TF", "TS", "TL"],
    },
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

SELECTED_SECTOR = [
    "时间分类",
    "有色金属",
]  # 设置为 None 表示手动指定合约，或设置为上述选项之一
# SELECTED_SECTOR = None  # 设置为 None 表示手动指定合约，或设置为上述选项之一

# ========== 手动指定合约列表（当 SELECTED_SECTOR 为 None 时使用）==========
# 合约列表（支持多个合约），格式为 "合约代码888"
# MANUAL_SYMBOLS = ["au888", "rb888","cu888","jd888","AP888"]  # 例如：["au888", "rb888", "cu888"]
MANUAL_SYMBOLS = ["au888", "ag888"]  # 例如：["au888", "rb888", "cu888"]


def get_symbols_by_sector(selected, sector_map, manual_symbols):
    if selected is None:
        return manual_symbols
    if selected == "all":
        return [
            f"{c}888" for s in sector_map.values() for cats in s.values() for c in cats
        ]
    if isinstance(selected, str):
        if selected in sector_map:
            return [f"{c}888" for cats in sector_map[selected].values() for c in cats]
        log_print(f"警告：未知的板块选择 '{selected}'，使用手动指定的合约列表")
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
                    log_print(f"警告：未找到板块 '{item}'，跳过")
                    all_valid = False
            elif isinstance(item, list) and len(item) == 2:
                sec, cat = item
                if sec in sector_map and cat in sector_map[sec]:
                    result.extend([f"{c}888" for c in sector_map[sec][cat]])
                else:
                    log_print(f"警告：未找到板块 '{sec}' 的子板块 '{cat}'，跳过")
                    all_valid = False
            else:
                log_print(f"警告：无效的选择项 '{item}'，跳过")
                all_valid = False

        if len(selected) == 2 and all(isinstance(x, str) for x in selected):
            sec, cat = selected
            if sec in sector_map and cat in sector_map[sec]:
                return [f"{c}888" for c in sector_map[sec][cat]]

        for item in selected:
            handle_item(item)

        if not all_valid:
            log_print("部分选择项无效，已使用有效的选择项生成合约列表")

        return result if result else manual_symbols

    log_print("警告：列表格式不正确，使用手动指定的合约列表")
    return manual_symbols


SYMBOLS = get_symbols_by_sector(SELECTED_SECTOR, FUTURES_SECTORS, MANUAL_SYMBOLS)

log_print(f"当前使用的合约列表（共 {len(SYMBOLS)} 个合约）：")
log_print(SYMBOLS)

IC_PERIOD = 1  # （默认 5）
IC_QUANTILES = 5  # （默认 5）
IC_WEIGHT_A = 0.4  # （默认 0.4/0.6）
IC_WEIGHT_B = 0.6  # （默认 0.4/0.6）
FITNESS_W_TRAIN = 0.4  # （默认 0.3/0.7）
FITNESS_W_TEST = 0.6  # （默认 0.3/0.7）
FITNESS_OVERFIT_LAMBDA = 0.2  # （默认 0.2）
FITNESS_SCHEME = "B"  # （"A" 或 "B"，默认 "B"）

# 时间范围
# BEGIN_TIME = "2020-01-01"
# END_TIME = "2023-12-31"
# BEGIN_TIME = "2025-06-01"
# END_TIME = "2025-12-31"
BEGIN_TIME = "2025-06-01"
END_TIME = "2025-08-31"
BEGIN_TIME_TEST = "2025-09-01"
END_TIME_TEST = "2025-12-31"
# 数据周期
SYMBOL_CYCLE = "15分钟"  # 例如："1天", "1小时"等
# SYMBOL_CYCLE = "1天"  # 例如："1天", "1小时"等

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

# 获取所有合约的数据，获取后将这部分注释掉，已注释
log_print("开始获取数据...")
# for symbol in SYMBOLS:
#     log_print(f"训练集：正在获取 {symbol} 的数据...")
#     asyncio.run(qmf_model_sdk.get_futures_data(symbol, BEGIN_TIME, END_TIME, SYMBOL_CYCLE))
# for symbol in SYMBOLS:
#     log_print(f"测试集：正在获取 {symbol} 的数据...")
#     asyncio.run(qmf_model_sdk.get_futures_data(symbol, BEGIN_TIME_TEST, END_TIME_TEST, SYMBOL_CYCLE))
log_print("数据获取完成！")


def get_futures_data(symbol, start_time=None, end_time=None, symbol_cycle=None):
    """获取单个合约的数据"""
    if symbol_cycle is None:
        symbol_cycle = SYMBOL_CYCLE
    
    if start_time is None:
        start_time = f"{BEGIN_TIME} 00:00:00"
    elif len(str(start_time)) == 10:
        start_time = f"{start_time} 00:00:00"

    if end_time is None:
        end_time = f"{END_TIME} 23:59:59"
    elif len(str(end_time)) == 10:
        end_time = f"{end_time} 23:59:59"

    data = load_kline(
        product=symbol,
        cycle=SYMBOL_CYCLE_MAP[symbol_cycle],
        start_time=start_time,
        end_time=end_time,
    )

    # 检查数据是否为空或没有 'date' 列
    if data is None or len(data) == 0 or "date" not in data.columns:
        log_print(f"警告：{symbol} 在指定时间范围内没有数据，跳过该合约")
        return pd.DataFrame(
            columns=[
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "open_interest",
                "asset",
            ]
        )

    data["date"] = pd.to_datetime(data["date"] + data["time"], errors="coerce")
    data.reset_index(inplace=True, drop=True)
    data["asset"] = symbol
    return data


# 获取所有合约数据并合并
df_list = []
df_list_test = []
for symbol in SYMBOLS:
    data = get_futures_data(symbol)
    if len(data) > 0:
        df_list.append(data)
    data_test = get_futures_data(symbol, BEGIN_TIME_TEST, END_TIME_TEST)
    if len(data_test) > 0:
        df_list_test.append(data_test)

# 检查是否有有效数据
if len(df_list) == 0:
    raise ValueError(
        "错误：所有合约在指定时间范围内都没有数据，请检查时间范围或合约代码！"
    )

if len(df_list_test) == 0:
    raise ValueError(
        "错误：所有合约在指定时间范围内都没有数据，请检查时间范围或合约代码！"
    )

df = pd.concat(df_list, ignore_index=True)
df = df[["date", "open", "high", "low", "close", "volume", "open_interest", "asset"]]

df_test = pd.concat(df_list_test, ignore_index=True)
df_test = df_test[
    ["date", "open", "high", "low", "close", "volume", "open_interest", "asset"]
]

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

# 去重
df.drop_duplicates(subset=["asset", "date"], keep="last", inplace=True)
df.sort_values(["asset", "date"], ignore_index=True, inplace=True)

df_test.drop_duplicates(subset=["asset", "date"], keep="last", inplace=True)
df_test.sort_values(["asset", "date"], ignore_index=True, inplace=True)

log_print(f"训练集数据预处理完成，共 {len(df)} 条记录")

log_print(f"测试集数据预处理完成，共 {len(df_test)} 条记录")

# 计算未来5天收益（目标变量）
log_print("计算训练集目标变量...")
df = df.sort_values(["asset", "date"]).reset_index(drop=True)
df["future_return"] = df.groupby("asset")["close"].shift(-IC_PERIOD) / df["close"] - 1

log_print("计算测试集目标变量...")
df_test = df_test.sort_values(["asset", "date"]).reset_index(drop=True)
df_test["future_return"] = (
    df_test.groupby("asset")["close"].shift(-IC_PERIOD) / df_test["close"] - 1
)

# 删除目标变量为空的行
df = df.dropna(subset=["future_return"]).reset_index(drop=True)
df_test = df_test.dropna(subset=["future_return"]).reset_index(drop=True)

# 将列名适配：date -> time, asset -> code
df["time"] = df["date"]
df["code"] = df["asset"]
df_test["time"] = df_test["date"]
df_test["code"] = df_test["asset"]

# 选择特征和目标（只保留基础行情数据，技术指标在遗传编程中通过函数调用生成）
features = ["open", "close", "high", "low", "volume", "open_interest"]
# features = []
target = "future_return"

# 数据预处理
log_print(f"原始数据行数: {len(df)}")
data = df[features + [target, "code", "time"]].copy()
data_test = df_test[features + [target, "code", "time"]].copy()

# 构造训练集X, y（T, N）格式，T为时间，N为合约数
log_print("\n构造训练数据...")
pivoted = {}
all_cols = features + [target]
unique_cols = list(dict.fromkeys(all_cols))
if len(unique_cols) != len(all_cols):
    dup_cols = [c for c in set(all_cols) if all_cols.count(c) > 1]
    logger.info("发现重复列，已去重: %s", dup_cols)
for col in unique_cols:
    try:
        pivoted[col] = data.pivot(index="time", columns="code", values=col)
        log_print(f"特征 {col} 转换成功，形状: {pivoted[col].shape}")
    except Exception as e:
        log_print(f"特征 {col} 转换失败: {e}")
        # 创建全零矩阵作为替代
        unique_dates = pd.Series(data["time"]).unique()
        unique_codes = pd.Series(data["code"]).unique()
        pivoted[col] = pd.DataFrame(0, index=unique_dates, columns=unique_codes)

# 构造测试集X, y（T, N）格式，T为时间，N为合约数
log_print("\n构造测试数据...")
pivoted_test = {}
all_cols_test = features + [target]
unique_cols_test = list(dict.fromkeys(all_cols_test))
if len(unique_cols_test) != len(all_cols_test):
    dup_cols_test = [c for c in set(all_cols_test) if all_cols_test.count(c) > 1]
    logger.info("发现重复列，已去重: %s", dup_cols_test)
for col in unique_cols_test:
    try:
        pivoted_test[col] = data_test.pivot(index="time", columns="code", values=col)
        log_print(f"特征 {col} 转换成功，形状: {pivoted_test[col].shape}")
    except Exception as e:
        log_print(f"特征 {col} 转换失败: {e}")
        # 创建全零矩阵作为替代
        unique_dates_test = pd.Series(data_test["time"]).unique()
        unique_codes_test = pd.Series(data_test["code"]).unique()
        pivoted_test[col] = pd.DataFrame(
            0, index=unique_dates_test, columns=unique_codes_test
        )

# 训练集
X_dict = {f: pivoted[f].values for f in features}
y = pivoted[target].values  # (T, N)
log_print(f"\n训练数据准备完成:")
log_print(f"时间点数量: {y.shape[0]}")
log_print(f"合约数量: {y.shape[1]}")
log_print(f"特征数量: {len(features)}")
log_print(
    f"特征列表: {features[:10]}..." if len(features) > 10 else f"特征列表: {features}"
)
# log_print(X_dict) 输出数据示例，已注释掉

# 将X_dict整体保存为一个CSV文件，已注释掉
# 合并为多层列索引的DataFrame，索引为时间，列为MultiIndex(特征, 合约)
# multi_cols = pd.MultiIndex.from_product([features, pivoted[features[0]].columns], names=["feature", "code"])
# X_df = pd.DataFrame(
#     np.hstack([X_dict[f] for f in features]),
#     index=pivoted[features[0]].index,
#     columns=multi_cols
# )
# X_df.to_csv("train_X_dict_all.csv")
# log_print("训练集X_dict已保存为 train_X_dict_all.csv。")

# 测试集
X_dict_test = {f: pivoted_test[f].values for f in features}
y_test = pivoted_test[target].values  # (T, N)
log_print(f"\n测试数据准备完成:")
log_print(f"时间点数量: {y_test.shape[0]}")
log_print(f"合约数量: {y_test.shape[1]}")
log_print(f"特征数量: {len(features)}")
log_print(
    f"特征列表: {features[:10]}..." if len(features) > 10 else f"特征列表: {features}"
)

# 将X_dict_test整体保存为一个CSV文件
# multi_cols_test = pd.MultiIndex.from_product([features, pivoted_test[features[0]].columns], names=["feature", "code"])
# X_df_test = pd.DataFrame(
#     np.hstack([X_dict_test[f] for f in features]),
#     index=pivoted_test[features[0]].index,
#     columns=multi_cols_test
# )
# X_df_test.to_csv("test_X_dict_all.csv")
# log_print("测试集X_dict_test已保存为 test_X_dict_all.csv。")


# 检查训练集数据质量
log_print(f"\n数据质量检查:")
log_print(f"y中NaN比例: {np.isnan(y).sum() / y.size:.2%}")
log_print(f"y中有效值数量: {np.sum(~np.isnan(y))}")
# 打印训练集最小时间和最大时间
train_times = (
    pivoted[features[0]].index if features and features[0] in pivoted else None
)
if train_times is not None and len(train_times) > 0:
    log_print(f"训练集最小时间: {train_times.min()}, 最大时间: {train_times.max()}")

for f in features[:5]:  # 只检查前5个特征
    if f in X_dict:
        nan_ratio = np.isnan(X_dict[f]).sum() / X_dict[f].size
        log_print(f"  {f}: NaN比例={nan_ratio:.2%}")

# 检查测试集数据质量
log_print(f"\n测试集数据质量检查:")
log_print(f"y_test中NaN比例: {np.isnan(y_test).sum() / y_test.size:.2%}")
log_print(f"y_test中有效值数量: {np.sum(~np.isnan(y_test))}")
# 打印测试集最小时间和最大时间
test_times = (
    pivoted_test[features[0]].index
    if features and features[0] in pivoted_test
    else None
)
if test_times is not None and len(test_times) > 0:
    log_print(f"测试集最小时间: {test_times.min()}, 最大时间: {test_times.max()}")

for f in features[:5]:  # 只检查前5个特征
    if f in X_dict_test:
        nan_ratio = np.isnan(X_dict_test[f]).sum() / X_dict_test[f].size
        log_print(f"  {f}: NaN比例={nan_ratio:.2%}")

X_TRAIN_SHAPE = X_dict["close"].shape
X_TEST_SHAPE = X_dict_test["close"].shape


def _trace_pred_all_nan_inf(prog, X_dict, function_set=None):
    try:
        trace_state = {"reported": False, "mode": "first", "path": []}
        fs = function_set or FunctionSet()
        _ = prog.evaluate(X_dict, fs, trace_state=trace_state)
        if not trace_state.get("reported"):
            log_print(
                "警告: 溯源未定位到全为NaN或inf的节点，可能在后处理或广播阶段产生。"
            )
    except Exception as e:
        log_print(f"警告: 溯源过程中异常: {e}")


def calculate_turnover(factor_data, period, quantiles):
    """
    计算因子换手率

    参数:
        factor_data: Alphalens 处理后的因子数据，包含 factor_quantile 列
        period: 持有期
        quantiles: 分位数数量

    返回:
        pd.DataFrame: 每个时间点每个分位数的换手率
    """
    # 获取分位数信息
    quantile_series = factor_data["factor_quantile"].copy()

    # 重置索引以便操作
    quantile_df = quantile_series.reset_index()
    quantile_df["date"] = pd.to_datetime(quantile_df["date"])

    # 按日期排序
    quantile_df = quantile_df.sort_values(["date", "asset"])

    # 获取所有日期
    dates = quantile_df["date"].unique()
    dates = sorted(dates)

    turnover_data = []

    for i in range(len(dates) - period):
        current_date = dates[i]
        future_date = dates[i + period] if i + period < len(dates) else None

        if future_date is None:
            continue

        # 获取当前日期和未来日期的分位数分配
        current_quantiles = quantile_df[quantile_df["date"] == current_date].set_index(
            "asset"
        )["factor_quantile"]
        future_quantiles = quantile_df[quantile_df["date"] == future_date].set_index(
            "asset"
        )["factor_quantile"]

        # 计算每个分位数的换手率
        for q in range(1, quantiles + 1):
            # 当前日期在该分位数的资产
            current_assets = set(current_quantiles[current_quantiles == q].index)

            if len(current_assets) == 0:
                turnover = 0.0
            else:
                # 未来日期仍在该分位数的资产
                future_assets = set(future_quantiles[future_quantiles == q].index)
                # 换手率 = 1 - 保留率
                turnover = 1.0 - len(current_assets & future_assets) / len(
                    current_assets
                )

            turnover_data.append(
                {"date": current_date, "quantile": q, "turnover": turnover}
            )

    if not turnover_data:
        return pd.DataFrame()

    turnover_df = pd.DataFrame(turnover_data)
    turnover_pivot = turnover_df.pivot(
        index="date", columns="quantile", values="turnover"
    )
    turnover_pivot.columns = [f"Quantile {q}" for q in turnover_pivot.columns]

    return turnover_pivot


def calculate_metrics(factor_data, periods, quantiles, trading_days_per_year=252):
    """
    计算因子测试指标

    参数:
        factor_data: Alphalens 处理后的因子数据
        periods: 持有期列表
        quantiles: 分位数数量
        trading_days_per_year: 年化交易日数

    返回:
        dict: 包含所有指标的字典
    """
    results = {}

    for period in periods:
        period_col = f"{period}D"
        if period_col not in factor_data.columns:
            continue

        period_results = {}

        # 按分位数分组计算收益
        quantile_returns = factor_data.groupby("factor_quantile")[period_col].mean()

        # 计算所有分位数的平均收益（作为基准）
        benchmark_return = factor_data[period_col].mean()

        # 计算超额收益（相对于基准）
        excess_returns = quantile_returns - benchmark_return

        # 年化超额收益率
        annualized_excess_returns = excess_returns * (trading_days_per_year / period)

        if annualized_excess_returns.empty:
            period_results["min_quantile_excess_annual_return"] = np.nan
            period_results["max_quantile_excess_annual_return"] = np.nan
            period_results["min_quantile"] = np.nan
            period_results["max_quantile"] = np.nan
        else:
            # 最小和最大分位数的超额年化收益率
            min_quantile = annualized_excess_returns.idxmin()
            max_quantile = annualized_excess_returns.idxmax()

            period_results["min_quantile_excess_annual_return"] = (
                annualized_excess_returns.min()
            )
            period_results["max_quantile_excess_annual_return"] = (
                annualized_excess_returns.max()
            )
            period_results["min_quantile"] = min_quantile
            period_results["max_quantile"] = max_quantile

        # 计算换手率
        turnover = calculate_turnover(factor_data, period, quantiles)
        if turnover is not None and len(turnover) > 0:
            # 获取每个分位数的换手率
            quantile_turnover = {}
            for q in range(1, quantiles + 1):
                col_name = f"Quantile {q}"
                if col_name in turnover.columns:
                    quantile_turnover[q] = turnover[col_name].mean()

            if quantile_turnover:
                period_results["min_quantile_turnover"] = min(
                    quantile_turnover.values()
                )
                period_results["max_quantile_turnover"] = max(
                    quantile_turnover.values()
                )
                period_results["min_quantile_turnover_id"] = min(
                    quantile_turnover, key=quantile_turnover.get
                )
                period_results["max_quantile_turnover_id"] = max(
                    quantile_turnover, key=quantile_turnover.get
                )

        # 计算IC（信息系数）
        # 注意：factor_information_coefficient 返回所有 period 的 IC，需要选择对应的 period
        # group_adjust=False 表示不进行分组调整，避免需要 'group' 列
        try:
            ic_all = alphalens.performance.factor_information_coefficient(
                factor_data, group_adjust=False, by_group=False
            )
            # 从返回的 DataFrame 中选择对应的 period 列
            if ic_all is not None and len(ic_all) > 0:
                period_col_name = f"{period}D"
                if period_col_name in ic_all.columns:
                    ic = ic_all[period_col_name].dropna()
                    if len(ic) > 0:
                        period_results["IC_mean"] = ic.mean()
                        period_results["IC_std"] = ic.std()
                        period_results["IC_IR"] = (
                            ic.mean() / ic.std() if ic.std() != 0 else np.nan
                        )
                        period_results["IC_positive_ratio"] = (ic > 0).sum() / len(ic)
        except Exception as e:
            log_print(f"计算 {period}D 的 IC 时出错: {e}")
            # 如果出错，手动计算 IC（因子值与未来收益率的相关系数）
            try:
                # 获取因子值和对应的未来收益
                ic_data = factor_data[[period_col, "factor"]].dropna()
                if len(ic_data) > 0:
                    ic_series = ic_data.groupby(
                        ic_data.index.get_level_values("date")
                    ).apply(lambda x: x["factor"].corr(x[period_col]))
                    if len(ic_series) > 0:
                        period_results["IC_mean"] = ic_series.mean()
                        period_results["IC_std"] = ic_series.std()
                        period_results["IC_IR"] = (
                            ic_series.mean() / ic_series.std()
                            if ic_series.std() != 0
                            else np.nan
                        )
                        period_results["IC_positive_ratio"] = (
                            ic_series > 0
                        ).sum() / len(ic_series)
            except Exception as e2:
                log_print(f"手动计算 IC 也失败: {e2}")

        results[f"{period}D"] = period_results

    return results


def fitness_func(
    prog,
    X_train,
    y_train,
    return_details=False,
    ic_objective="max",
    function_set=None,
    X_dict_test=None,
    y_test=None,
    min_ts_points=10,
    min_valid_ts_ratio=0.6,
):
    """
    计算因子表达式的适应度（时间截面rankIC均值）

    参数:
        factor: 因子 (T, N)
        y: (T, N) 收益
        return_details: 是否返回详细信息（用于分析）

    返回:
        float: 所有时间截面rankIC的均值
    """
    try:
        fs = function_set or FunctionSet()
        # 动态约束：截面最小样本数不超过当前合约数量，且至少为2
        # 固定目标阈值为20，但不再暴露min_cs_points参数
        target_min_cs_points = 20
        n_cs_train = y_train.shape[1] if y_train.ndim >= 2 else 0
        min_cs_points_train = (
            max(2, min(int(target_min_cs_points), int(n_cs_train)))
            if n_cs_train > 0
            else 2
        )

        # 计算因子值（使用传入的function_set实例，避免重复创建）
        pred = prog.evaluate(X_train, fs)  # (T, N)

        # 确保pred是二维数组
        # 全量兜底，保证pred是float64二维数组
        pred = np.asarray(pred, dtype=np.float64)
        if pred.shape != y_train.shape:
            log_print("pred.shape != y_train.shape")
            if return_details:
                return {
                    "fitness": -np.inf,
                    "mean_ic": -np.inf,
                    "icir": -np.inf,
                    "valid_ts": 0,
                    "total_ts": y_train.shape[0],
                }
            return -np.inf if ic_objective == "max" else np.inf

        # 向量化计算每个时间截面的rankIC
        ic_values = []
        valid_ts = 0

        for t in range(y_train.shape[0]):
            # 当前时间截面的预测值和真实值
            pred_t = pred[t].ravel()
            y_t = y_train[t].ravel()

            # 过滤缺失值
            mask = ~np.isnan(pred_t) & ~np.isnan(y_t)

            if np.sum(mask) < min_cs_points_train:
                continue

            # 计算Spearman秩相关系数（优先用scipy.rankdata）
            pred_rank = _rankdata(pred_t[mask], method="average")
            y_rank = _rankdata(y_t[mask], method="average")
            corr = np.corrcoef(pred_rank, y_rank)[0, 1]

            if not np.isnan(corr):
                ic_values.append(corr)
                valid_ts += 1

        # 训练集最小有效时点阈值：同时满足绝对阈值与覆盖率阈值
        min_required_ts_train = max(
            int(min_ts_points),
            int(np.ceil(float(y_train.shape[0]) * float(min_valid_ts_ratio))),
        )
        # 计算最终适应度（增加稳定性惩罚项）
        if len(ic_values) < min_required_ts_train:
            # log_print(len(ic_values), pred.shape)
            # 输出因子表达式，已注释掉
            # log_print(
            #     f"有效时点不足(train): {valid_ts}/{y_train.shape[0]} < {min_required_ts_train} "
            #     f"(min_ts={min_ts_points}, 覆盖率阈值={min_valid_ts_ratio:.0%}), 表达式: {prog.to_str()}"
            # )
            if return_details:
                return {
                    "fitness": -np.inf,
                    "mean_ic": -np.inf,
                    "icir": -np.inf,
                    "valid_ts": 0,
                    "total_ts": y_train.shape[0],
                    "valid_ts_test": 0,
                    "total_ts_test": y_test.shape[0] if y_test is not None else 0,
                }
            return -np.inf if ic_objective == "max" else np.inf

        mean_ic = np.mean(ic_values)
        icir = mean_ic / (np.std(ic_values) + 1e-8)

        # 测试集IC IR计算
        mean_ic_test = np.nan
        icir_test = np.nan
        valid_ts_test = 0
        if X_dict_test is not None and y_test is not None:
            pred_test = prog.evaluate(X_dict_test, fs)
            factor_test = pred_test

        if X_dict_test is not None and y_test is not None and factor_test is not None:
            n_cs_test = y_test.shape[1] if y_test.ndim >= 2 else 0
            min_cs_points_test = (
                max(2, min(int(target_min_cs_points), int(n_cs_test)))
                if n_cs_test > 0
                else 2
            )
            ic_values_test = []
            valid_ts_test = 0
            for t in range(y_test.shape[0]):
                pred_test_t = factor_test[t].ravel()
                y_test_t = y_test[t].ravel()

                mask_test = ~np.isnan(pred_test_t) & ~np.isnan(y_test_t)

                if np.sum(mask_test) < min_cs_points_test:
                    continue

                pred_rank_test = _rankdata(pred_test_t[mask_test], method="average")
                y_rank_test = _rankdata(y_test_t[mask_test], method="average")
                corr_test = np.corrcoef(pred_rank_test, y_rank_test)[0, 1]

                if not np.isnan(corr_test):
                    ic_values_test.append(corr_test)
                    valid_ts_test += 1

            # 测试集最小有效时点阈值：同时满足绝对阈值与覆盖率阈值
            min_required_ts_test = max(
                int(min_ts_points),
                int(np.ceil(float(y_test.shape[0]) * float(min_valid_ts_ratio))),
            )
            # 计算最终适应度（增加稳定性惩罚项）
            if len(ic_values_test) < min_required_ts_test:
                # # 输出因子表达式，已注释掉
                # log_print(
                #     f"有效时点不足(test): {valid_ts_test}/{y_test.shape[0]} < {min_required_ts_test} "
                #     f"(min_ts={min_ts_points}, 覆盖率阈值={min_valid_ts_ratio:.0%}), 表达式: {prog.to_str()}"
                # )
                if return_details:
                    return {
                        "fitness": -np.inf,
                        "mean_ic": -np.inf,
                        "icir": -np.inf,
                        "valid_ts": 0,
                        "total_ts": y_train.shape[0],
                        "valid_ts_test": 0,
                        "total_ts_test": y_test.shape[0],
                    }
                return -np.inf if ic_objective == "max" else np.inf

            mean_ic_test = np.mean(ic_values_test)
            icir_test = mean_ic_test / (np.std(ic_values_test) + 1e-8)

        # # 综合评分：IC均值 + ICIR（稳定性），已注释掉
        # fitness = mean_ic + 0.2 * np.tanh(icir)

        # 综合评分：IC均值 + ICIR（稳定性）
        if ic_objective not in ("max", "min"):
            ic_objective = "max"
        ic_sign = 1.0 if ic_objective == "max" else -1.0

        a = globals().get("IC_WEIGHT_A", 0.4)
        b = globals().get("IC_WEIGHT_B", 0.6)
        base_train = ic_sign * (a * mean_ic + b * np.tanh(icir))
        base_test = ic_sign * (a * mean_ic_test + b * np.tanh(icir_test))

        w_train = globals().get("FITNESS_W_TRAIN", 0.4)
        w_test = globals().get("FITNESS_W_TEST", 0.6)
        fitness_scheme = globals().get("FITNESS_SCHEME", "B")
        overfit_lambda = globals().get("FITNESS_OVERFIT_LAMBDA", 0.2)
        if not np.isfinite(base_train) or not np.isfinite(base_test):
            base_fitness = -np.inf
        elif fitness_scheme == "A":
            base_fitness = w_train * base_train + w_test * base_test
        else:
            base_fitness = (
                w_train * base_train
                + w_test * base_test
                - overfit_lambda * abs(base_train - base_test)
            )

        if return_details:
            return {
                "fitness": base_fitness,
                "mean_ic": mean_ic,
                "icir": icir,
                "valid_ts": valid_ts,
                "total_ts": y_train.shape[0],
                "mean_ic_test": mean_ic_test,
                "icir_test": icir_test,
                "valid_ts_test": valid_ts_test,
                "total_ts_test": y_test.shape[0],
            }

        return base_fitness

    except Exception as e:
        log_print(f"适应度计算错误: {e}")
        return -np.inf


# 优化后的遗传规划器配置
gp = GeneticProgrammer(
    generations=15,
    population_size=120,
    tournament_size=4,
    n_components=5,
    hall_of_fame=6,
    function_set=FunctionSet(),
    variable_names=features,
    ts_window=20,
    random_state=None,
    const_range=(2, 120),
    p_crossover=0.30,
    p_subtree_mutation=0.30,
    p_hoist_mutation=0.10,
    p_point_mutation=0.20,
    immigration_rate=0.20,
    parsimony_coefficient=0.002,
    init_depth=(3, 8),
    suit_size=(4, 14),
    stagnation_threshold=6,
    min_improvement=0.001,
    max_restarts=3,
    max_program_size=24,
    max_best_program_size=24,
    ic_objective="max",
    # 限制某些函数内某些参数是否允许为常数, True表示允许, False表示不允许, 列表表示允许的参数位置, 不填表示允许所有参数
    allow_const_in_function={
        # 基础数学运算（2个参数）
        "ADD": [False, True],
        "SUB": [False, True],
        "MUL": [False, True],
        "DIV": [False, True],
        # 单元函数（1个参数）
        "INV": [False],
        "SIGNEDPOWER": [False],
        'SIGMOID': [False],
        # 新增算子函数（1个参数）
        "ABS": [False],  # 绝对值
        "LN": [False],  # 自然对数
        "SQRT": [False],  # 平方根
        "SIN": [False],  # 正弦
        "COS": [False],  # 余弦
        "TAN": [False],  # 正切
        # 新增算子函数（2个参数）
        "MAX": [False, False],  # 最大值
        "MIN": [False, False],  # 最小值
        "REF": [False, True],  # 引用
        "DIFF": [False, True],  # 差分
        "HHV": [False, True],  # 最高值
        "HV": [False, True],  # 最高值（不包括当前K线）
        "LLV": [False, True],  # 最低值
        "LV": [False, True],  # 最低值（不包括当前K线）
        "HHVBARS": [False, True],  # 最高值到当前周期数
        "LLVBARS": [False, True],  # 最低值到当前周期数
        "MA": [False, True],  # 简单移动平均
        "EMA": [False, True],  # 指数移动平均
        "SMA": [False, True, True],  # 中国式SMA（3个参数：x, d, m）
        "WMA": [False, True],  # 加权移动平均
        "DMA": [False, True],  # 动态移动平均
        "AVEDEV": [False, True],  # 平均绝对偏差
        "SLOPE": [False, True],  # 线性回归斜率
        "FORCAST": [False, True],  # 线性回归预测值
        "TOPRANGE": [False],  # 最高价周期数
        "LOWRANGE": [False],  # 最低价周期数
        # 时间序列函数（2个参数：数据+窗口）
        "RANK": [False, True],
        "SCALE": [False, True],
        "TS_RANK": [False, True],
        "TS_ZSCORE": [False, True],
        # 双变量时间序列函数（3个参数：数据1+数据2+窗口）
        "CORR": [False, False, True],
        "COVA": [False, False, True],
        "RANK_SUB": [False, False, True],
        "RANK_DIV": [False, False, True],
    },
)

start_time = time.time()
log_print("训练前关键配置检查：")
log_print(f"  SYMBOLS数量: {len(SYMBOLS)}")
log_print(f"  训练集shape: {X_dict['close'].shape}")
log_print(f"  测试集shape: {X_dict_test['close'].shape}")
log_print(f"  IC_PERIOD: {IC_PERIOD}")
log_print(f"  FITNESS_W_TRAIN: {FITNESS_W_TRAIN}")
log_print(f"  FITNESS_W_TEST: {FITNESS_W_TEST}")

# 训练
gp.fit(
    fitness_func=fitness_func,
    fitness_args=(X_dict, y),
    fitness_kwargs={"X_dict_test": X_dict_test, "y_test": y_test},
)

# 输出最优因子及其绩效指标
log_print("\n最优因子表达式及其绩效:")
log_print("=" * 100)

end_time = time.time()
elapsed_time = end_time - start_time
log_print(f"遗传编程训练完成，耗时: {elapsed_time:.2f} 秒")

def render_text_tree(node):
    def node_label(n):
        return n.name if n.value is None else str(n.value)

    def walk(n, prefix="", is_last=True, lines=None):
        if lines is None:
            lines = []
        connector = "└─ " if is_last else "├─ "
        lines.append(f"{prefix}{connector}{node_label(n)}")
        child_prefix = f"{prefix}{'   ' if is_last else '│  '}"
        for idx, child in enumerate(n.children):
            walk(child, child_prefix, idx == len(n.children) - 1, lines)
        return lines

    lines = [node_label(node)]
    for idx, child in enumerate(node.children):
        walk(child, "", idx == len(node.children) - 1, lines)
    return "\n".join(lines)


for i, prog in enumerate(gp.best_programs_[:]):  # 显示所有因子
    details = fitness_func(
        prog,
        X_dict,
        y,
        return_details=True,
        function_set=gp.function_set,
        X_dict_test=X_dict_test,
        y_test=y_test,
    )
    expr = prog.to_str()
    depth = prog.depth()
    size = prog.size()
    log_print(f"因子 {i + 1}:")
    log_print(f"  表达式: {expr}")
    log_print(
        f"  适应度: {details['fitness']:.6f} | 训练IC: {details['mean_ic']:.6f} | 训练IR: {details['icir']:.6f} | "
        f"测试IC: {details['mean_ic_test']:.6f} | 测试IR: {details['icir_test']:.6f}"
    )
    log_print(
        f"  复杂度: 深度={depth}, 节点数={size}, 有效时间点={details['valid_ts']}/{details['total_ts']}"
    )

    # 纯文本树形输出（逐行打印，便于日志查看）
    log_print("  因子树形结构:")
    for line in render_text_tree(prog).splitlines():
        log_print(f"  {line}")

    log_print("-" * 100)
