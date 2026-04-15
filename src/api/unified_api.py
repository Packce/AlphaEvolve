from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import uuid
import time
import os
import json
import base64
import hashlib
from pathlib import Path
import sys
from datetime import datetime
import threading
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning, module="matplotlib")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core", "factor"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core", "genetic"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))

SINGLE_FACTOR_AVAILABLE = None
MULTI_FACTOR_AVAILABLE = None
MINING_AVAILABLE = None

def check_dependencies():
    global SINGLE_FACTOR_AVAILABLE, MULTI_FACTOR_AVAILABLE, MINING_AVAILABLE
    import importlib.util

    def module_exists(module_name):
        spec = importlib.util.find_spec(module_name)
        if spec is not None:
            return True
        local_paths = [
            "src/core/factor",
            "src/core/genetic",
            os.path.join(os.path.dirname(__file__), "..", "core", "factor"),
            os.path.join(os.path.dirname(__file__), "..", "core", "genetic"),
        ]
        for path in local_paths:
            full_path = os.path.join(path, f"{module_name}.py")
            if os.path.exists(full_path):
                return True
        return False

    SINGLE_FACTOR_AVAILABLE = module_exists("single_factor_analysis")
    MULTI_FACTOR_AVAILABLE = module_exists("multi_factor_analysis")
    MINING_AVAILABLE = module_exists("factor_mining")

    if not SINGLE_FACTOR_AVAILABLE:
        print(f"警告: single_factor_analysis 模块未安装")
    if not MULTI_FACTOR_AVAILABLE:
        print(f"警告: multi_factor_analysis 模块未安装")
    if not MINING_AVAILABLE:
        print(f"警告: factor_mining 模块未安装")

check_dependencies()

app = FastAPI(
    title="量化分析统一API",
    description="单因子分析、多因子分析、因子挖掘统一服务",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = "api_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _merge_legacy_keys(data: Any, key_map: Dict[str, str]) -> Any:
    if not isinstance(data, dict):
        return data
    merged = dict(data)
    for old_key, new_key in key_map.items():
        if old_key in data and new_key not in merged:
            merged[new_key] = data[old_key]
    return merged


class BaseParams(BaseModel):
    begin_time: str = Field(default="2025-06-01", description="训练集开始时间")
    end_time: str = Field(default="2025-08-31", description="训练集结束时间")
    begin_time_test: str = Field(default="2025-09-01", description="测试集开始时间")
    end_time_test: str = Field(default="2025-12-31", description="测试集结束时间")
    begin_time_now: Optional[str] = Field(default="2026-01-01", description="验证集开始时间")
    symbol_cycle: str = Field(default="15分钟", description="数据周期")
    y_period: int = Field(default=1, ge=1, le=20, description="预测周期")
    selected_sector: Optional[List[str]] = Field(default=["时间分类", "有色金属"], description="选择的行业板块")
    symbols: Optional[List[str]] = Field(default=None, description="指定的期货代码列表")
    quantiles: int = Field(default=5, ge=2, le=10, description="分位数数量")

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_fields(cls, data: Any) -> Any:
        return _merge_legacy_keys(
            data,
            {
                "BEGIN_TIME": "begin_time",
                "END_TIME": "end_time",
                "BEGIN_TIME_TEST": "begin_time_test",
                "END_TIME_TEST": "end_time_test",
                "BEGIN_TIME_NOW": "begin_time_now",
                "SYMBOL_CYCLE": "symbol_cycle",
                "Y_PERIOD": "y_period",
                "SELECTED_SECTOR": "selected_sector",
                "SYMBOLS": "symbols",
                "MANUAL_SYMBOLS": "symbols",
                "QUANTILES": "quantiles",
            },
        )


class SingleFactorParams(BaseParams):
    formula: str = Field(default="RANK(WR(2), BOLL_UPPER(24, 2.26))", description="因子表达式")

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_single_fields(cls, data: Any) -> Any:
        return _merge_legacy_keys(data, {"FORMULA": "formula"})


DEFAULT_MULTI_FACTOR_FORMULAS = [
    "RANK(WR(2), BOLL_UPPER(24, 2.26))",
    "LOWRANGE(BIAS(90))",
    "HHVBARS(KDJ_K(4, 84), COS(MACD_DEA(49, 46, 88)))",
    "WR(2)",
]


class MultiFactorParams(BaseParams):
    formula: List[str] = Field(
        default_factory=lambda: DEFAULT_MULTI_FACTOR_FORMULAS.copy(),
        description="因子表达式列表",
    )
    use_lightgbm: bool = Field(default=True, description="是否使用LightGBM模型")
    use_elastic_net: bool = Field(default=True, description="是否使用Elastic Net模型")
    use_instashap: bool = Field(default=True, description="是否使用InstaSHAP模型")

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_multi_fields(cls, data: Any) -> Any:
        return _merge_legacy_keys(
            data,
            {
                "USE_LIGHTGBM": "use_lightgbm",
                "USE_ELASTIC_NET": "use_elastic_net",
                "USE_INSTASHAP": "use_instashap",
            },
        )


class MiningParams(BaseParams):
    generations: int = Field(default=15, ge=1, le=100, description="进化代数")
    population_size: int = Field(default=120, ge=10, le=1000, description="种群规模")
    tournament_size: int = Field(default=4, ge=2, le=20, description="锦标赛规模")
    n_components: int = Field(default=5, ge=1, le=20, description="保留的最优个体数量")
    hall_of_fame: int = Field(default=6, ge=1, le=20, description="精英保留数量")
    ts_window: int = Field(default=20, ge=5, le=250, description="时间窗口范围")
    const_range: tuple = Field(default=(-2, 120), description="常数范围")
    p_crossover: float = Field(default=0.30, ge=0, le=1, description="交叉概率")
    p_subtree_mutation: float = Field(default=0.30, ge=0, le=1, description="子树变异概率")
    p_hoist_mutation: float = Field(default=0.10, ge=0, le=1, description="提升变异概率")
    p_point_mutation: float = Field(default=0.20, ge=0, le=1, description="点变异概率")
    immigration_rate: float = Field(default=0.20, ge=0, le=1, description="每代注入随机个体比例")
    parsimony_coefficient: float = Field(default=0.002, ge=0, le=1, description="简约系数")
    init_depth: tuple = Field(default=(3, 8), description="初始树深度范围")
    suit_size: tuple = Field(default=(4, 14), description="表达树节点数上下界")
    stagnation_threshold: int = Field(default=6, ge=1, le=50, description="停滞检测阈值")
    min_improvement: float = Field(default=0.001, ge=0, le=1, description="最小显著提升阈值")
    max_restarts: int = Field(default=3, ge=0, le=10, description="最大自动重启次数")
    max_program_size: int = Field(default=24, ge=5, le=100, description="进化过程最大节点数限制")
    max_best_program_size: int = Field(default=24, ge=5, le=100, description="最终最优个体最大节点数限制")
    ic_objective: str = Field(default="max", description="IC优化方向: max 或 min")
    features: List[str] = Field(
        default=["open", "close", "high", "low", "volume", "open_interest"],
        description="使用的特征列表"
    )
    ic_period: int = Field(default=20, ge=5, le=250, description="IC计算周期")
    fitness_w_train: float = Field(default=0.6, ge=0, le=1, description="训练集适应度权重")
    fitness_w_test: float = Field(default=0.4, ge=0, le=1, description="测试集适应度权重")
    random_state: Optional[int] = Field(default=None, description="随机数种子")
    use_mock_data: bool = Field(default=False, description="是否使用模拟数据（仅用于测试）")


class TaskStatus(BaseModel):
    task_id: str
    task_type: str
    status: str
    progress: float
    message: str
    result: Optional[Dict[str, Any]] = None
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    elapsed_time: Optional[float] = None


tasks_storage: Dict[str, TaskStatus] = {}
tasks_lock = threading.Lock()
stop_flags: Dict[str, bool] = {}


def create_task(task_id: str, task_type: str, message: str = "任务已创建") -> TaskStatus:
    task = TaskStatus(
        task_id=task_id,
        task_type=task_type,
        status="pending",
        progress=0.0,
        message=message,
        created_at=datetime.now().isoformat()
    )
    with tasks_lock:
        tasks_storage[task_id] = task
        stop_flags[task_id] = False
    return task


def is_task_stopped(task_id: str) -> bool:
    with tasks_lock:
        return stop_flags.get(task_id, False)


def stop_task(task_id: str) -> bool:
    with tasks_lock:
        if task_id in stop_flags:
            stop_flags[task_id] = True
            return True
        return False


def update_task(task_id: str, **kwargs) -> None:
    with tasks_lock:
        if task_id in tasks_storage:
            task = tasks_storage[task_id]
            for key, value in kwargs.items():
                if hasattr(task, key):
                    setattr(task, key, value)
            if task.status == "running" and task.started_at is None:
                task.started_at = datetime.now().isoformat()


def get_task(task_id: str) -> TaskStatus:
    with tasks_lock:
        return tasks_storage.get(task_id)


class OutputCapture:
    def __init__(self, mirror=None):
        self.outputs = []
        self.mirror = mirror
        
    def write(self, text):
        if self.mirror is not None:
            self.mirror.write(text)
        if text.strip():
            self.outputs.append(str(text))
            
    def flush(self):
        if self.mirror is not None:
            self.mirror.flush()
        
    def get_output(self):
        return "\n".join(self.outputs)


def _safe_float(value, default=0.0):
    try:
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def _normalize_single_factor_ic_stats(stats: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if stats is None:
        return None

    out = dict(stats)

    ic_mean = out.get("ic_mean", out.get("mean_ic", 0.0))
    icir = out.get("icir", 0.0)

    ic_series = out.get("ic_series", None)
    if ic_series is None:
        series = np.asarray([], dtype=np.float64)
    else:
        series = np.asarray(ic_series, dtype=np.float64).reshape(-1)
        series = series[np.isfinite(series)]

    ic_std = out.get("ic_std", None)
    if ic_std is None:
        ic_std = float(np.std(series)) if series.size > 0 else 0.0

    ic_pos_ratio = out.get("ic_pos_ratio", None)
    if ic_pos_ratio is None:
        if series.size > 0:
            ic_pos_ratio = float(np.mean(series > 0))
        else:
            ic_pos_ratio = 1.0 if _safe_float(ic_mean) > 0 else 0.0

    ric_mean = out.get("ric_mean", out.get("mean_ic", ic_mean))
    ricir = out.get("ricir", icir)

    out["ic_mean"] = _safe_float(ic_mean)
    out["ic_std"] = _safe_float(ic_std)
    out["icir"] = _safe_float(icir)
    out["ic_pos_ratio"] = _safe_float(ic_pos_ratio)
    out["ric_mean"] = _safe_float(ric_mean)
    out["ricir"] = _safe_float(ricir)
    return out


def _normalize_multi_factor_ic_stats(stats: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if stats is None:
        return None

    out = dict(stats)

    ic_mean = out.get("ic_mean", out.get("mean_ic", 0.0))
    ic_series = out.get("ic_series", None)
    if ic_series is None:
        series = np.asarray([], dtype=np.float64)
    else:
        series = np.asarray(ic_series, dtype=np.float64).reshape(-1)
        series = series[np.isfinite(series)]

    ic_std = out.get("ic_std", None)
    if ic_std is None:
        ic_std = float(np.std(series)) if series.size > 0 else 0.0

    icir = out.get("icir", None)
    if icir is None:
        icir = float(ic_mean / (ic_std + 1e-8))

    out["ic_mean"] = _safe_float(ic_mean)
    out["ic_std"] = _safe_float(ic_std)
    out["icir"] = _safe_float(icir)
    return out


def _configure_plot_style(plt_module) -> None:
    plt_module.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    plt_module.rcParams["axes.unicode_minus"] = False
    plt_module.rcParams["font.family"] = "sans-serif"


def _save_chart_and_encode(fig, task_id: str, filename: str) -> tuple[str, str]:
    chart_dir = os.path.join(OUTPUT_DIR, task_id, "charts")
    os.makedirs(chart_dir, exist_ok=True)
    chart_path = os.path.join(chart_dir, filename)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(chart_path, format="png", dpi=150, bbox_inches="tight")
    with open(chart_path, "rb") as f:
        chart_base64 = base64.b64encode(f.read()).decode("ascii")
    return chart_base64, chart_path


def _save_figure_to_file(fig, output_dir: str, filename: str, dpi: int = 150, bbox_inches: Optional[str] = None) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    save_kwargs: Dict[str, Any] = {"dpi": dpi}
    if bbox_inches is not None:
        save_kwargs["bbox_inches"] = bbox_inches
    fig.savefig(path, **save_kwargs)
    return path


def _chart_priority(filename: str) -> tuple[int, str]:
    name = filename.lower()
    if name.startswith("因子") and ("rolling_ic" in name or "滚动ic" in name):
        return (0, filename)
    if name.startswith("因子") and "多空累计收益" in filename:
        return (1, filename)
    if name.startswith("lightgbm_"):
        return (2, filename)
    if name.startswith("elasticnet_"):
        return (3, filename)
    if name.startswith("instashap_"):
        return (4, filename)
    if name.startswith("多因子_") or name.startswith("综合"):
        return (5, filename)
    return (6, filename)


def _build_chart_item(image_path: str) -> Dict[str, Any]:
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("ascii")
    image_name = Path(image_path).stem
    abs_path = os.path.abspath(image_path)
    return {
        "name": image_name,
        "image": image_b64,
        "image_data_uri": f"data:image/png;base64,{image_b64}",
        "source_path": abs_path,
    }


def _collect_multi_factor_charts(task_id: str, preferred_paths: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    preferred_paths = preferred_paths or []
    chart_dirs = [os.path.join(OUTPUT_DIR, task_id, "charts")]

    unique_files: List[str] = []
    seen_content_hash: set[str] = set()

    def _try_add(path_str: str) -> None:
        if not path_str:
            return
        p = Path(path_str)
        if not p.exists() or not p.is_file():
            return

        file_hash = hashlib.sha1()
        with open(p, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                file_hash.update(chunk)
        key = file_hash.hexdigest()

        if key in seen_content_hash:
            return
        seen_content_hash.add(key)
        unique_files.append(str(p))

    for p in preferred_paths:
        _try_add(p)

    for d in chart_dirs:
        dp = Path(d)
        if not dp.exists() or not dp.is_dir():
            continue
        files = sorted(
            [str(p) for p in dp.glob("*.png") if p.is_file()],
            key=lambda x: _chart_priority(Path(x).name),
        )
        for p in files:
            _try_add(p)

    return [_build_chart_item(path) for path in unique_files]


def _generate_per_factor_detail_charts(
    mfa,
    plt_module,
    formulas: List[str],
    factor_arrays: List[np.ndarray],
    factor_arrays_test: List[np.ndarray],
    factor_arrays_now: List[np.ndarray],
    y: np.ndarray,
    y_test: Optional[np.ndarray],
    y_now: Optional[np.ndarray],
    pivoted: Dict[str, pd.DataFrame],
    pivoted_test: Dict[str, pd.DataFrame],
    pivoted_now: Dict[str, pd.DataFrame],
    chart_dir: str,
    quantiles: int,
) -> List[str]:
    generated: List[str] = []

    for i, expr in enumerate(formulas):
        factor_train = factor_arrays[i]
        factor_test = factor_arrays_test[i] if i < len(factor_arrays_test) else None
        factor_now = factor_arrays_now[i] if i < len(factor_arrays_now) else None

        analysis_train = mfa.panel_to_long_factor_df(factor_train, y, pivoted, "train")
        analysis_test = None
        analysis_now = None
        if factor_test is not None and y_test is not None:
            analysis_test = mfa.panel_to_long_factor_df(factor_test, y_test, pivoted_test, "test")
        if factor_now is not None and y_now is not None:
            analysis_now = mfa.panel_to_long_factor_df(factor_now, y_now, pivoted_now, "now")

        fig_ic = plt_module.figure(figsize=(12, 4))
        for label, long_df in [("训练集", analysis_train), ("测试集", analysis_test), ("验证集", analysis_now)]:
            if long_df is None or long_df.empty:
                continue
            ic_s = mfa.ic_curve_from_long(long_df)
            if not ic_s.empty:
                ic_s.rolling(20, min_periods=5).mean().plot(label=f"{label} 滚动IC(20)")
        plt_module.axhline(0.0, linestyle="--")
        plt_module.title(f"因子{i + 1} 滚动IC\n{expr}")
        plt_module.legend()
        plt_module.tight_layout()
        ic_path = _save_figure_to_file(fig_ic, chart_dir, f"因子{i + 1}_滚动IC.png", dpi=150)
        plt_module.close(fig_ic)
        generated.append(ic_path)

        fig_ls = plt_module.figure(figsize=(12, 4))
        for label, long_df in [("训练集", analysis_train), ("测试集", analysis_test), ("验证集", analysis_now)]:
            if long_df is None or long_df.empty:
                continue
            _, ls = mfa.build_quantile_report(long_df, quantiles=quantiles)
            if not ls.empty:
                ls.fillna(0).cumsum().plot(label=f"{label} 多空累计收益")
        plt_module.axhline(0.0, linestyle="--")
        plt_module.title(f"因子{i + 1} 多空累计收益\n{expr}")
        plt_module.legend()
        plt_module.tight_layout()
        ls_path = _save_figure_to_file(fig_ls, chart_dir, f"因子{i + 1}_多空累计收益.png", dpi=150)
        plt_module.close(fig_ls)
        generated.append(ls_path)

    return generated


def _cross_sectional_standardize(X: np.ndarray) -> np.ndarray:
    X_arr = np.asarray(X, dtype=np.float64)
    X_std = np.full_like(X_arr, np.nan, dtype=np.float64)
    for t in range(X_arr.shape[0]):
        for k in range(X_arr.shape[2]):
            col = X_arr[t, :, k]
            mask = np.isfinite(col)
            if int(mask.sum()) < 2:
                continue
            mean = float(col[mask].mean())
            std = float(col[mask].std())
            if std > 1e-8:
                X_std[t, mask, k] = (col[mask] - mean) / std
            else:
                X_std[t, mask, k] = 0.0
    return X_std


def _fill_nan_with_zero(X: np.ndarray) -> np.ndarray:
    X_filled = np.asarray(X, dtype=np.float64).copy()
    X_filled[np.isnan(X_filled)] = 0.0
    return X_filled


def _resolve_time_asset_axes(pivoted_dict: Dict[str, pd.DataFrame]) -> tuple[Optional[pd.Index], Optional[pd.Index]]:
    if not pivoted_dict:
        return None, None
    if "close" in pivoted_dict:
        base_df = pivoted_dict["close"]
    else:
        base_df = next(iter(pivoted_dict.values()))
    return base_df.index, base_df.columns


def _flatten_panel(
    X: Optional[np.ndarray],
    y_arr: Optional[np.ndarray],
    times: Optional[pd.Index],
    assets: Optional[pd.Index],
) -> Optional[pd.DataFrame]:
    if X is None or y_arr is None or times is None or assets is None:
        return None
    if X.ndim != 3 or y_arr.ndim != 2:
        return None

    t_max = min(X.shape[0], y_arr.shape[0], len(times))
    n_max = min(X.shape[1], y_arr.shape[1], len(assets))
    if t_max <= 0 or n_max <= 0:
        return None

    X_crop = X[:t_max, :n_max, :]
    y_crop = y_arr[:t_max, :n_max]
    times_crop = times[:t_max]
    assets_crop = assets[:n_max]

    records: List[Dict[str, Any]] = []
    k_size = int(X_crop.shape[2])
    for t in range(t_max):
        for n in range(n_max):
            target_val = y_crop[t, n]
            if not np.isfinite(target_val):
                continue
            row: Dict[str, Any] = {
                "time": times_crop[t],
                "asset": assets_crop[n],
                "target": float(target_val),
            }
            for k in range(k_size):
                row[f"factor_{k}"] = float(X_crop[t, n, k])
            records.append(row)

    if not records:
        return None
    return pd.DataFrame(records)


def _predictions_to_panel(
    df_pred: Optional[pd.DataFrame],
    times: Optional[pd.Index],
    assets: Optional[pd.Index],
    target_shape: Optional[tuple[int, int]],
) -> Optional[np.ndarray]:
    if df_pred is None or df_pred.empty or times is None or assets is None or target_shape is None:
        return None

    panel = np.full(target_shape, np.nan, dtype=np.float64)
    time_to_idx = {t: i for i, t in enumerate(times)}
    asset_to_idx = {a: j for j, a in enumerate(assets)}

    for _, row in df_pred.iterrows():
        i = time_to_idx.get(row["time"])
        j = asset_to_idx.get(row["asset"])
        if i is None or j is None:
            continue
        if i >= panel.shape[0] or j >= panel.shape[1]:
            continue
        panel[i, j] = float(row["pred"])
    return panel


def _compute_ic_ls_from_pred_panel(
    mfa,
    pred_panel: Optional[np.ndarray],
    y_arr: Optional[np.ndarray],
    pivoted_dict: Dict[str, pd.DataFrame],
    quantiles: int,
) -> tuple[pd.Series, pd.Series]:
    empty = pd.Series(dtype=np.float64)
    if pred_panel is None or y_arr is None or not pivoted_dict:
        return empty, empty
    try:
        long_df = mfa.panel_to_long_factor_df(pred_panel, y_arr, pivoted_dict, "pred")
    except Exception:
        return empty, empty
    if long_df is None or long_df.empty:
        return empty, empty

    try:
        ic_s = mfa.ic_curve_from_long(long_df)
    except Exception:
        ic_s = empty
    try:
        _, ls = mfa.build_quantile_report(long_df, quantiles=quantiles)
    except Exception:
        ls = empty

    if ic_s is None:
        ic_s = empty
    if ls is None:
        ls = empty
    return ic_s, ls


def _generate_model_charts(
    mfa,
    plt_module,
    formulas: List[str],
    factor_arrays_train: List[np.ndarray],
    factor_arrays_test: List[np.ndarray],
    factor_arrays_now: List[np.ndarray],
    y: np.ndarray,
    y_test: Optional[np.ndarray],
    y_now: Optional[np.ndarray],
    pivoted: Dict[str, pd.DataFrame],
    pivoted_test: Dict[str, pd.DataFrame],
    pivoted_now: Dict[str, pd.DataFrame],
    chart_dir: str,
    quantiles: int,
    use_lightgbm: bool,
    use_elastic_net: bool,
) -> List[str]:
    generated: List[str] = []
    if not factor_arrays_train:
        return generated

    K = len(factor_arrays_train)
    X_train_raw = np.stack([np.asarray(arr, dtype=np.float64) for arr in factor_arrays_train], axis=2)

    X_test_raw: Optional[np.ndarray] = None
    if factor_arrays_test and len(factor_arrays_test) >= K:
        X_test_raw = np.stack([np.asarray(arr, dtype=np.float64) for arr in factor_arrays_test[:K]], axis=2)

    X_now_raw: Optional[np.ndarray] = None
    if factor_arrays_now and len(factor_arrays_now) >= K:
        X_now_raw = np.stack([np.asarray(arr, dtype=np.float64) for arr in factor_arrays_now[:K]], axis=2)

    X_train = _fill_nan_with_zero(_cross_sectional_standardize(X_train_raw))
    X_test = _fill_nan_with_zero(_cross_sectional_standardize(X_test_raw)) if X_test_raw is not None else None
    X_now = _fill_nan_with_zero(_cross_sectional_standardize(X_now_raw)) if X_now_raw is not None else None

    times_train, assets_train = _resolve_time_asset_axes(pivoted)
    times_test, assets_test = _resolve_time_asset_axes(pivoted_test)
    times_now, assets_now = _resolve_time_asset_axes(pivoted_now)

    df_train = _flatten_panel(X_train, y, times_train, assets_train)
    df_test = _flatten_panel(X_test, y_test, times_test, assets_test)
    df_now = _flatten_panel(X_now, y_now, times_now, assets_now)
    if df_train is None or df_train.empty:
        return generated

    feature_cols = [f"factor_{k}" for k in range(K)]
    split_colors = {"训练集": "#1f77b4", "测试集": "#ff7f0e", "验证集": "#2ca02c"}

    def _append_prediction_charts(prefix: str, df_train_pred: pd.DataFrame, df_test_pred: Optional[pd.DataFrame], df_now_pred: Optional[pd.DataFrame]) -> None:
        nonlocal generated
        pred_panel_train = _predictions_to_panel(df_train_pred, times_train, assets_train, y.shape if y is not None else None)
        pred_panel_test = _predictions_to_panel(df_test_pred, times_test, assets_test, y_test.shape if y_test is not None else None)
        pred_panel_now = _predictions_to_panel(df_now_pred, times_now, assets_now, y_now.shape if y_now is not None else None)

        split_stats = [
            ("训练集", *_compute_ic_ls_from_pred_panel(mfa, pred_panel_train, y, pivoted, quantiles)),
            ("测试集", *_compute_ic_ls_from_pred_panel(mfa, pred_panel_test, y_test, pivoted_test, quantiles)),
            ("验证集", *_compute_ic_ls_from_pred_panel(mfa, pred_panel_now, y_now, pivoted_now, quantiles)),
        ]

        fig_ic = plt_module.figure(figsize=(12, 4))
        plotted_ic = 0
        for label, ic_s, _ in split_stats:
            if ic_s is None or ic_s.empty:
                continue
            ic_s.rolling(20, min_periods=5).mean().plot(label=f"{label} 滚动IC(20)", color=split_colors.get(label))
            plotted_ic += 1
        plt_module.axhline(0.0, linestyle="--", color="black", alpha=0.6)
        plt_module.title(f"{prefix} 合成因子 滚动IC")
        if plotted_ic > 0:
            plt_module.legend()
        plt_module.tight_layout()
        rolling_name = f"{prefix}_滚动IC.png"
        rolling_path = _save_figure_to_file(fig_ic, chart_dir, rolling_name, dpi=150, bbox_inches="tight")
        plt_module.close(fig_ic)
        generated.append(rolling_path)

        fig_ls = plt_module.figure(figsize=(12, 4))
        plotted_ls = 0
        for label, _, ls in split_stats:
            if ls is None or ls.empty:
                continue
            ls.fillna(0).cumsum().plot(label=f"{label} 多空累计收益", color=split_colors.get(label))
            plotted_ls += 1
        plt_module.axhline(0.0, linestyle="--", color="black", alpha=0.6)
        plt_module.title(f"{prefix} 合成因子 Top-Bottom 多空累计收益")
        if plotted_ls > 0:
            plt_module.legend()
        plt_module.tight_layout()
        ls_name = f"{prefix}_多空累计收益.png"
        ls_path = _save_figure_to_file(fig_ls, chart_dir, ls_name, dpi=150, bbox_inches="tight")
        plt_module.close(fig_ls)
        generated.append(ls_path)

        if df_now_pred is not None and not df_now_pred.empty:
            fig_scatter = plt_module.figure(figsize=(6, 6))
            sample = df_now_pred
            if len(sample) > 5000:
                sample = sample.sample(n=5000, random_state=42)
            plt_module.scatter(sample["pred"], sample["target"], alpha=0.3, s=4)
            plt_module.xlabel("预测值")
            plt_module.ylabel("真实未来收益")
            plt_module.title("验证集：预测 vs 真实")
            plt_module.grid(True, alpha=0.3)
            plt_module.tight_layout()
            scatter_name = f"{prefix}_预测vs真实.png"
            scatter_path = _save_figure_to_file(fig_scatter, chart_dir, scatter_name, dpi=150, bbox_inches="tight")
            plt_module.close(fig_scatter)
            generated.append(scatter_path)

    if use_lightgbm:
        try:
            import lightgbm as lgb

            lgb_params = {
                "n_estimators": 200,
                "learning_rate": 0.05,
                "max_depth": 6,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 0.1,
                "min_child_samples": 20,
                "random_state": 42,
                "verbosity": -1,
            }
            lgb_model = lgb.LGBMRegressor(**lgb_params)
            lgb_model.fit(df_train[feature_cols], df_train["target"])

            df_train_lgb = df_train.copy()
            df_train_lgb["pred"] = lgb_model.predict(df_train_lgb[feature_cols])
            df_test_lgb = None
            if df_test is not None and not df_test.empty:
                df_test_lgb = df_test.copy()
                df_test_lgb["pred"] = lgb_model.predict(df_test_lgb[feature_cols])
            df_now_lgb = None
            if df_now is not None and not df_now.empty:
                df_now_lgb = df_now.copy()
                df_now_lgb["pred"] = lgb_model.predict(df_now_lgb[feature_cols])

            _append_prediction_charts("LightGBM", df_train_lgb, df_test_lgb, df_now_lgb)

            fig_imp = plt_module.figure(figsize=(10, 6))
            importances = np.asarray(lgb_model.feature_importances_, dtype=np.float64)
            order = np.argsort(importances)[::-1]
            labels = [formulas[i] if i < len(formulas) else f"因子{i + 1}" for i in order]
            vals = importances[order]
            plt_module.barh(range(len(labels)), vals[::-1], color="#3b82f6")
            plt_module.yticks(range(len(labels)), labels[::-1])
            plt_module.xlabel("特征重要性")
            plt_module.title("LightGBM 特征重要性")
            plt_module.tight_layout()
            imp_path = _save_figure_to_file(fig_imp, chart_dir, "LightGBM_特征重要性.png", dpi=150)
            plt_module.close(fig_imp)
            generated.append(imp_path)

            # 残差分析图（优先使用验证集，无验证集时回退测试集/训练集）
            residual_df = None
            residual_label = ""
            if df_now_lgb is not None and not df_now_lgb.empty:
                residual_df = df_now_lgb
                residual_label = "验证集"
            elif df_test_lgb is not None and not df_test_lgb.empty:
                residual_df = df_test_lgb
                residual_label = "测试集"
            elif df_train_lgb is not None and not df_train_lgb.empty:
                residual_df = df_train_lgb
                residual_label = "训练集"

            if residual_df is not None:
                residuals = (residual_df["target"] - residual_df["pred"]).replace([np.inf, -np.inf], np.nan).dropna()
                if not residuals.empty:
                    fig_res, axes_res = plt_module.subplots(1, 2, figsize=(12, 4))

                    # 左图：残差分布
                    axes_res[0].hist(residuals.values, bins=50, color="#60a5fa", edgecolor="black", alpha=0.75)
                    axes_res[0].axvline(0.0, color="red", linestyle="--", alpha=0.7)
                    axes_res[0].set_title(f"残差分布（{residual_label}）")
                    axes_res[0].set_xlabel("残差")
                    axes_res[0].set_ylabel("频数")

                    # 右图：残差 vs 预测值
                    scatter_df = residual_df[["pred", "target"]].replace([np.inf, -np.inf], np.nan).dropna()
                    if len(scatter_df) > 5000:
                        scatter_df = scatter_df.sample(n=5000, random_state=42)
                    res_vals = scatter_df["target"] - scatter_df["pred"]
                    axes_res[1].scatter(scatter_df["pred"], res_vals, s=6, alpha=0.35, color="#1d4ed8")
                    axes_res[1].axhline(0.0, color="red", linestyle="--", alpha=0.7)
                    axes_res[1].set_title(f"残差 vs 预测值（{residual_label}）")
                    axes_res[1].set_xlabel("预测值")
                    axes_res[1].set_ylabel("残差")

                    fig_res.suptitle("LightGBM 残差分析", fontsize=12, fontweight="bold")
                    fig_res.tight_layout(rect=[0, 0, 1, 0.95])
                    residual_path = _save_figure_to_file(fig_res, chart_dir, "LightGBM_残差分析.png", dpi=150)
                    plt_module.close(fig_res)
                    generated.append(residual_path)
        except Exception as exc:
            print(f"[Multi][Charts] LightGBM 图表生成失败: {exc}")

    if use_elastic_net:
        try:
            from sklearn.linear_model import ElasticNet

            enet_params = {
                "alpha": 0.05,
                "l1_ratio": 0.5,
                "fit_intercept": True,
                "max_iter": 5000,
                "random_state": 42,
                "selection": "cyclic",
            }
            enet_model = ElasticNet(**enet_params)
            enet_model.fit(df_train[feature_cols], df_train["target"])

            df_train_enet = df_train.copy()
            df_train_enet["pred"] = enet_model.predict(df_train_enet[feature_cols])
            df_test_enet = None
            if df_test is not None and not df_test.empty:
                df_test_enet = df_test.copy()
                df_test_enet["pred"] = enet_model.predict(df_test_enet[feature_cols])
            df_now_enet = None
            if df_now is not None and not df_now.empty:
                df_now_enet = df_now.copy()
                df_now_enet["pred"] = enet_model.predict(df_now_enet[feature_cols])

            _append_prediction_charts("ElasticNet", df_train_enet, df_test_enet, df_now_enet)

            fig_coef = plt_module.figure(figsize=(10, 6))
            coef = np.asarray(enet_model.coef_, dtype=np.float64)
            coef_labels = [formulas[i] if i < len(formulas) else f"因子{i + 1}" for i in range(len(coef))]
            coef_series = pd.Series(coef, index=coef_labels)
            coef_series = coef_series.reindex(coef_series.abs().sort_values(ascending=False).index)
            colors = ["#16a34a" if v >= 0 else "#dc2626" for v in coef_series.values]
            plt_module.barh(range(len(coef_series)), coef_series.values, color=colors, alpha=0.8)
            plt_module.yticks(range(len(coef_series)), coef_series.index)
            plt_module.axvline(0.0, color="black", linestyle="--", alpha=0.6)
            plt_module.xlabel("系数值")
            plt_module.title("Elastic Net 系数条形图")
            plt_module.tight_layout()
            coef_path = _save_figure_to_file(fig_coef, chart_dir, "ElasticNet_系数条形图.png", dpi=150)
            plt_module.close(fig_coef)
            generated.append(coef_path)
        except Exception as exc:
            print(f"[Multi][Charts] ElasticNet 图表生成失败: {exc}")

    return generated


def _build_factor_corr_matrix(
    factor_arrays: List[np.ndarray],
    factor_names: List[str],
) -> pd.DataFrame:
    if not factor_arrays:
        return pd.DataFrame([[1.0]], index=["因子1"], columns=["因子1"], dtype=np.float64)

    flattened: Dict[str, np.ndarray] = {}
    for idx, factor_arr in enumerate(factor_arrays):
        name = factor_names[idx] if idx < len(factor_names) else f"因子{idx + 1}"
        arr = np.asarray(factor_arr, dtype=np.float64)
        flattened[name] = arr.reshape(-1)

    lengths = [v.size for v in flattened.values()]
    if not lengths or min(lengths) == 0:
        names = list(flattened.keys())[:1] or ["因子1"]
        return pd.DataFrame([[1.0]], index=names, columns=names, dtype=np.float64)

    if len(set(lengths)) != 1:
        min_len = min(lengths)
        flattened = {k: v[:min_len] for k, v in flattened.items()}

    corr_df = pd.DataFrame(flattened).corr(method="pearson", min_periods=2)
    if corr_df.empty:
        names = list(flattened.keys())[:1] or ["因子1"]
        return pd.DataFrame([[1.0]], index=names, columns=names, dtype=np.float64)

    for col in corr_df.columns:
        corr_df.loc[col, col] = 1.0
    return corr_df


def _select_factor_corr_split(
    factor_names: List[str],
    factor_arrays_train: List[np.ndarray],
    factor_arrays_test: List[np.ndarray],
    factor_arrays_now: List[np.ndarray],
) -> tuple[str, pd.DataFrame]:
    candidates = [
        ("验证集", factor_arrays_now),
        ("测试集", factor_arrays_test),
        ("训练集", factor_arrays_train),
    ]
    for split_name, split_factors in candidates:
        if not split_factors:
            continue
        corr_df = _build_factor_corr_matrix(split_factors, factor_names)
        finite_count = int(np.isfinite(corr_df.to_numpy(dtype=np.float64)).sum())
        if finite_count > 0:
            return split_name, corr_df

    return "训练集", _build_factor_corr_matrix(factor_arrays_train, factor_names)


def run_single_factor_task(task_id: str, params: SingleFactorParams):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        _configure_plot_style(plt)

        use_mock = params.use_mock_data if hasattr(params, 'use_mock_data') else False
        sfa = None

        if not use_mock:
            try:
                import single_factor_analysis as sfa
                from single_factor_analysis import (
                    get_symbols_by_sector as sfa_get_symbols,
                    get_futures_data as sfa_get_data,
                    FUTURES_SECTORS as SFA_FUTURES_SECTORS,
                )

                if not getattr(sfa, 'QMF_DATA_AVAILABLE', True):
                    raise ValueError("qmf_data 不可用")

                if params.symbols:
                    SYMBOLS = params.symbols
                else:
                    SYMBOLS = sfa_get_symbols(params.selected_sector, SFA_FUTURES_SECTORS, None)

                features = ["open", "close", "high", "low", "volume", "open_interest"]

                df_list = []
                for symbol in SYMBOLS:
                    data = sfa_get_data(symbol, params.begin_time, params.end_time, params.symbol_cycle)
                    if len(data) > 0:
                        df_list.append(data)

                if len(df_list) == 0:
                    raise ValueError("训练集没有有效数据")

                df = pd.concat(df_list, ignore_index=True)

                df_list_test = []
                for symbol in SYMBOLS:
                    data = sfa_get_data(symbol, params.begin_time_test, params.end_time_test, params.symbol_cycle)
                    if len(data) > 0:
                        df_list_test.append(data)

                if len(df_list_test) == 0:
                    raise ValueError("测试集没有有效数据")

                df_test = pd.concat(df_list_test, ignore_index=True)

                df_list_now = []
                if params.begin_time_now:
                    validation_end_time = getattr(sfa, "END_TIME_NOW", datetime.now().strftime("%Y-%m-%d"))
                    for symbol in SYMBOLS:
                        data = sfa_get_data(symbol, params.begin_time_now, validation_end_time, params.symbol_cycle)
                        if len(data) > 0:
                            df_list_now.append(data)
                
                df_now = pd.concat(df_list_now, ignore_index=True) if df_list_now else pd.DataFrame()
                use_mock = False
            except Exception as e:
                print(f"警告: 无法导入单因子分析模块或获取数据: {e}")
                print("自动切换到 Mock 模式进行演示")
                use_mock = True
        
        if use_mock:
            print("使用模拟数据进行演示")
            SYMBOLS = params.symbols if params.symbols else ["au888", "ag888", "cu888"]
            features = ["open", "close", "high", "low", "volume", "open_interest"]
            
            df_list, _ = generate_mock_df_list(SYMBOLS, n_timepoints=500, start_date=params.begin_time)
            df_list_test, _ = generate_mock_df_list(SYMBOLS, n_timepoints=200, start_date=params.begin_time_test)
            df_list_now, _ = generate_mock_df_list(SYMBOLS, n_timepoints=100, start_date=params.begin_time_now or "2026-01-01")
            
            df = pd.concat(df_list, ignore_index=True)
            df_test = pd.concat(df_list_test, ignore_index=True)
            df_now = pd.concat(df_list_now, ignore_index=True) if df_list_now else pd.DataFrame()
        
        start_time = time.time()
        update_task(task_id, status="running", message="正在初始化...")
        
        old_stdout = sys.stdout
        capture = OutputCapture(mirror=old_stdout)
        sys.stdout = capture
        
        update_task(task_id, message="正在获取合约列表...", progress=10)
        
        print(f"当前使用的合约列表（共 {len(SYMBOLS)} 个合约）:")
        print(SYMBOLS)
        
        update_task(task_id, message="正在预处理数据...", progress=20)
        
        for col in features:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        df.drop_duplicates(subset=["asset", "date"], keep="last", inplace=True)
        df.sort_values(["asset", "date"], ignore_index=True, inplace=True)
        
        print(f"训练集数据预处理完成，共 {len(df)} 条记录")
        
        df = df.sort_values(["asset", "date"]).reset_index(drop=True)
        df["future_return"] = df.groupby("asset")["close"].shift(-params.y_period) / df["close"] - 1
        df = df.dropna(subset=["future_return"]).reset_index(drop=True)
        
        df["time"] = df["date"]
        df["code"] = df["asset"]
        
        target = "future_return"
        pivoted = {}
        for col in features + [target]:
            try:
                pivoted[col] = df.pivot(index="time", columns="code", values=col)
            except Exception as e:
                print(f"特征 {col} 转换失败: {e}")
                unique_dates = pd.Series(df["time"]).unique()
                unique_codes = pd.Series(df["code"]).unique()
                pivoted[col] = pd.DataFrame(0, index=unique_dates, columns=unique_codes)
        
        X_dict = {f: pivoted[f].values for f in features}
        y = pivoted[target].values
        
        print(f"训练数据准备完成: 时间点={y.shape[0]}, 合约数={y.shape[1]}")
        
        update_task(task_id, message="正在预处理测试数据...", progress=30)
        
        for col in features:
            df_test[col] = pd.to_numeric(df_test[col], errors="coerce")
        
        df_test.drop_duplicates(subset=["asset", "date"], keep="last", inplace=True)
        df_test.sort_values(["asset", "date"], ignore_index=True, inplace=True)
        
        print(f"测试集数据预处理完成，共 {len(df_test)} 条记录")
        
        df_test = df_test.sort_values(["asset", "date"]).reset_index(drop=True)
        df_test["future_return"] = df_test.groupby("asset")["close"].shift(-params.y_period) / df_test["close"] - 1
        df_test = df_test.dropna(subset=["future_return"]).reset_index(drop=True)
        
        df_test["time"] = df_test["date"]
        df_test["code"] = df_test["asset"]
        
        pivoted_test = {}
        for col in features + [target]:
            try:
                pivoted_test[col] = df_test.pivot(index="time", columns="code", values=col)
            except:
                unique_dates = pd.Series(df_test["time"]).unique()
                unique_codes = pd.Series(df_test["code"]).unique()
                pivoted_test[col] = pd.DataFrame(0, index=unique_dates, columns=unique_codes)
        
        X_dict_test = {f: pivoted_test[f].values for f in features}
        y_test = pivoted_test[target].values
        
        print(f"测试数据准备完成: 时间点={y_test.shape[0]}, 合约数={y_test.shape[1]}")
        
        update_task(task_id, message="正在加载验证集数据...", progress=40)
        
        X_dict_now = None
        y_now = None
        if params.begin_time_now and not df_now.empty:
            df_now_processed = df_now.copy()
            for col in features:
                df_now_processed[col] = pd.to_numeric(df_now_processed[col], errors="coerce")
            df_now_processed.drop_duplicates(subset=["asset", "date"], keep="last", inplace=True)
            df_now_processed.sort_values(["asset", "date"], ignore_index=True, inplace=True)
            
            df_now_processed = df_now_processed.sort_values(["asset", "date"]).reset_index(drop=True)
            df_now_processed["future_return"] = df_now_processed.groupby("asset")["close"].shift(-params.y_period) / df_now_processed["close"] - 1
            df_now_processed = df_now_processed.dropna(subset=["future_return"]).reset_index(drop=True)
            
            df_now_processed["time"] = df_now_processed["date"]
            df_now_processed["code"] = df_now_processed["asset"]
            
            pivoted_now = {}
            for col in features + [target]:
                try:
                    pivoted_now[col] = df_now_processed.pivot(index="time", columns="code", values=col)
                except:
                    unique_dates = pd.Series(df_now_processed["time"]).unique()
                    unique_codes = pd.Series(df_now_processed["code"]).unique()
                    pivoted_now[col] = pd.DataFrame(0, index=unique_dates, columns=unique_codes)
            
            X_dict_now = {f: pivoted_now[f].values for f in features}
            y_now = pivoted_now[target].values
        
        update_task(task_id, message="正在评估因子...", progress=50)
        
        print(f"\n因子表达式: {params.formula}")
        print(f"品种合约: {SYMBOLS}")
        print(f"预测周期: {params.y_period}")
        
        if use_mock:
            factor = np.random.randn(y.shape[0], y.shape[1])
            factor_test = np.random.randn(y_test.shape[0], y_test.shape[1]) if y_test is not None else None
            factor_now = np.random.randn(y_now.shape[0], y_now.shape[1]) if y_now is not None else None
            
            analysis_train = pd.DataFrame({
                "factor": factor.flatten(),
                "return": y.flatten(),
                "dataset": "训练集"
            }).dropna()
            
            analysis_test = None
            if y_test is not None:
                analysis_test = pd.DataFrame({
                    "factor": factor_test.flatten(),
                    "return": y_test.flatten(),
                    "dataset": "测试集"
                }).dropna()
            
            analysis_now = None
            if y_now is not None:
                analysis_now = pd.DataFrame({
                    "factor": factor_now.flatten(),
                    "return": y_now.flatten(),
                    "dataset": "验证集"
                }).dropna()
            
            ic_train = {"ic_mean": np.random.uniform(-0.05, 0.05), "ic_std": 0.1, "icir": np.random.uniform(-0.5, 0.5), "ic_pos_ratio": 0.5, "ric_mean": np.random.uniform(-0.05, 0.05), "ricir": np.random.uniform(-0.5, 0.5)}
            ic_test = {"ic_mean": np.random.uniform(-0.05, 0.05), "ic_std": 0.1, "icir": np.random.uniform(-0.5, 0.5), "ic_pos_ratio": 0.5, "ric_mean": np.random.uniform(-0.05, 0.05), "ricir": np.random.uniform(-0.5, 0.5)} if analysis_test is not None else None
            ic_now = {"ic_mean": np.random.uniform(-0.05, 0.05), "ic_std": 0.1, "icir": np.random.uniform(-0.5, 0.5), "ic_pos_ratio": 0.5, "ric_mean": np.random.uniform(-0.05, 0.05), "ricir": np.random.uniform(-0.5, 0.5)} if analysis_now is not None else None
        else:
            my_cls = sfa.My
            exprs = sfa.eval_factors(params.formula, my_cls, X_dict)
            
            factor = exprs["factor"]
            
            analysis_train = sfa.panel_to_long_factor_df(factor, y, pivoted, "训练集")
            
            analysis_test = None
            if X_dict_test is not None:
                exprs_test = sfa.eval_factors(params.formula, my_cls, X_dict_test)
                factor_test = exprs_test["factor"]
                analysis_test = sfa.panel_to_long_factor_df(factor_test, y_test, pivoted_test, "测试集")
            
            analysis_now = None
            if X_dict_now is not None:
                exprs_now = sfa.eval_factors(params.formula, my_cls, X_dict_now)
                factor_now = exprs_now["factor"]
                analysis_now = sfa.panel_to_long_factor_df(factor_now, y_now, pivoted_now, "验证集")
            
            ic_train = sfa.calc_ic_stats(factor, y)
            ic_test = sfa.calc_ic_stats(factor_test, y_test) if analysis_test is not None else None
            ic_now = sfa.calc_ic_stats(factor_now, y_now) if analysis_now is not None else None
        
        ic_train = _normalize_single_factor_ic_stats(ic_train)
        ic_test = _normalize_single_factor_ic_stats(ic_test)
        ic_now = _normalize_single_factor_ic_stats(ic_now)

        summary_data = []
        if ic_train is not None:
            summary_data.append({
                "数据集": "训练集",
                "IC均值": ic_train.get("ic_mean", 0),
                "IC标准差": ic_train.get("ic_std", 0),
                "ICIR": ic_train.get("icir", 0),
                "IC>0比例": ic_train.get("ic_pos_ratio", 0),
                "Rank IC均值": ic_train.get("ric_mean", 0),
                "Rank ICIR": ic_train.get("ricir", 0),
            })
        
        if ic_test is not None:
            summary_data.append({
                "数据集": "测试集",
                "IC均值": ic_test.get("ic_mean", 0),
                "IC标准差": ic_test.get("ic_std", 0),
                "ICIR": ic_test.get("icir", 0),
                "IC>0比例": ic_test.get("ic_pos_ratio", 0),
                "Rank IC均值": ic_test.get("ric_mean", 0),
                "Rank ICIR": ic_test.get("ricir", 0),
            })
        
        if ic_now is not None:
            summary_data.append({
                "数据集": "验证集",
                "IC均值": ic_now.get("ic_mean", 0),
                "IC标准差": ic_now.get("ic_std", 0),
                "ICIR": ic_now.get("icir", 0),
                "IC>0比例": ic_now.get("ic_pos_ratio", 0),
                "Rank IC均值": ic_now.get("ric_mean", 0),
                "Rank ICIR": ic_now.get("ricir", 0),
            })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df)
        
        update_task(task_id, message="正在生成图表...", progress=85)
        charts = []
        split_frames = [
            ("训练集", analysis_train),
            ("测试集", analysis_test),
            ("验证集", analysis_now),
        ]
        available_frames = [
            (label, df_part)
            for label, df_part in split_frames
            if df_part is not None and not df_part.empty
        ]
        split_colors = {
            "训练集": "#1f77b4",
            "测试集": "#ff7f0e",
            "验证集": "#2ca02c",
        }

        if available_frames:
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            fig.suptitle("单因子综合分析图", fontsize=14, fontweight="bold")
            plotted_ic = 0
            plotted_ls = 0
            quantile_means = {}

            for label, long_df in available_frames:
                color = split_colors.get(label, None)
                ic_s = sfa.ic_curve_from_long(long_df)
                if not ic_s.empty:
                    axes[0, 0].plot(ic_s.index, ic_s.values, label=label, color=color)
                    plotted_ic += 1

                qret, ls = sfa.build_quantile_report(long_df, quantiles=params.quantiles)
                if not qret.empty:
                    # qret: index=time, columns=quantile；这里按列求均值得到“分位数平均收益”
                    quantile_means[label] = qret.mean(axis=0)
                if not ls.empty:
                    ls_curve = ls.fillna(0).cumsum()
                    axes[1, 0].plot(ls_curve.index, ls_curve.values, label=label, color=color)
                    plotted_ls += 1

            axes[0, 0].axhline(0, color="r", linestyle="--")
            axes[0, 0].set_title("分组IC序列（训练/测试/验证）")
            if plotted_ic > 0:
                axes[0, 0].legend()

            if quantile_means:
                quantile_idx = sorted(
                    {q for series in quantile_means.values() for q in series.index}
                )
                n_group = len(quantile_means)
                x = np.arange(len(quantile_idx))
                width = 0.8 / max(n_group, 1)
                for i, (label, series) in enumerate(quantile_means.items()):
                    color = split_colors.get(label, None)
                    values = [float(series.get(q, np.nan)) for q in quantile_idx]
                    axes[0, 1].bar(x + i * width, values, width=width, label=label, color=color, alpha=0.85)
                axes[0, 1].set_xticks(x + width * (n_group - 1) / 2)
                axes[0, 1].set_xticklabels([str(q) for q in quantile_idx])
                axes[0, 1].axhline(0, color="black", linestyle="--", alpha=0.6)
                axes[0, 1].set_title("分位数平均收益（训练/测试/验证）")
                axes[0, 1].legend()
            else:
                axes[0, 1].set_title("分位数平均收益（训练/测试/验证）")

            axes[1, 0].axhline(0, color="black", linestyle="--", alpha=0.6)
            axes[1, 0].set_title("多空累计收益（训练/测试/验证）")
            if plotted_ls > 0:
                axes[1, 0].legend()

            plotted_hist = 0
            for label, long_df in available_frames:
                vals = long_df["factor"].dropna()
                if vals.empty:
                    continue
                color = split_colors.get(label, None)
                axes[1, 1].hist(vals, bins=40, alpha=0.35, label=label, color=color)
                plotted_hist += 1
            axes[1, 1].set_title("因子值分布（训练/测试/验证）")
            if plotted_hist > 0:
                axes[1, 1].legend()

            chart_base64, chart_path = _save_chart_and_encode(fig, task_id, "single_factor_combined.png")
            charts.append({
                "name": "综合分析",
                "image": chart_base64,
                "image_data_uri": f"data:image/png;base64,{chart_base64}",
                "source_path": chart_path,
            })
            plt.close(fig)

        elapsed_time = time.time() - start_time
        
        sys.stdout = old_stdout
        output_text = capture.get_output()
        
        update_task(
            task_id,
            status="completed",
            progress=100.0,
            message="分析完成",
            result={
                "formula": params.formula,
                "symbols": SYMBOLS,
                "ic_stats": summary_data,
                "charts": charts,
                "output_text": output_text,
                "elapsed_time": elapsed_time,
            },
            completed_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        import traceback
        sys.stdout = old_stdout
        output_text = capture.get_output() + f"\n错误: {str(e)}\n{traceback.format_exc()}" if 'capture' in locals() else f"错误: {str(e)}\n{traceback.format_exc()}"
        
        update_task(
            task_id,
            status="failed",
            message=f"任务失败: {str(e)}",
            result={"output_text": output_text, "error": str(e)},
            completed_at=datetime.now().isoformat()
        )


def run_multi_factor_task(task_id: str, params: MultiFactorParams):
    try:
        start_time = time.time()
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        _configure_plot_style(plt)

        def _update_multi_task(message: str, progress: Optional[float] = None, status: Optional[str] = None, **kwargs) -> None:
            payload: Dict[str, Any] = dict(kwargs)
            payload["message"] = message
            if progress is not None:
                payload["progress"] = progress
            if status is not None:
                payload["status"] = status
            payload["elapsed_time"] = time.time() - start_time
            update_task(task_id, **payload)

        use_mock = not MULTI_FACTOR_AVAILABLE

        if not use_mock:
            import multi_factor_analysis as mfa
            if not getattr(mfa, 'QMF_DATA_AVAILABLE', True):
                use_mock = True
                print("警告: qmf_data 不可用，自动切换到 Mock 模式")

        if use_mock:
            print("使用模拟数据进行演示")
            SYMBOLS = params.symbols if params.symbols else ["au888", "ag888", "cu888"]
            features = ['open', 'close', 'high', 'low', 'volume', 'open_interest']

            df_list, _ = generate_mock_df_list(SYMBOLS, n_timepoints=500, start_date=params.begin_time)
            df_list_test, _ = generate_mock_df_list(SYMBOLS, n_timepoints=200, start_date=params.begin_time_test)
            df_list_now, _ = generate_mock_df_list(SYMBOLS, n_timepoints=100, start_date=params.begin_time_now or "2026-01-01")

            df = pd.concat(df_list, ignore_index=True)
            df_test = pd.concat(df_list_test, ignore_index=True)
            df_now = pd.concat(df_list_now, ignore_index=True) if df_list_now else pd.DataFrame()

            target = 'future_return'
            formulas = params.formula if isinstance(params.formula, list) else [params.formula]
        else:
            mfa.SELECTED_SECTOR = params.selected_sector
            mfa.MANUAL_SYMBOLS = params.symbols if params.symbols else ["au888", "ag888"]
            mfa.BEGIN_TIME = params.begin_time
            mfa.END_TIME = params.end_time
            mfa.BEGIN_TIME_TEST = params.begin_time_test
            mfa.END_TIME_TEST = params.end_time_test
            mfa.BEGIN_TIME_NOW = params.begin_time_now
            mfa.END_TIME_NOW = getattr(mfa, "END_TIME_NOW", datetime.now().strftime("%Y-%m-%d"))
            mfa.SYMBOL_CYCLE = params.symbol_cycle
            mfa.Y_PERIOD = params.y_period
            mfa.USE_LIGHTGBM = params.use_lightgbm
            mfa.USE_ELASTIC_NET = params.use_elastic_net
            mfa.USE_INSTASHAP = params.use_instashap
            
            if isinstance(params.formula, list):
                mfa.formula = params.formula
            else:
                mfa.formula = [params.formula]
            
            SYMBOLS = mfa.get_symbols_by_sector(mfa.SELECTED_SECTOR, mfa.FUTURES_SECTORS, mfa.MANUAL_SYMBOLS)

            df_list = []
            for symbol in SYMBOLS:
                data = mfa.get_futures_data(symbol, mfa.BEGIN_TIME, mfa.END_TIME, mfa.SYMBOL_CYCLE)
                if len(data) > 0:
                    df_list.append(data)

            if len(df_list) == 0:
                raise ValueError("训练集没有有效数据")

            df = pd.concat(df_list, ignore_index=True)

            df_list_test = []
            for symbol in SYMBOLS:
                data = mfa.get_futures_data(symbol, mfa.BEGIN_TIME_TEST, mfa.END_TIME_TEST, mfa.SYMBOL_CYCLE)
                if len(data) > 0:
                    df_list_test.append(data)

            if len(df_list_test) == 0:
                raise ValueError("测试集没有有效数据")

            df_test = pd.concat(df_list_test, ignore_index=True)

            df_list_now = []
            if params.begin_time_now:
                validation_end_time = mfa.END_TIME_NOW
                for symbol in SYMBOLS:
                    data = mfa.get_futures_data(symbol, mfa.BEGIN_TIME_NOW, validation_end_time, mfa.SYMBOL_CYCLE)
                    if len(data) > 0:
                        df_list_now.append(data)
            
            df_now = pd.concat(df_list_now, ignore_index=True) if df_list_now else pd.DataFrame()
            
            target = 'future_return'
            features = ['open', 'close', 'high', 'low', 'volume', 'open_interest']
            formulas = mfa.formula
        
        _update_multi_task(message="正在初始化...", status="running")
        
        old_stdout = sys.stdout
        capture = OutputCapture(mirror=old_stdout)
        sys.stdout = capture
        
        print(f"当前使用的合约列表（共 {len(SYMBOLS)} 个合约）:")
        print(SYMBOLS)

        if use_mock and 'mfa' not in locals():
            raise ValueError("multi_factor_analysis 模块不可用，无法执行多因子表达式评估")

        formula_list = formulas if use_mock else mfa.formula
        y_period = params.y_period if use_mock else mfa.Y_PERIOD

        for col in features:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df.drop_duplicates(subset=["asset", "date"], keep="last", inplace=True)
        df.sort_values(["asset", "date"], ignore_index=True, inplace=True)
        print(f"训练集数据预处理完成，共 {len(df)} 条记录")

        df = df.sort_values(["asset", "date"]).reset_index(drop=True)
        df['future_return'] = df.groupby("asset")['close'].shift(-y_period) / df['close'] - 1
        df = df.dropna(subset=['future_return']).reset_index(drop=True)
        df['time'] = df['date']
        df['code'] = df['asset']

        data = df[features + [target, 'code', 'time']].copy()
        pivoted = {}
        for col in features + [target]:
            try:
                pivoted[col] = data.pivot(index='time', columns='code', values=col).sort_index().sort_index(axis=1)
            except Exception:
                unique_dates = pd.Series(data['time']).unique()
                unique_codes = pd.Series(data['code']).unique()
                pivoted[col] = pd.DataFrame(0, index=unique_dates, columns=unique_codes)

        X_dict = {f: pivoted[f].values for f in features}
        y = pivoted[target].values
        print(f"训练数据准备完成: 时间点={y.shape[0]}, 合约数={y.shape[1]}")

        _update_multi_task(message="正在加载测试数据...", progress=20)

        if df_test.empty:
            raise ValueError("测试集没有有效数据")

        for col in features:
            df_test[col] = pd.to_numeric(df_test[col], errors="coerce")
        df_test.drop_duplicates(subset=["asset", "date"], keep="last", inplace=True)
        df_test.sort_values(["asset", "date"], ignore_index=True, inplace=True)
        print(f"测试集数据预处理完成，共 {len(df_test)} 条记录")

        df_test = df_test.sort_values(["asset", "date"]).reset_index(drop=True)
        df_test['future_return'] = df_test.groupby("asset")['close'].shift(-y_period) / df_test['close'] - 1
        df_test = df_test.dropna(subset=['future_return']).reset_index(drop=True)
        df_test['time'] = df_test['date']
        df_test['code'] = df_test['asset']

        data_test = df_test[features + [target, 'code', 'time']].copy()
        pivoted_test = {}
        for col in features + [target]:
            try:
                pivoted_test[col] = data_test.pivot(index='time', columns='code', values=col).sort_index().sort_index(axis=1)
            except Exception:
                unique_dates_test = pd.Series(data_test['time']).unique()
                unique_codes_test = pd.Series(data_test['code']).unique()
                pivoted_test[col] = pd.DataFrame(0, index=unique_dates_test, columns=unique_codes_test)

        X_dict_test = {f: pivoted_test[f].values for f in features}
        y_test = pivoted_test[target].values
        print(f"测试数据准备完成: 时间点={y_test.shape[0]}, 合约数={y_test.shape[1]}")

        _update_multi_task(message="正在加载验证集数据...", progress=30)

        X_dict_now = None
        y_now = None
        pivoted_now = {}
        if (params.begin_time_now is not None) and (not df_now.empty):
            for col in features:
                df_now[col] = pd.to_numeric(df_now[col], errors="coerce")
            df_now.drop_duplicates(subset=["asset", "date"], keep="last", inplace=True)
            df_now.sort_values(["asset", "date"], ignore_index=True, inplace=True)

            df_now = df_now.sort_values(["asset", "date"]).reset_index(drop=True)
            df_now['future_return'] = df_now.groupby("asset")['close'].shift(-y_period) / df_now['close'] - 1
            df_now = df_now.dropna(subset=['future_return']).reset_index(drop=True)
            df_now['time'] = df_now['date']
            df_now['code'] = df_now['asset']

            data_now = df_now[features + [target, 'code', 'time']].copy()
            for col in features + [target]:
                try:
                    pivoted_now[col] = data_now.pivot(index='time', columns='code', values=col).sort_index().sort_index(axis=1)
                except Exception:
                    unique_dates_now = pd.Series(data_now['time']).unique()
                    unique_codes_now = pd.Series(data_now['code']).unique()
                    pivoted_now[col] = pd.DataFrame(0, index=unique_dates_now, columns=unique_codes_now)

            X_dict_now = {f: pivoted_now[f].values for f in features}
            y_now = pivoted_now[target].values
        
        _update_multi_task(message="正在评估因子...", progress=40)
        
        print(f"\n因子表达式列表: {formula_list}")
        print(f"品种合约: {SYMBOLS}")
        print(f"预测周期: {y_period}")
        
        my_cls = mfa.My
        
        factor_arrays = []
        factor_arrays_test = []
        factor_arrays_now = []
        
        for i, expr in enumerate(formula_list):
            print(f"\n评估因子 {i+1}/{len(formula_list)}: {expr}")
            
            exprs = mfa.eval_factors(expr, my_cls, X_dict)
            factor = exprs["factor"]
            factor_arrays.append(factor)
            
            if X_dict_test is not None:
                exprs_test = mfa.eval_factors(expr, my_cls, X_dict_test)
                factor_arrays_test.append(exprs_test["factor"])
            
            if X_dict_now is not None:
                exprs_now = mfa.eval_factors(expr, my_cls, X_dict_now)
                factor_arrays_now.append(exprs_now["factor"])
        
        print(f"\n所有因子评估完成，共 {len(factor_arrays)} 个因子")
        
        _update_multi_task(message="正在进行分层分析...", progress=50)
        
        ic_stats_list = []
        train_ic_means, test_ic_means, now_ic_means = [], [], []
        train_icirs, test_icirs, now_icirs = [], [], []
        
        for i, (factor, expr) in enumerate(zip(factor_arrays, formula_list)):
            ic_train = _normalize_multi_factor_ic_stats(mfa.calc_ic_stats(factor, y))
            ic_test = None
            ic_now = None
            if i < len(factor_arrays_test):
                ic_test = _normalize_multi_factor_ic_stats(mfa.calc_ic_stats(factor_arrays_test[i], y_test))
            if i < len(factor_arrays_now):
                ic_now = _normalize_multi_factor_ic_stats(mfa.calc_ic_stats(factor_arrays_now[i], y_now))
            
            train_ic_mean = ic_train.get("ic_mean", 0) if ic_train else 0
            train_icir = ic_train.get("icir", 0) if ic_train else 0
            test_ic_mean = ic_test.get("ic_mean", np.nan) if ic_test else np.nan
            test_icir = ic_test.get("icir", np.nan) if ic_test else np.nan
            now_ic_mean = ic_now.get("ic_mean", np.nan) if ic_now else np.nan
            now_icir = ic_now.get("icir", np.nan) if ic_now else np.nan
            
            train_ic_means.append(train_ic_mean)
            test_ic_means.append(test_ic_mean)
            now_ic_means.append(now_ic_mean)
            train_icirs.append(train_icir)
            test_icirs.append(test_icir)
            now_icirs.append(now_icir)
            
            ic_stats_list.append({
                "因子": f"因子{i+1}",
                "表达式": expr,
                "IC均值": train_ic_mean,
                "IC标准差": ic_train.get("ic_std", 0) if ic_train else 0,
                "ICIR": train_icir,
                "测试集IC均值": test_ic_mean,
                "测试集ICIR": test_icir,
                "验证集IC均值": now_ic_mean,
                "验证集ICIR": now_icir,
            })
            
            print(f"\n因子 {i+1} IC统计:")
            print(f"  训练集 IC均值: {train_ic_mean:.4f}, ICIR: {train_icir:.4f}")
            if np.isfinite(test_ic_mean):
                print(f"  测试集 IC均值: {test_ic_mean:.4f}, ICIR: {test_icir:.4f}")
            if np.isfinite(now_ic_mean):
                print(f"  验证集 IC均值: {now_ic_mean:.4f}, ICIR: {now_icir:.4f}")
        
        summary_df = pd.DataFrame(ic_stats_list)
        print("\n因子IC汇总:")
        print(summary_df)
        
        _update_multi_task(message="正在生成图表...", progress=70)
        
        charts = []
        names = [f"因子{i+1}" for i in range(len(formula_list))]
        x = np.arange(len(names))
        split_specs = [
            ("训练集", train_ic_means, train_icirs, "#1f77b4"),
            ("测试集", test_ic_means, test_icirs, "#ff7f0e"),
            ("验证集", now_ic_means, now_icirs, "#2ca02c"),
        ]
        available_splits = [
            (label, ic_vals, ir_vals, color)
            for label, ic_vals, ir_vals, color in split_specs
            if np.isfinite(np.asarray(ic_vals, dtype=np.float64)).any()
            or np.isfinite(np.asarray(ir_vals, dtype=np.float64)).any()
        ]
        width = 0.75 / max(len(available_splits), 1)
        
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle("多因子综合分析图", fontsize=14, fontweight="bold")
        ax1 = fig.add_subplot(221)
        for idx, (label, ic_vals, _, color) in enumerate(available_splits):
            offsets = x + (idx - (len(available_splits) - 1) / 2) * width
            ax1.bar(offsets, ic_vals, width=width, label=label, color=color, alpha=0.85)
        ax1.axhline(0, c="black", ls="--", alpha=0.6)
        ax1.set_ylabel("IC均值")
        ax1.set_title("因子IC均值对比（训练/测试/验证）")
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=45, ha="right")
        if available_splits:
            ax1.legend()
        
        ax2 = fig.add_subplot(222)
        for idx, (label, _, ir_vals, color) in enumerate(available_splits):
            offsets = x + (idx - (len(available_splits) - 1) / 2) * width
            ax2.bar(offsets, ir_vals, width=width, label=label, color=color, alpha=0.85)
        ax2.axhline(0, c="black", ls="--", alpha=0.6)
        ax2.set_ylabel("信息比率（ICIR）")
        ax2.set_title("因子信息比率对比（训练/测试/验证）")
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, rotation=45, ha="right")
        if available_splits:
            ax2.legend()
        
        def _mean_long_short_curve(factor_list, y_arr, pivot_dict):
            if y_arr is None or not factor_list:
                return None
            ls_series = []
            for idx, factor_arr in enumerate(factor_list):
                long_df = mfa.panel_to_long_factor_df(factor_arr, y_arr, pivot_dict, f"F{idx+1}")
                _, ls = mfa.build_quantile_report(long_df, quantiles=params.quantiles)
                if not ls.empty:
                    ls_series.append(ls.fillna(0))
            if not ls_series:
                return None
            merged = pd.concat(ls_series, axis=1)
            return merged.mean(axis=1, skipna=True).fillna(0).cumsum()
        
        ax3 = fig.add_subplot(223)
        ls_specs = [
            ("训练集", _mean_long_short_curve(factor_arrays, y, pivoted), "#1f77b4"),
            ("测试集", _mean_long_short_curve(factor_arrays_test, y_test, pivoted_test), "#ff7f0e"),
            ("验证集", _mean_long_short_curve(factor_arrays_now, y_now, pivoted_now), "#2ca02c"),
        ]
        plotted_ls = 0
        for label, ls_curve, color in ls_specs:
            if ls_curve is None or ls_curve.empty:
                continue
            ax3.plot(ls_curve.index, ls_curve.values, label=label, color=color)
            plotted_ls += 1
        ax3.axhline(0, c="black", ls="--", alpha=0.6)
        ax3.set_title("平均多空累计收益对比（训练/测试/验证）")
        if plotted_ls > 0:
            ax3.legend()
        
        corr_split_name, corr_df = _select_factor_corr_split(
            factor_names=names,
            factor_arrays_train=factor_arrays,
            factor_arrays_test=factor_arrays_test,
            factor_arrays_now=factor_arrays_now,
        )
        factor_corr = corr_df.to_numpy(dtype=np.float64)
        corr_labels = corr_df.columns.tolist()
        ax4 = fig.add_subplot(224)
        im = ax4.imshow(np.ma.masked_invalid(factor_corr), cmap="coolwarm", vmin=-1, vmax=1)
        ax4.set_facecolor("#f0f0f0")
        ax4.set_xticks(np.arange(len(corr_labels)))
        ax4.set_yticks(np.arange(len(corr_labels)))
        ax4.set_xticklabels(corr_labels)
        ax4.set_yticklabels(corr_labels)
        ax4.set_title(f"因子相关性热力图（{corr_split_name}）")
        if len(corr_labels) <= 10:
            for r in range(factor_corr.shape[0]):
                for c in range(factor_corr.shape[1]):
                    value = factor_corr[r, c]
                    txt = f"{value:.2f}" if np.isfinite(value) else "N/A"
                    ax4.text(c, r, txt, ha="center", va="center", fontsize=8, color="black")
        plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
        
        chart_base64, chart_path = _save_chart_and_encode(fig, task_id, "multi_factor_combined.png")
        charts.append({
            "name": "综合因子分析",
            "image": chart_base64,
            "image_data_uri": f"data:image/png;base64,{chart_base64}",
            "source_path": chart_path,
        })
        plt.close(fig)

        _update_multi_task(message="正在生成单因子拆分图...", progress=82)
        task_chart_dir = os.path.join(OUTPUT_DIR, task_id, "charts")
        detail_paths = _generate_per_factor_detail_charts(
            mfa=mfa,
            plt_module=plt,
            formulas=formula_list,
            factor_arrays=factor_arrays,
            factor_arrays_test=factor_arrays_test,
            factor_arrays_now=factor_arrays_now,
            y=y,
            y_test=y_test,
            y_now=y_now,
            pivoted=pivoted,
            pivoted_test=pivoted_test,
            pivoted_now=pivoted_now,
            chart_dir=task_chart_dir,
            quantiles=params.quantiles,
        )

        _update_multi_task(message="正在生成机器学习模型图...", progress=86)
        model_paths = _generate_model_charts(
            mfa=mfa,
            plt_module=plt,
            formulas=formula_list,
            factor_arrays_train=factor_arrays,
            factor_arrays_test=factor_arrays_test,
            factor_arrays_now=factor_arrays_now,
            y=y,
            y_test=y_test,
            y_now=y_now,
            pivoted=pivoted,
            pivoted_test=pivoted_test,
            pivoted_now=pivoted_now,
            chart_dir=task_chart_dir,
            quantiles=params.quantiles,
            use_lightgbm=params.use_lightgbm,
            use_elastic_net=params.use_elastic_net,
        )

        _update_multi_task(message="正在整理全量图表...", progress=90)
        charts = _collect_multi_factor_charts(
            task_id=task_id,
            preferred_paths=[chart_path, *detail_paths, *model_paths],
        )
        
        _update_multi_task(message="分析完成", progress=100)
        
        elapsed_time = time.time() - start_time
        
        sys.stdout = old_stdout
        output_text = capture.get_output()
        
        update_task(
            task_id,
            status="completed",
            progress=100.0,
            message="分析完成",
            result={
                "formula": formula_list,
                "symbols": SYMBOLS,
                "ic_stats": ic_stats_list,
                "charts": charts,
                "output_text": output_text,
                "elapsed_time": elapsed_time,
            },
            elapsed_time=elapsed_time,
            completed_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        import traceback
        sys.stdout = old_stdout
        output_text = capture.get_output() + f"\n错误: {str(e)}\n{traceback.format_exc()}" if 'capture' in locals() else f"错误: {str(e)}\n{traceback.format_exc()}"
        
        elapsed_time_on_error = (time.time() - start_time) if 'start_time' in locals() else None
        update_task(
            task_id,
            status="failed",
            message=f"任务失败: {str(e)}",
            result={"output_text": output_text, "error": str(e)},
            elapsed_time=elapsed_time_on_error,
            completed_at=datetime.now().isoformat()
        )


def generate_mock_data(symbols: List[str], features: List[str], n_timepoints: int = 500):
    np.random.seed(42)
    n_contracts = len(symbols)
    
    X_dict = {}
    for feature in features:
        X_dict[feature] = np.random.randn(n_timepoints, n_contracts) * 100 + 1000
    
    close_prices = X_dict["close"]
    future_return = np.roll(close_prices, -20, axis=0) / close_prices - 1
    future_return = np.nan_to_num(future_return, nan=0.0)
    
    return X_dict, future_return


def generate_mock_df_list(symbols: List[str], n_timepoints: int = 500, start_date: str = "2025-06-01") -> tuple:
    """生成模拟的 DataFrame 列表（模拟 get_futures_data 的返回格式）"""
    import pandas as pd
    np.random.seed(42)
    
    df_list = []
    dates = pd.date_range(start=start_date, periods=n_timepoints, freq='D')
    
    for symbol in symbols:
        base_price = np.random.uniform(1000, 5000)
        prices = base_price + np.cumsum(np.random.randn(n_timepoints) * 50)
        prices = np.maximum(prices, 100)
        
        df = pd.DataFrame({
            'date': dates,
            'open': prices * (1 + np.random.randn(n_timepoints) * 0.01),
            'high': prices * (1 + np.abs(np.random.randn(n_timepoints)) * 0.02),
            'low': prices * (1 - np.abs(np.random.randn(n_timepoints)) * 0.02),
            'close': prices,
            'volume': np.random.randint(10000, 1000000, n_timepoints),
            'open_interest': np.random.randint(10000, 500000, n_timepoints),
            'asset': symbol,
        })
        
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col].round(2)
        
        df['date'] = df['date'].astype(str)
        df_list.append(df)
    
    return df_list, dates


def create_mock_genetic_programmer(features):
    class MockNode:
        def __init__(self, name, value=None, children=None):
            self.name = name
            self.value = value
            self.children = children or []
        
        def to_str(self):
            if self.value is not None:
                return str(self.value)
            if self.children:
                args = ", ".join(c.to_str() for c in self.children)
                return f"{self.name}({args})"
            return self.name
        
        def depth(self):
            if not self.children:
                return 1
            return 1 + max(c.depth() for c in self.children)
        
        def size(self):
            if not self.children:
                return 1
            return 1 + sum(c.size() for c in self.children)
        
        def evaluate(self, X_dict, function_set=None, **kwargs):
            if self.value is not None:
                return self.value
            
            feature_map = {
                "CLOSE": "close",
                "OPEN": "open", 
                "HIGH": "high",
                "LOW": "low",
                "VOLUME": "volume",
                "OPEN_INTEREST": "open_interest",
            }
            
            if self.name in feature_map:
                return X_dict.get(feature_map[self.name], np.ones_like(list(X_dict.values())[0]) if X_dict else np.array([[1]]))
            
            if self.name == "ADD":
                results = [c.evaluate(X_dict, function_set) for c in self.children]
                return results[0] + results[1] if len(results) > 1 else results[0]
            elif self.name == "SUB":
                results = [c.evaluate(X_dict, function_set) for c in self.children]
                return results[0] - results[1] if len(results) > 1 else results[0]
            elif self.name == "MUL":
                results = [c.evaluate(X_dict, function_set) for c in self.children]
                result = results[0]
                for r in results[1:]:
                    result = result * r
                return result
            elif self.name == "DIV":
                results = [c.evaluate(X_dict, function_set) for c in self.children]
                denom = results[1] if len(results) > 1 else 1
                denom = np.where(denom == 0, 1, denom)
                return results[0] / denom
            return np.random.randn()
    
    class MockFunctionSet:
        def get_random_function(self, random_state=None):
            functions = ["ADD", "SUB", "MUL", "DIV"]
            return random_state.choice(functions) if random_state else np.random.choice(functions)
    
    class MockGeneticProgrammer:
        def __init__(self, **kwargs):
            self.generations = kwargs.get("generations", 15)
            self.variable_names = kwargs.get("variable_names", ["close"])
            self.best_programs_ = []
            self.function_set = MockFunctionSet()
            self._random_state = np.random.RandomState(kwargs.get("random_state", 42))
        
        def _random_program(self):
            depth = self._random_state.randint(1, 4)
            return self._build_random_tree(depth)
        
        def _build_random_tree(self, depth):
            if depth == 0:
                if self._random_state.random() < 0.5:
                    return MockNode("CONST", self._random_state.uniform(-1, 1))
                else:
                    return MockNode(self._random_state.choice(self.variable_names))
            
            func = self._random_state.choice(["ADD", "SUB", "MUL", "DIV"])
            children = [self._build_random_tree(depth - 1) for _ in range(2)]
            return MockNode(func, children=children)
        
        def _parse_expression(self, expr, features):
            def parse_recursive(s, start=0):
                i = s.find('(', start)
                if i == -1:
                    if s[start:].isdigit() or s[start:].replace('.', '').isdigit():
                        return MockNode("CONST", float(s[start:])), len(s)
                    for f in features:
                        if s[start:].upper() == f.upper():
                            return MockNode(f.upper()), len(s)
                    return MockNode("CONST", 1.0), len(s)
                
                name = s[start:i]
                name = name.upper()
                
                depth = 1
                j = i + 1
                args = []
                last_arg_start = j
                while j < len(s) and depth > 0:
                    if s[j] == '(':
                        depth += 1
                    elif s[j] == ')':
                        depth -= 1
                        if depth == 0:
                            arg_str = s[last_arg_start:j].strip()
                            if arg_str:
                                arg_node, _ = parse_recursive(arg_str, 0)
                                args.append(arg_node)
                            break
                    elif s[j] == ',' and depth == 1:
                        arg_str = s[last_arg_start:j].strip()
                        if arg_str:
                            arg_node, _ = parse_recursive(arg_str, 0)
                            args.append(arg_node)
                        last_arg_start = j + 1
                    j += 1
                
                return MockNode(name, children=args), j + 1
            
            try:
                node, _ = parse_recursive(expr, 0)
                return node
            except:
                return None
        
        def fit(self, fitness_func, fitness_args=(), fitness_kwargs=None, progress_callback=None):
            fitness_kwargs = fitness_kwargs or {}
            X_dict, y = fitness_args
            
            all_features = ["close", "open", "high", "low", "volume", "open_interest"]
            
            template_expressions = [
                "MUL(CLOSE, VOLUME)",
                "ADD(CLOSE, VOLUME)",
                "SUB(HIGH, LOW)",
                "DIV(CLOSE, OPEN)",
                "MUL(SUB(CLOSE, OPEN), VOLUME)",
                "ADD(DIV(CLOSE, OPEN), DIV(VOLUME, OPEN))",
                "MUL(CLOSE, SUB(HIGH, LOW))",
                "DIV(ADD(OPEN, CLOSE), 2)",
            ]
            
            for gen in range(self.generations):
                if progress_callback:
                    avg_fitness = self._random_state.uniform(0.01, 0.05)
                    best_fitness = avg_fitness + self._random_state.uniform(0.01, 0.03)
                    progress_callback(gen, 5.0, avg_fitness, 8, best_fitness)
                
                for expr in template_expressions:
                    try:
                        prog = self._parse_expression(expr, all_features)
                        if prog:
                            pred = prog.evaluate(X_dict)
                            if pred is not None and not np.all(np.isnan(pred)) and pred.shape == y.shape:
                                ic = np.corrcoef(pred.flatten(), y.flatten())[0, 1]
                                if not np.isnan(ic):
                                    self.best_programs_.append(prog)
                    except:
                        pass
            
            self.best_programs_ = self.best_programs_[:5] if len(self.best_programs_) >= 5 else self.best_programs_
            
            if not self.best_programs_:
                for expr in template_expressions[:3]:
                    prog = self._parse_expression(expr, all_features)
                    if prog:
                        self.best_programs_.append(prog)
    
    return MockGeneticProgrammer, MockNode


class ProgressCallback:
    def __init__(self, task_id: str, total_generations: int):
        self.task_id = task_id
        self.start_time = None
        self.total_generations = total_generations

    def __call__(self, generation: int, avg_length: float, avg_fitness: float,
                 best_length: int, best_fitness: float):
        if is_task_stopped(self.task_id):
            raise StopIteration("任务已停止")

        if self.start_time is None:
            self.start_time = time.time()

        progress = (generation + 1) / self.total_generations * 100
        elapsed = time.time() - self.start_time if self.start_time else 0

        update_task(
            self.task_id,
            status="running",
            progress=progress,
            message=f"正在进化第 {generation + 1}/{self.total_generations} 代 | 平均适应度: {avg_fitness:.6f} | 最优适应度: {best_fitness:.6f}",
            elapsed_time=elapsed
        )


def prepare_data_mining(
    symbols: List[str],
    features: List[str],
    ic_period: int,
    train: bool = True,
    begin_time: str = None,
    end_time: str = None,
    symbol_cycle: str = "15分钟"
):
    from factor_mining import get_futures_data

    target = "future_return"
    print(
        f"[Mining][prepare_data_mining] train={train} symbol_cycle={symbol_cycle!r} "
        f"begin_time={begin_time!r} end_time={end_time!r} symbols={len(symbols)}"
    )

    if begin_time is None or end_time is None:
        from factor_mining import BEGIN_TIME, END_TIME, BEGIN_TIME_TEST, END_TIME_TEST
        if train:
            begin_time = BEGIN_TIME
            end_time = END_TIME
        else:
            begin_time = BEGIN_TIME_TEST
            end_time = END_TIME_TEST

    df_list = []
    for symbol in symbols:
        try:
            data = get_futures_data(symbol, begin_time, end_time, symbol_cycle=symbol_cycle)
            if len(data) > 0:
                df_list.append(data)
        except Exception as e:
            print(
                f"[Mining][prepare_data_mining] 跳过合约 {symbol!r}，"
                f"获取数据失败: {e}"
            )
    
    if len(df_list) == 0:
        raise ValueError(f"{'训练' if train else '测试'}集没有有效数据")
    
    df = pd.concat(df_list, ignore_index=True)
    
    for col in ["open", "high", "low", "close", "volume", "open_interest"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df.drop_duplicates(subset=["asset", "date"], keep="last", inplace=True)
    df.sort_values(["asset", "date"], ignore_index=True, inplace=True)
    
    df = df.sort_values(["asset", "date"]).reset_index(drop=True)
    df["future_return"] = df.groupby("asset")["close"].shift(-ic_period) / df["close"] - 1
    
    df = df.dropna(subset=["future_return"]).reset_index(drop=True)
    
    df["time"] = df["date"]
    df["code"] = df["asset"]
    
    pivoted = {}
    for col in features + [target]:
        try:
            pivoted[col] = df.pivot(index="time", columns="code", values=col)
        except Exception:
            unique_dates = pd.Series(df["time"]).unique()
            unique_codes = pd.Series(df["code"]).unique()
            pivoted[col] = pd.DataFrame(0, index=unique_dates, columns=unique_codes)
    
    X_dict = {f: pivoted[f].values for f in features}
    y = pivoted[target].values
    
    return X_dict, y


def run_mining_task(task_id: str, params: MiningParams):
    try:
        if is_task_stopped(task_id):
            update_task(task_id, status="stopped", message="任务已停止")
            return

        print(f"[Mining][{task_id}] params.symbol_cycle={params.symbol_cycle!r}")
        os.environ["FACTOR_MINING_DEFAULT_SYMBOL_CYCLE"] = params.symbol_cycle
        print(
            f"[Mining][{task_id}] env.FACTOR_MINING_DEFAULT_SYMBOL_CYCLE="
            f"{os.environ['FACTOR_MINING_DEFAULT_SYMBOL_CYCLE']!r}"
        )

        import factor_mining as fm
        from factor_mining import (
            GeneticProgrammer,
            FunctionSet,
            fitness_func,
            get_symbols_by_sector as fm_get_symbols,
            FUTURES_SECTORS as FM_FUTURES_SECTORS,
        )
        fm.SYMBOL_CYCLE = params.symbol_cycle
        print(f"[Mining][{task_id}] factor_mining.SYMBOL_CYCLE={fm.SYMBOL_CYCLE!r}")

        update_task(task_id, status="running", message="正在初始化数据...")

        is_mock_mode = params.use_mock_data or not MINING_AVAILABLE
        print(
            f"[Mining][{task_id}] is_mock_mode={is_mock_mode} "
            f"MINING_AVAILABLE={MINING_AVAILABLE} "
            f"QMF_DATA_AVAILABLE={getattr(fm, 'QMF_DATA_AVAILABLE', None)!r}"
        )

        if not is_mock_mode and not getattr(fm, 'QMF_DATA_AVAILABLE', True):
            is_mock_mode = True
            print("警告: qmf_data 不可用，自动切换到 Mock 模式")

        if is_mock_mode:
            symbols = params.symbols if params.symbols else ["au888", "ag888", "cu888"]
            features = params.features

            update_task(task_id, message="正在生成模拟数据...")
            X_dict, y = generate_mock_data(symbols, features, n_timepoints=500)
            X_dict_test, y_test = generate_mock_data(symbols, features, n_timepoints=200)
        else:
            if params.symbols:
                symbols = params.symbols
            else:
                manual_symbols = getattr(fm, "MANUAL_SYMBOLS", ["au888", "ag888"])
                symbols = fm_get_symbols(params.selected_sector, FM_FUTURES_SECTORS, manual_symbols)

            symbols = list(dict.fromkeys(symbols)) if symbols else []
            print(
                f"[Mining][{task_id}] params.selected_sector={params.selected_sector!r} "
                f"resolved_symbols_count={len(symbols)}"
            )
            if symbols:
                print(f"[Mining][{task_id}] resolved_symbols={symbols}")
            
            features = params.features
            
            update_task(task_id, message="正在加载训练数据...")
            X_dict, y = prepare_data_mining(
                symbols, features, params.ic_period, train=True,
                begin_time=params.begin_time,
                end_time=params.end_time,
                symbol_cycle=params.symbol_cycle
            )
            
            update_task(task_id, message="正在加载测试数据...")
            X_dict_test, y_test = prepare_data_mining(
                symbols, features, params.ic_period, train=False,
                begin_time=params.begin_time_test,
                end_time=params.end_time_test,
                symbol_cycle=params.symbol_cycle
            )
        
        update_task(task_id, message="正在初始化遗传编程器...")
        
        if is_mock_mode:
            MockGP, MockNode = create_mock_genetic_programmer(features)
            gp = MockGP(
                generations=params.generations,
                population_size=params.population_size,
                variable_names=features,
                random_state=params.random_state,
            )
            
            progress_callback = ProgressCallback(task_id, params.generations)
            
            def mock_fitness(prog, X_dict, y, return_details=False, function_set=None, X_dict_test=None, y_test=None):
                pred = prog.evaluate(X_dict)
                if pred is None:
                    ic = 0.0
                else:
                    ic = np.corrcoef(pred.flatten(), y.flatten())[0, 1]
                    if np.isnan(ic):
                        ic = 0.0
                
                return {
                    'fitness': abs(ic),
                    'mean_ic': ic,
                    'icir': ic * 0.8,
                    'mean_ic_test': ic * 0.9,
                    'icir_test': ic * 0.7,
                    'valid_ts': y.shape[0] - 20,
                    'total_ts': y.shape[0],
                }
            
            gp.fit(
                fitness_func=mock_fitness,
                fitness_args=(X_dict, y),
                fitness_kwargs={},
                progress_callback=progress_callback,
            )
            
            results = []
            for i, prog in enumerate(gp.best_programs_):
                details = mock_fitness(prog, X_dict, y, return_details=True)
                
                results.append({
                    "rank": i + 1,
                    "expression": prog.to_str(),
                    "depth": prog.depth(),
                    "size": prog.size(),
                    "fitness": details['fitness'],
                    "train_ic": details['mean_ic'],
                    "train_ir": details['icir'],
                    "test_ic": details['mean_ic_test'],
                    "test_ir": details['icir_test'],
                    "valid_ts": details['valid_ts'],
                    "total_ts": details['total_ts'],
                })
        else:
            gp = GeneticProgrammer(
                generations=params.generations,
                population_size=params.population_size,
                tournament_size=params.tournament_size,
                n_components=params.n_components,
                hall_of_fame=params.hall_of_fame,
                function_set=FunctionSet(),
                variable_names=features,
                ts_window=params.ts_window,
                random_state=params.random_state,
                const_range=params.const_range,
                p_crossover=params.p_crossover,
                p_subtree_mutation=params.p_subtree_mutation,
                p_hoist_mutation=params.p_hoist_mutation,
                p_point_mutation=params.p_point_mutation,
                immigration_rate=params.immigration_rate,
                parsimony_coefficient=params.parsimony_coefficient,
                init_depth=params.init_depth,
                suit_size=params.suit_size,
                stagnation_threshold=params.stagnation_threshold,
                min_improvement=params.min_improvement,
                max_restarts=params.max_restarts,
                max_program_size=params.max_program_size,
                max_best_program_size=params.max_best_program_size,
                ic_objective=params.ic_objective,
            )
            
            progress_callback = ProgressCallback(task_id, params.generations)
            
            gp.fit(
                fitness_func=fitness_func,
                fitness_args=(X_dict, y),
                fitness_kwargs={
                    "X_dict_test": X_dict_test, 
                    "y_test": y_test,
                },
                progress_callback=progress_callback,
            )
            
            results = []
            for i, prog in enumerate(gp.best_programs_):
                details = fitness_func(
                    prog,
                    X_dict,
                    y,
                    return_details=True,
                    function_set=gp.function_set,
                    X_dict_test=X_dict_test,
                    y_test=y_test,
                )
                
                results.append({
                    "rank": i + 1,
                    "expression": prog.to_str(),
                    "depth": prog.depth(),
                    "size": prog.size(),
                    "fitness": details['fitness'],
                    "train_ic": details['mean_ic'],
                    "train_ir": details['icir'],
                    "test_ic": details['mean_ic_test'],
                    "test_ir": details['icir_test'],
                    "valid_ts": details['valid_ts'],
                    "total_ts": details['total_ts'],
                })
        
        update_task(
            task_id,
            status="completed",
            progress=100.0,
            message="因子挖掘完成",
            result={
                "best_factors": results,
                "total_generations": params.generations,
                "population_size": params.population_size,
                "symbols_count": len(symbols),
                "features": features,
            },
            completed_at=datetime.now().isoformat()
        )

    except StopIteration:
        update_task(
            task_id,
            status="stopped",
            message="任务已停止",
            completed_at=datetime.now().isoformat()
        )
    except Exception as e:
        import traceback
        update_task(
            task_id,
            status="failed",
            message=f"任务失败: {str(e)}",
            result={"error": str(e), "traceback": traceback.format_exc()},
            completed_at=datetime.now().isoformat()
        )


@app.get("/")
async def root():
    return {
        "message": "量化分析统一API服务", 
        "version": "2.0.0",
        "services": {
            "single_factor": SINGLE_FACTOR_AVAILABLE,
            "multi_factor": MULTI_FACTOR_AVAILABLE,
            "mining": MINING_AVAILABLE
        }
    }


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "services": {
            "single_factor": SINGLE_FACTOR_AVAILABLE,
            "multi_factor": MULTI_FACTOR_AVAILABLE,
            "mining": MINING_AVAILABLE
        }
    }


@app.post("/api/single-factor/start", response_model=TaskStatus)
async def start_single_factor(params: SingleFactorParams, background_tasks: BackgroundTasks):
    if not SINGLE_FACTOR_AVAILABLE:
        raise HTTPException(status_code=503, detail="单因子分析服务不可用")
    
    task_id = str(uuid.uuid4())
    create_task(task_id, "single_factor", "任务已创建，等待执行")
    
    background_tasks.add_task(run_single_factor_task, task_id, params)
    
    return get_task(task_id)


@app.post("/api/multi-factor/start", response_model=TaskStatus)
async def start_multi_factor(params: MultiFactorParams, background_tasks: BackgroundTasks):
    if not MULTI_FACTOR_AVAILABLE:
        raise HTTPException(status_code=503, detail="多因子分析服务不可用")
    
    task_id = str(uuid.uuid4())
    create_task(task_id, "multi_factor", "任务已创建，等待执行")
    
    background_tasks.add_task(run_multi_factor_task, task_id, params)
    
    return get_task(task_id)


@app.post("/api/mining/start", response_model=TaskStatus)
async def start_mining(params: MiningParams, background_tasks: BackgroundTasks):
    if not MINING_AVAILABLE and not params.use_mock_data:
        raise HTTPException(status_code=503, detail="因子挖掘服务不可用")
    
    task_id = str(uuid.uuid4())
    create_task(task_id, "mining", "任务已创建，等待执行")
    
    background_tasks.add_task(run_mining_task, task_id, params)
    
    return get_task(task_id)


@app.get("/api/task/status/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return task


@app.get("/api/task/result/{task_id}")
async def get_task_result(task_id: str):
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    if task.status != "completed":
        return {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "status": task.status,
            "progress": task.progress,
            "message": task.message,
            "result": task.result,
        }
    
    return {
        "task_id": task.task_id,
        "task_type": task.task_type,
        "status": task.status,
        "result": task.result,
        "elapsed_time": task.elapsed_time,
    }


# ===== 兼容旧版因子挖掘前端路由（保持与 api_server.py 一致） =====
@app.get("/api/mining/status/{task_id}", response_model=TaskStatus)
async def get_mining_status_compat(task_id: str):
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    if task.task_type != "mining":
        raise HTTPException(status_code=404, detail="任务类型不匹配（非因子挖掘任务）")
    return task


@app.get("/api/mining/result/{task_id}")
async def get_mining_result_compat(task_id: str):
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    if task.task_type != "mining":
        raise HTTPException(status_code=404, detail="任务类型不匹配（非因子挖掘任务）")

    if task.status != "completed":
        return {
            "status": task.status,
            "progress": task.progress,
            "message": task.message,
            "result": task.result,
        }

    return {
        "status": task.status,
        "result": task.result,
        "elapsed_time": task.elapsed_time,
    }


@app.get("/api/task/list")
async def list_tasks(limit: int = 10, task_type: Optional[str] = None):
    with tasks_lock:
        tasks = list(tasks_storage.values())
    
    if task_type:
        tasks = [t for t in tasks if t.task_type == task_type]
    
    tasks.sort(key=lambda x: x.created_at, reverse=True)
    return {"tasks": tasks[:limit], "total": len(tasks)}


@app.delete("/api/task/{task_id}")
async def delete_task(task_id: str):
    with tasks_lock:
        if task_id in tasks_storage:
            del tasks_storage[task_id]
            if task_id in stop_flags:
                del stop_flags[task_id]
            return {"message": "任务已删除"}
    raise HTTPException(status_code=404, detail="任务不存在")


@app.post("/api/task/pause/{task_id}")
async def pause_task(task_id: str):
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")

    if task.status not in ("running", "pending"):
        return {"success": False, "message": "任务不在运行中，无法暂停"}

    stop_task(task_id)
    update_task(task_id, status="paused", message="任务已暂停")

    return {"success": True, "message": "任务已暂停"}


@app.get("/api/config/single-factor/default")
async def get_single_factor_default_config():
    return {
        "formula": "RANK(WR(2), BOLL_UPPER(24, 2.26))",
        "selected_sector": ["时间分类", "有色金属"],
        "begin_time": "2025-06-01",
        "end_time": "2025-08-31",
        "begin_time_test": "2025-09-01",
        "end_time_test": "2025-12-31",
        "begin_time_now": "2026-01-01",
        "symbol_cycle": "15分钟",
        "y_period": 1,
        "quantiles": 5,
    }


@app.get("/api/config/multi-factor/default")
async def get_multi_factor_default_config():
    return {
        "formula": DEFAULT_MULTI_FACTOR_FORMULAS.copy(),
        "use_lightgbm": True,
        "use_elastic_net": True,
        "use_instashap": True,
        "selected_sector": ["时间分类", "有色金属"],
        "symbols": None,
        "begin_time": "2025-06-01",
        "end_time": "2025-08-31",
        "begin_time_test": "2025-09-01",
        "end_time_test": "2025-12-31",
        "begin_time_now": "2026-01-01",
        "symbol_cycle": "15分钟",
        "y_period": 1,
        "quantiles": 5,
    }


@app.get("/api/config/mining/default")
async def get_mining_default_config():
    return {
        "begin_time": "2025-06-01",
        "end_time": "2025-08-31",
        "begin_time_test": "2025-09-01",
        "end_time_test": "2025-12-31",
        "begin_time_now": "2026-01-01",
        "symbol_cycle": "15分钟",
        "y_period": 1,
        "selected_sector": ["时间分类", "有色金属"],
        "symbols": None,
        "quantiles": 5,
        "generations": 15,
        "population_size": 120,
        "tournament_size": 4,
        "n_components": 5,
        "hall_of_fame": 6,
        "ts_window": 20,
        "const_range": [-2, 120],
        "p_crossover": 0.30,
        "p_subtree_mutation": 0.30,
        "p_hoist_mutation": 0.10,
        "p_point_mutation": 0.20,
        "immigration_rate": 0.20,
        "parsimony_coefficient": 0.002,
        "init_depth": [3, 8],
        "suit_size": [4, 14],
        "stagnation_threshold": 6,
        "min_improvement": 0.001,
        "max_restarts": 3,
        "max_program_size": 24,
        "max_best_program_size": 24,
        "ic_objective": "max",
        "features": ["open", "close", "high", "low", "volume", "open_interest"],
        "ic_period": 20,
        "fitness_w_train": 0.6,
        "fitness_w_test": 0.4,
        "random_state": None,
        "use_mock_data": False,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


