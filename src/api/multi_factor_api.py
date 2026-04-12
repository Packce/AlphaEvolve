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
from pathlib import Path
from datetime import datetime
import threading
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = "multi_factor_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _merge_legacy_keys(data: Any, key_map: Dict[str, str]) -> Any:
    if not isinstance(data, dict):
        return data
    merged = dict(data)
    for old_key, new_key in key_map.items():
        if old_key in data and new_key not in merged:
            merged[new_key] = data[old_key]
    return merged


DEFAULT_MULTI_FACTOR_FORMULAS = [
    "RANK(WR(2), BOLL_UPPER(24, 2.26))",
    "LOWRANGE(BIAS(90))",
    "HHVBARS(KDJ_K(4, 84), COS(MACD_DEA(49, 46, 88)))",
    "WR(2)",
]


class MultiFactorAnalysisParams(BaseModel):
    formula: List[str] = Field(
        default_factory=lambda: DEFAULT_MULTI_FACTOR_FORMULAS.copy(),
        description="因子表达式列表",
    )
    USE_LIGHTGBM: bool = Field(default=True, description="是否使用LightGBM模型")
    USE_ELASTIC_NET: bool = Field(default=True, description="是否使用Elastic Net模型")
    USE_INSTASHAP: bool = Field(default=True, description="是否使用InstaSHAP模型")
    SELECTED_SECTOR: Optional[List[str]] = Field(default=None, description="选择的行业板块")
    MANUAL_SYMBOLS: Optional[List[str]] = Field(default=None, description="手动指定的合约列表")
    BEGIN_TIME: str = Field(default="2025-06-01", description="训练集开始时间")
    END_TIME: str = Field(default="2025-08-31", description="训练集结束时间")
    BEGIN_TIME_TEST: str = Field(default="2025-09-01", description="测试集开始时间")
    END_TIME_TEST: str = Field(default="2025-12-31", description="测试集结束时间")
    BEGIN_TIME_NOW: str = Field(default="2026-01-01", description="验证集开始时间")
    SYMBOL_CYCLE: str = Field(default="15分钟", description="数据周期")
    Y_PERIOD: int = Field(default=1, ge=1, le=20, description="预测周期")
    quantiles: int = Field(default=5, ge=2, le=10, description="分位数数量")

    @model_validator(mode="before")
    @classmethod
    def _normalize_lowercase_fields(cls, data: Any) -> Any:
        return _merge_legacy_keys(
            data,
            {
                "selected_sector": "SELECTED_SECTOR",
                "manual_symbols": "MANUAL_SYMBOLS",
                "symbols": "MANUAL_SYMBOLS",
                "begin_time": "BEGIN_TIME",
                "end_time": "END_TIME",
                "begin_time_test": "BEGIN_TIME_TEST",
                "end_time_test": "END_TIME_TEST",
                "begin_time_now": "BEGIN_TIME_NOW",
                "symbol_cycle": "SYMBOL_CYCLE",
                "y_period": "Y_PERIOD",
                "use_lightgbm": "USE_LIGHTGBM",
                "use_elastic_net": "USE_ELASTIC_NET",
                "use_instashap": "USE_INSTASHAP",
            },
        )


class TaskStatus(BaseModel):
    task_id: str
    status: str
    progress: float
    message: str
    result: Optional[Dict[str, Any]] = None
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    elapsed_time: Optional[float] = None


app = FastAPI(
    title="多因子分析API",
    description="多因子分析服务 - 支持多因子合成和模型分析",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tasks_storage: Dict[str, TaskStatus] = {}
tasks_lock = threading.Lock()


def create_task(task_id: str, message: str = "任务已创建") -> TaskStatus:
    task = TaskStatus(
        task_id=task_id,
        status="pending",
        progress=0.0,
        message=message,
        created_at=datetime.now().isoformat()
    )
    with tasks_lock:
        tasks_storage[task_id] = task
    return task


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


def _normalize_multi_factor_ic_stats(stats: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
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

    out["ic_mean"] = _safe_float(ic_mean)
    out["icir"] = _safe_float(icir)
    out["ic_std"] = _safe_float(ic_std)
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
    if name.startswith("因子") and "rolling_ic" in name:
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
    chart_dirs = [
        os.path.join(OUTPUT_DIR, task_id, "charts"),
        "多因子分析可视化",
        os.path.join("src", "core", "factor", "多因子分析可视化"),
        os.path.join("src", "api", "多因子分析可视化"),
    ]

    unique_files: List[str] = []
    seen_names: set[str] = set()

    def _try_add(path_str: str) -> None:
        if not path_str:
            return
        p = Path(path_str)
        if not p.exists() or not p.is_file():
            return
        key = p.name
        if key in seen_names:
            return
        seen_names.add(key)
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
        for label, long_df in [("训练集", analysis_train), ("测试集", analysis_test), ("真实集", analysis_now)]:
            if long_df is None or long_df.empty:
                continue
            ic_s = mfa.ic_curve_from_long(long_df)
            if not ic_s.empty:
                ic_s.rolling(20, min_periods=5).mean().plot(label=f"{label} Rolling IC(20)")
        plt_module.axhline(0.0, linestyle="--")
        plt_module.title(f"因子{i + 1} Rolling Rank IC\n{expr}")
        plt_module.legend()
        plt_module.tight_layout()
        ic_path = _save_figure_to_file(fig_ic, chart_dir, f"因子{i + 1}_Rolling_IC.png", dpi=150)
        plt_module.close(fig_ic)
        generated.append(ic_path)

        fig_ls = plt_module.figure(figsize=(12, 4))
        for label, long_df in [("训练集", analysis_train), ("测试集", analysis_test), ("真实集", analysis_now)]:
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


def run_analysis_task(task_id: str, params: MultiFactorAnalysisParams):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import sys
        import multi_factor_analysis as mfa
        _configure_plot_style(plt)
        
        start_time = time.time()
        update_task(task_id, status="running", message="正在初始化...")
        
        old_stdout = sys.stdout
        capture = OutputCapture(mirror=old_stdout)
        sys.stdout = capture
        
        mfa.SELECTED_SECTOR = params.SELECTED_SECTOR
        mfa.MANUAL_SYMBOLS = params.MANUAL_SYMBOLS if params.MANUAL_SYMBOLS else ["au888", "ag888"]
        mfa.BEGIN_TIME = params.BEGIN_TIME
        mfa.END_TIME = params.END_TIME
        mfa.BEGIN_TIME_TEST = params.BEGIN_TIME_TEST
        mfa.END_TIME_TEST = params.END_TIME_TEST
        mfa.BEGIN_TIME_NOW = params.BEGIN_TIME_NOW
        mfa.SYMBOL_CYCLE = params.SYMBOL_CYCLE
        mfa.Y_PERIOD = params.Y_PERIOD
        mfa.USE_LIGHTGBM = params.USE_LIGHTGBM
        mfa.USE_ELASTIC_NET = params.USE_ELASTIC_NET
        mfa.USE_INSTASHAP = params.USE_INSTASHAP
        
        if isinstance(params.formula, list):
            mfa.formula = params.formula
        else:
            mfa.formula = [params.formula]
        
        update_task(task_id, message="正在加载数据...", progress=10)
        
        SYMBOLS = mfa.get_symbols_by_sector(mfa.SELECTED_SECTOR, mfa.FUTURES_SECTORS, mfa.MANUAL_SYMBOLS)
        
        print(f"当前使用的合约列表（共 {len(SYMBOLS)} 个合约）:")
        print(SYMBOLS)
        
        df_list = []
        for symbol in SYMBOLS:
            data = mfa.get_futures_data(symbol, mfa.BEGIN_TIME, mfa.END_TIME, mfa.SYMBOL_CYCLE)
            if len(data) > 0:
                df_list.append(data)
        
        if len(df_list) == 0:
            raise ValueError("训练集没有有效数据")
        
        df = pd.concat(df_list, ignore_index=True)
        
        features = ['open', 'close', 'high', 'low', 'volume', 'open_interest']
        target = 'future_return'
        
        for col in features:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        df.drop_duplicates(subset=["asset", "date"], keep="last", inplace=True)
        df.sort_values(["asset", "date"], ignore_index=True, inplace=True)
        
        print(f"训练集数据预处理完成，共 {len(df)} 条记录")
        
        df = df.sort_values(["asset", "date"]).reset_index(drop=True)
        df['future_return'] = df.groupby("asset")['close'].shift(-mfa.Y_PERIOD) / df['close'] - 1
        df = df.dropna(subset=['future_return']).reset_index(drop=True)
        
        df['time'] = df['date']
        df['code'] = df['asset']
        
        data = df[features + [target, 'code', 'time']].copy()
        
        pivoted = {}
        for col in features + [target]:
            try:
                pivoted[col] = data.pivot(index='time', columns='code', values=col).sort_index().sort_index(axis=1)
            except:
                unique_dates = pd.Series(data['time']).unique()
                unique_codes = pd.Series(data['code']).unique()
                pivoted[col] = pd.DataFrame(0, index=unique_dates, columns=unique_codes)
        
        X_dict = {f: pivoted[f].values for f in features}
        y = pivoted[target].values
        
        print(f"训练数据准备完成: 时间点={y.shape[0]}, 合约数={y.shape[1]}")
        
        update_task(task_id, message="正在加载测试数据...", progress=20)
        
        df_list_test = []
        for symbol in SYMBOLS:
            data = mfa.get_futures_data(symbol, mfa.BEGIN_TIME_TEST, mfa.END_TIME_TEST, mfa.SYMBOL_CYCLE)
            if len(data) > 0:
                df_list_test.append(data)
        
        if len(df_list_test) == 0:
            raise ValueError("测试集没有有效数据")
        
        df_test = pd.concat(df_list_test, ignore_index=True)
        for col in features:
            df_test[col] = pd.to_numeric(df_test[col], errors="coerce")
        df_test.drop_duplicates(subset=["asset", "date"], keep="last", inplace=True)
        df_test.sort_values(["asset", "date"], ignore_index=True, inplace=True)
        
        print(f"测试集数据预处理完成，共 {len(df_test)} 条记录")
        
        df_test = df_test.sort_values(["asset", "date"]).reset_index(drop=True)
        df_test['future_return'] = df_test.groupby("asset")['close'].shift(-mfa.Y_PERIOD) / df_test['close'] - 1
        df_test = df_test.dropna(subset=['future_return']).reset_index(drop=True)
        
        df_test['time'] = df_test['date']
        df_test['code'] = df_test['asset']
        
        data_test = df_test[features + [target, 'code', 'time']].copy()
        
        pivoted_test = {}
        for col in features + [target]:
            try:
                pivoted_test[col] = data_test.pivot(index='time', columns='code', values=col).sort_index().sort_index(axis=1)
            except:
                unique_dates_test = pd.Series(data_test['time']).unique()
                unique_codes_test = pd.Series(data_test['code']).unique()
                pivoted_test[col] = pd.DataFrame(0, index=unique_dates_test, columns=unique_codes_test)
        
        X_dict_test = {f: pivoted_test[f].values for f in features}
        y_test = pivoted_test[target].values
        
        print(f"测试数据准备完成: 时间点={y_test.shape[0]}, 合约数={y_test.shape[1]}")
        
        update_task(task_id, message="正在加载验证集数据...", progress=30)
        
        df_list_now = []
        for symbol in SYMBOLS:
            data = mfa.get_futures_data(symbol, mfa.BEGIN_TIME_NOW, None, mfa.SYMBOL_CYCLE)
            if len(data) > 0:
                df_list_now.append(data)
        
        if len(df_list_now) > 0:
            df_now = pd.concat(df_list_now, ignore_index=True)
            for col in features:
                df_now[col] = pd.to_numeric(df_now[col], errors="coerce")
            df_now.drop_duplicates(subset=["asset", "date"], keep="last", inplace=True)
            df_now.sort_values(["asset", "date"], ignore_index=True, inplace=True)
            
            df_now = df_now.sort_values(["asset", "date"]).reset_index(drop=True)
            df_now['future_return'] = df_now.groupby("asset")['close'].shift(-mfa.Y_PERIOD) / df_now['close'] - 1
            df_now = df_now.dropna(subset=['future_return']).reset_index(drop=True)
            
            df_now['time'] = df_now['date']
            df_now['code'] = df_now['asset']
            
            data_now = df_now[features + [target, 'code', 'time']].copy()
            
            pivoted_now = {}
            for col in features + [target]:
                try:
                    pivoted_now[col] = data_now.pivot(index='time', columns='code', values=col).sort_index().sort_index(axis=1)
                except:
                    unique_dates_now = pd.Series(data_now['time']).unique()
                    unique_codes_now = pd.Series(data_now['code']).unique()
                    pivoted_now[col] = pd.DataFrame(0, index=unique_dates_now, columns=unique_codes_now)
            
            X_dict_now = {f: pivoted_now[f].values for f in features}
            y_now = pivoted_now[target].values
        else:
            X_dict_now = None
            y_now = None
            pivoted_now = {}
        
        update_task(task_id, message="正在评估因子...", progress=40)
        
        print(f"\n因子表达式列表: {mfa.formula}")
        print(f"品种合约: {SYMBOLS}")
        print(f"预测周期: {mfa.Y_PERIOD}")
        
        my_cls = mfa.My
        
        factor_arrays = []
        factor_arrays_test = []
        factor_arrays_now = []
        
        for i, expr in enumerate(mfa.formula):
            print(f"\n评估因子 {i+1}/{len(mfa.formula)}: {expr}")
            
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
        
        update_task(task_id, message="正在进行分层分析...", progress=50)
        
        ic_stats_list = []
        train_ic_means, test_ic_means, now_ic_means = [], [], []
        train_icirs, test_icirs, now_icirs = [], [], []
        
        for i, (factor, expr) in enumerate(zip(factor_arrays, mfa.formula)):
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
        
        update_task(task_id, message="正在生成图表...", progress=70)
        
        charts = []
        names = [f"因子{i+1}" for i in range(len(mfa.formula))]
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

        update_task(task_id, message="正在生成单因子拆分图...", progress=82)
        task_chart_dir = os.path.join(OUTPUT_DIR, task_id, "charts")
        detail_paths = _generate_per_factor_detail_charts(
            mfa=mfa,
            plt_module=plt,
            formulas=mfa.formula,
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

        update_task(task_id, message="正在整理全量图表...", progress=90)
        charts = _collect_multi_factor_charts(
            task_id=task_id,
            preferred_paths=[chart_path, *detail_paths],
        )
        
        update_task(task_id, message="分析完成", progress=100)
        
        elapsed_time = time.time() - start_time
        
        sys.stdout = old_stdout
        output_text = capture.get_output()
        
        update_task(
            task_id,
            status="completed",
            progress=100.0,
            message="分析完成",
            result={
                "formula": mfa.formula,
                "symbols": SYMBOLS,
                "ic_stats": ic_stats_list,
                "charts": charts,
                "output_text": output_text,
                "elapsed_time": elapsed_time,
            },
            completed_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        import traceback
        sys.stdout = old_stdout
        output_text = capture.get_output() + f"\n错误: {str(e)}\n{traceback.format_exc()}"
        
        update_task(
            task_id,
            status="failed",
            message=f"任务失败: {str(e)}",
            result={"output_text": output_text, "error": str(e)},
            completed_at=datetime.now().isoformat()
        )


@app.get("/")
async def root():
    return {
        "message": "多因子分析API服务", 
        "version": "1.0.0"
    }


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/analysis/start", response_model=TaskStatus)
async def start_analysis(params: MultiFactorAnalysisParams, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    create_task(task_id, "任务已创建，等待执行")
    
    background_tasks.add_task(run_analysis_task, task_id, params)
    
    return get_task(task_id)


@app.get("/api/analysis/status/{task_id}", response_model=TaskStatus)
async def get_analysis_status(task_id: str):
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return task


@app.get("/api/analysis/result/{task_id}")
async def get_analysis_result(task_id: str):
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    
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


@app.get("/api/analysis/list")
async def list_analysis_tasks(limit: int = 10):
    with tasks_lock:
        tasks = list(tasks_storage.values())
    tasks.sort(key=lambda x: x.created_at, reverse=True)
    return {"tasks": tasks[:limit], "total": len(tasks)}


@app.delete("/api/analysis/task/{task_id}")
async def delete_task(task_id: str):
    with tasks_lock:
        if task_id in tasks_storage:
            del tasks_storage[task_id]
            return {"message": "任务已删除"}
    raise HTTPException(status_code=404, detail="任务不存在")


@app.get("/api/config/default")
async def get_default_config():
    return {
        "formula": DEFAULT_MULTI_FACTOR_FORMULAS.copy(),
        "USE_LIGHTGBM": True,
        "USE_ELASTIC_NET": True,
        "USE_INSTASHAP": True,
        "SELECTED_SECTOR": ["时间分类", "有色金属"],
        "MANUAL_SYMBOLS": ["au888", "ag888"],
        "BEGIN_TIME": "2025-06-01",
        "END_TIME": "2025-08-31",
        "BEGIN_TIME_TEST": "2025-09-01",
        "END_TIME_TEST": "2025-12-31",
        "BEGIN_TIME_NOW": "2026-01-01",
        "SYMBOL_CYCLE": "15分钟",
        "Y_PERIOD": 1,
        "quantiles": 5,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
