from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import uuid
import time
import io
import os
import json
import base64
import sys
from datetime import datetime
import threading
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SINGLE_FACTOR_AVAILABLE = None
MULTI_FACTOR_AVAILABLE = None
MINING_AVAILABLE = None

def check_dependencies():
    global SINGLE_FACTOR_AVAILABLE, MULTI_FACTOR_AVAILABLE, MINING_AVAILABLE
    import importlib.util
    
    def module_exists(module_name):
        spec = importlib.util.find_spec(module_name)
        return spec is not None
    
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


class SingleFactorParams(BaseParams):
    formula: str = Field(default="RANK(WR(2), BOLL_UPPER(24, 2.26))", description="因子表达式")


class MultiFactorParams(BaseParams):
    formula: List[str] = Field(default=["RANK(WR(2), BOLL_UPPER(24, 2.26))"], description="因子表达式列表")
    use_lightgbm: bool = Field(default=True, description="是否使用LightGBM模型")
    use_elastic_net: bool = Field(default=True, description="是否使用Elastic Net模型")
    use_instashap: bool = Field(default=True, description="是否使用InstaSHAP模型")


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
    def __init__(self):
        self.outputs = []
        
    def write(self, text):
        if text.strip():
            self.outputs.append(str(text))
            
    def flush(self):
        pass
        
    def get_output(self):
        return "\n".join(self.outputs)


def run_single_factor_task(task_id: str, params: SingleFactorParams):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        use_mock = not SINGLE_FACTOR_AVAILABLE
        
        if use_mock:
            print("警告: single_factor_analysis 模块不可用，使用模拟数据进行演示")
            SYMBOLS = params.symbols if params.symbols else ["au888", "ag888", "cu888"]
            features = ["open", "close", "high", "low", "volume", "open_interest"]
            
            df_list, _ = generate_mock_df_list(SYMBOLS, n_timepoints=500, start_date=params.begin_time)
            df_list_test, _ = generate_mock_df_list(SYMBOLS, n_timepoints=200, start_date=params.begin_time_test)
            df_list_now, _ = generate_mock_df_list(SYMBOLS, n_timepoints=100, start_date=params.begin_time_now or "2026-01-01")
            
            df = pd.concat(df_list, ignore_index=True)
            df_test = pd.concat(df_list_test, ignore_index=True)
            df_now = pd.concat(df_list_now, ignore_index=True) if df_list_now else pd.DataFrame()
        else:
            import single_factor_analysis as sfa
            from single_factor_analysis import (
                get_symbols_by_sector as sfa_get_symbols,
                get_futures_data as sfa_get_data,
                futuers_sectors as SFA_FUTURES_SECTORS,
            )
            
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
                for symbol in SYMBOLS:
                    data = sfa_get_data(symbol, params.begin_time_now, None, params.symbol_cycle)
                    if len(data) > 0:
                        df_list_now.append(data)
            
            df_now = pd.concat(df_list_now, ignore_index=True) if df_list_now else pd.DataFrame()
        
        start_time = time.time()
        update_task(task_id, status="running", message="正在初始化...")
        
        capture = OutputCapture()
        old_stdout = sys.stdout
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
            import single_factor_analysis as sfa
            my_cls = sfa.My
            exprs = sfa.eval_factors(params.formula, my_cls, X_dict, features)
            
            factor = exprs["factor"]
            
            analysis_train = sfa.panel_to_long_factor_df(factor, y, pivoted, "训练集")
            
            analysis_test = None
            if X_dict_test is not None:
                exprs_test = sfa.eval_factors(params.formula, my_cls, X_dict_test, features)
                factor_test = exprs_test["factor"]
                analysis_test = sfa.panel_to_long_factor_df(factor_test, y_test, pivoted_test, "测试集")
            
            analysis_now = None
            if X_dict_now is not None:
                exprs_now = sfa.eval_factors(params.formula, my_cls, X_dict_now, features)
                factor_now = exprs_now["factor"]
                analysis_now = sfa.panel_to_long_factor_df(factor_now, y_now, pivoted_now, "验证集")
            
            ic_train = sfa.calc_ic_stats(factor, y)
            ic_test = sfa.calc_ic_stats(factor_test, y_test) if analysis_test is not None else None
            ic_now = sfa.calc_ic_stats(factor_now, y_now) if analysis_now is not None else None
        
        update_task(task_id, message="正在计算IC统计...", progress=70)
        
        ic_train = sfa.calc_ic_stats(factor, y)
        ic_test = sfa.calc_ic_stats(factor_test, y_test) if analysis_test is not None else None
        ic_now = sfa.calc_ic_stats(factor_now, y_now) if analysis_now is not None else None
        
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
        
        if analysis_train is not None:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            if use_mock:
                axes[0, 0].plot(np.cumsum(np.random.randn(100)))
                axes[0, 0].set_title('训练集 IC序列 (Mock)')
                axes[0, 0].axhline(0, color='r', linestyle='--')
                
                axes[0, 1].bar(range(5), np.random.randn(5))
                axes[0, 1].set_title('训练集 分位数收益 (Mock)')
                
                axes[1, 0].plot(np.cumsum(np.random.randn(100)))
                axes[1, 0].set_title('训练集 多空累计收益 (Mock)')
                
                axes[1, 1].hist(np.random.randn(1000), bins=50)
                axes[1, 1].set_title('训练集 因子值分布 (Mock)')
            else:
                ic_s = sfa.ic_curve_from_long(analysis_train)
                if not ic_s.empty:
                    axes[0, 0].plot(ic_s.values)
                    axes[0, 0].set_title('训练集 IC序列')
                    axes[0, 0].axhline(0, color='r', linestyle='--')
                
                qret, ls = sfa.build_quantile_report(analysis_train, quantiles=params.quantiles)
                if not qret.empty:
                    qret.mean(axis=1).plot(kind='bar', ax=axes[0, 1])
                    axes[0, 1].set_title('训练集 分位数收益')
                
                if not ls.empty:
                    ls.fillna(0).cumsum().plot(ax=axes[1, 0])
                    axes[1, 0].set_title('训练集 多空累计收益')
                
                axes[1, 1].hist(analysis_train['factor'].dropna(), bins=50)
                axes[1, 1].set_title('训练集 因子值分布')
            
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            chart_base64 = base64.b64encode(buf.read()).decode()
            charts.append({"name": "训练集分析", "image": chart_base64})
            plt.close()
        
        if analysis_test is not None:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            if use_mock:
                axes[0, 0].plot(np.cumsum(np.random.randn(100)))
                axes[0, 0].set_title('测试集 IC序列 (Mock)')
                axes[0, 0].axhline(0, color='r', linestyle='--')
                
                axes[0, 1].bar(range(5), np.random.randn(5))
                axes[0, 1].set_title('测试集 分位数收益 (Mock)')
                
                axes[1, 0].plot(np.cumsum(np.random.randn(100)))
                axes[1, 0].set_title('测试集 多空累计收益 (Mock)')
                
                axes[1, 1].hist(np.random.randn(1000), bins=50)
                axes[1, 1].set_title('测试集 因子值分布 (Mock)')
            else:
                ic_s = sfa.ic_curve_from_long(analysis_test)
                if not ic_s.empty:
                    axes[0, 0].plot(ic_s.values)
                    axes[0, 0].set_title('测试集 IC序列')
                    axes[0, 0].axhline(0, color='r', linestyle='--')
                
                qret, ls = sfa.build_quantile_report(analysis_test, quantiles=params.quantiles)
                if not qret.empty:
                    qret.mean(axis=1).plot(kind='bar', ax=axes[0, 1])
                    axes[0, 1].set_title('测试集 分位数收益')
                
                if not ls.empty:
                    ls.fillna(0).cumsum().plot(ax=axes[1, 0])
                    axes[1, 0].set_title('测试集 多空累计收益')
                
                axes[1, 1].hist(analysis_test['factor'].dropna(), bins=50)
                axes[1, 1].set_title('测试集 因子值分布')
            
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            chart_base64 = base64.b64encode(buf.read()).decode()
            charts.append({"name": "测试集分析", "image": chart_base64})
            plt.close()
        
        if analysis_now is not None:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            if use_mock:
                axes[0, 0].plot(np.cumsum(np.random.randn(100)))
                axes[0, 0].set_title('验证集 IC序列 (Mock)')
                axes[0, 0].axhline(0, color='r', linestyle='--')
                
                axes[0, 1].bar(range(5), np.random.randn(5))
                axes[0, 1].set_title('验证集 分位数收益 (Mock)')
                
                axes[1, 0].plot(np.cumsum(np.random.randn(100)))
                axes[1, 0].set_title('验证集 多空累计收益 (Mock)')
                
                axes[1, 1].hist(np.random.randn(1000), bins=50)
                axes[1, 1].set_title('验证集 因子值分布 (Mock)')
            else:
                ic_s = sfa.ic_curve_from_long(analysis_now)
                if not ic_s.empty:
                    axes[0, 0].plot(ic_s.values)
                    axes[0, 0].set_title('验证集 IC序列')
                    axes[0, 0].axhline(0, color='r', linestyle='--')
                
                qret, ls = sfa.build_quantile_report(analysis_now, quantiles=params.quantiles)
                if not qret.empty:
                    qret.mean(axis=1).plot(kind='bar', ax=axes[0, 1])
                    axes[0, 1].set_title('验证集 分位数收益')
                
                if not ls.empty:
                    ls.fillna(0).cumsum().plot(ax=axes[1, 0])
                    axes[1, 0].set_title('验证集 多空累计收益')
                
                axes[1, 1].hist(analysis_now['factor'].dropna(), bins=50)
                axes[1, 1].set_title('验证集 因子值分布')
            
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            chart_base64 = base64.b64encode(buf.read()).decode()
            charts.append({"name": "验证集分析", "image": chart_base64})
            plt.close()
        
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
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        use_mock = not MULTI_FACTOR_AVAILABLE
        
        if use_mock:
            print("警告: multi_factor_analysis 模块不可用，使用模拟数据进行演示")
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
            import multi_factor_analysis as mfa
            
            mfa.SELECTED_SECTOR = params.selected_sector
            mfa.MANUAL_SYMBOLS = params.symbols if params.symbols else ["au888", "ag888"]
            mfa.BEGIN_TIME = params.begin_time
            mfa.END_TIME = params.end_time
            mfa.BEGIN_TIME_TEST = params.begin_time_test
            mfa.END_TIME_TEST = params.end_time_test
            mfa.BEGIN_TIME_NOW = params.begin_time_now
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
                for symbol in SYMBOLS:
                    data = mfa.get_futures_data(symbol, mfa.BEGIN_TIME_NOW, None, mfa.SYMBOL_CYCLE)
                    if len(data) > 0:
                        df_list_now.append(data)
            
            df_now = pd.concat(df_list_now, ignore_index=True) if df_list_now else pd.DataFrame()
            
            target = 'future_return'
            features = ['open', 'close', 'high', 'low', 'volume', 'open_interest']
            formulas = mfa.formula
        
        start_time = time.time()
        update_task(task_id, status="running", message="正在初始化...")
        
        capture = OutputCapture()
        old_stdout = sys.stdout
        sys.stdout = capture
        
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
        
        X_dict_now = None
        y_now = None
        pivoted_now = {}
        if params.begin_time_now:
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
            
            exprs = mfa.eval_factors(expr, my_cls, X_dict, features)
            factor = exprs["factor"]
            factor_arrays.append(factor)
            
            if X_dict_test is not None:
                exprs_test = mfa.eval_factors(expr, my_cls, X_dict_test, features)
                factor_arrays_test.append(exprs_test["factor"])
            
            if X_dict_now is not None:
                exprs_now = mfa.eval_factors(expr, my_cls, X_dict_now, features)
                factor_arrays_now.append(exprs_now["factor"])
        
        print(f"\n所有因子评估完成，共 {len(factor_arrays)} 个因子")
        
        update_task(task_id, message="正在进行分层分析...", progress=50)
        
        ic_stats_list = []
        
        for i, (factor, expr) in enumerate(zip(factor_arrays, mfa.formula)):
            factor_train = mfa.panel_to_long_factor_df(factor, y, pivoted, f"因子{i+1}")
            ic_train = mfa.calc_ic_stats(factor, y)
            
            ic_stats_list.append({
                "因子": f"因子{i+1}",
                "表达式": expr,
                "IC均值": ic_train.get("ic_mean", 0) if ic_train else 0,
                "ICIR": ic_train.get("icir", 0) if ic_train else 0,
            })
            
            print(f"\n因子 {i+1} IC统计:")
            print(f"  IC均值: {ic_train.get('ic_mean', 0):.4f}" if ic_train else "  无数据")
            print(f"  ICIR: {ic_train.get('icir', 0):.4f}" if ic_train else "  无数据")
        
        summary_df = pd.DataFrame(ic_stats_list)
        print("\n因子IC汇总:")
        print(summary_df)
        
        update_task(task_id, message="正在生成图表...", progress=70)
        
        charts = []
        
        fig = plt.figure(figsize=(16, 10))
        names = [f"F{i+1}" for i in range(len(mfa.formula))]
        ic_means = [s["IC均值"] for s in ic_stats_list]
        ic_stds = [s.get("IC标准差", 0.1) for s in ic_stats_list]
        irs = [s["ICIR"] for s in ic_stats_list]
        
        x = np.arange(len(names))
        width = 0.25
        
        ax1 = fig.add_subplot(221)
        ax1.bar(x - width, ic_means, width, label='IC均值', color='steelblue')
        ax1.set_ylabel('IC均值')
        ax1.set_title('IC均值')
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=45, ha='right')
        ax1.axhline(0.2, c='orange', ls='--')
        ax1.axhline(0.5, c='red', ls='--')
        
        ax2 = fig.add_subplot(222)
        ax2.bar(x, ic_stds, width, label='IC标准差', color='coral')
        ax2.set_ylabel('IC标准差')
        ax2.set_title('IC标准差')
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, rotation=45, ha='right')
        
        ax3 = fig.add_subplot(223)
        ax3.bar(x + width, irs, width, label='IR', color='seagreen')
        ax3.set_ylabel('IR')
        ax3.set_title('IR')
        ax3.set_xticks(x)
        ax3.set_xticklabels(names, rotation=45, ha='right')
        
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        chart_base64 = base64.b64encode(buf.read()).decode()
        charts.append({"name": "因子IC分析", "image": chart_base64})
        plt.close()
        
        factor_corr = np.corrcoef(np.array(factor_arrays)[:, :, 0])
        fig, ax = plt.subplots(figsize=(9, 7))
        im = ax.imshow(factor_corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(np.arange(len(names)))
        ax.set_yticks(np.arange(len(names)))
        ax.set_xticklabels(names)
        ax.set_yticklabels(names)
        plt.colorbar(im, ax=ax)
        ax.set_title("因子相关性")
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        chart_base64 = base64.b64encode(buf.read()).decode()
        charts.append({"name": "因子相关性", "image": chart_base64})
        plt.close()
        
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
        output_text = capture.get_output() + f"\n错误: {str(e)}\n{traceback.format_exc()}" if 'capture' in locals() else f"错误: {str(e)}\n{traceback.format_exc()}"
        
        update_task(
            task_id,
            status="failed",
            message=f"任务失败: {str(e)}",
            result={"output_text": output_text, "error": str(e)},
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
    import factor_mining as fm
    from factor_mining import get_futures_data, SYMBOL_CYCLE_MAP

    target = "future_return"

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
        data = get_futures_data(symbol, begin_time, end_time, symbol_cycle)
        if len(data) > 0:
            df_list.append(data)
    
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
        import factor_mining as fm
        from factor_mining import (
            GeneticProgrammer,
            FunctionSet,
            fitness_func,
            get_symbols_by_sector as fm_get_symbols,
            FUTURES_SECTORS as FM_FUTURES_SECTORS,
        )
        
        update_task(task_id, status="running", message="正在初始化数据...")
        
        is_mock_mode = params.use_mock_data or not MINING_AVAILABLE
        
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
                symbols = fm_get_symbols(params.selected_sector[0] if isinstance(params.selected_sector, list) else params.selected_sector, FM_FUTURES_SECTORS, None)
            
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
                    "FITNESS_W_TRAIN": params.fitness_w_train,
                    "FITNESS_W_TEST": params.fitness_w_test,
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
            return {"message": "任务已删除"}
    raise HTTPException(status_code=404, detail="任务不存在")


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
        "formula": ["RANK(WR(2), BOLL_UPPER(24, 2.26))"],
        "use_lightgbm": True,
        "use_elastic_net": True,
        "use_instashap": True,
        "selected_sector": ["时间分类", "有色金属"],
        "symbols": ["au888", "ag888"],
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
        "selected_sector": ["有色金属"],
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
