from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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
from datetime import datetime
import threading
import warnings
warnings.filterwarnings('ignore')

try:
    from single_factor_analysis import (
        get_symbols_by_sector,
        get_futures_data,
        futuers_sectors as FUTURES_SECTORS,
        My,
        eval_factors,
        calc_ic_stats,
        build_quantile_report,
        ic_curve_from_long,
        convert_formula,
    )
    DATA_SOURCE_AVAILABLE = True
except ImportError as e:
    print(f"警告: 无法导入 single_factor_analysis: {e}")
    DATA_SOURCE_AVAILABLE = False
    get_symbols_by_sector = None
    get_futures_data = None
    FUTURES_SECTORS = {}

app = FastAPI(
    title="单因子分析API",
    description="单因子分析服务 - 支持因子表达式评估和IC/IR分析",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = "single_factor_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class FactorAnalysisParams(BaseModel):
    formula: str = Field(default="RANK(WR(2), BOLL_UPPER(24, 2.26))", description="因子表达式")
    SELECTED_SECTOR: List[str] = Field(default=["时间分类", "有色金属"], description="选择的行业板块")
    BEGIN_TIME: str = Field(default="2025-06-01", description="训练集开始时间")
    END_TIME: str = Field(default="2025-08-31", description="训练集结束时间")
    BEGIN_TIME_TEST: str = Field(default="2025-09-01", description="测试集开始时间")
    END_TIME_TEST: str = Field(default="2025-12-31", description="测试集结束时间")
    BEGIN_TIME_NOW: str = Field(default="2026-01-01", description="当前/验证集开始时间")
    SYMBOL_CYCLE: str = Field(default="15分钟", description="数据周期")
    Y_PERIOD: int = Field(default=1, ge=1, le=20, description="预测周期")
    symbols: Optional[List[str]] = Field(default=None, description="指定的期货代码列表")
    quantiles: int = Field(default=5, ge=2, le=10, description="分位数数量")


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
    def __init__(self):
        self.outputs = []
        
    def write(self, text):
        if text.strip():
            self.outputs.append(str(text))
            
    def flush(self):
        pass
        
    def get_output(self):
        return "\n".join(self.outputs)


def run_analysis_task(task_id: str, params: FactorAnalysisParams):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import single_factor_analysis as sfa
        
        start_time = time.time()
        update_task(task_id, status="running", message="正在初始化...")
        
        capture = OutputCapture()
        old_stdout = sys.stdout
        sys.stdout = capture
        
        update_task(task_id, message="正在获取合约列表...", progress=10)
        
        if params.symbols:
            SYMBOLS = params.symbols
        else:
            SYMBOLS = get_symbols_by_sector(params.SELECTED_SECTOR, FUTURES_SECTORS, None)
        
        print(f"当前使用的合约列表（共 {len(SYMBOLS)} 个合约）:")
        print(SYMBOLS)
        
        update_task(task_id, message="正在加载数据...", progress=20)
        
        features = ["open", "close", "high", "low", "volume", "open_interest"]
        
        df_list = []
        for symbol in SYMBOLS:
            data = get_futures_data(symbol, params.BEGIN_TIME, params.END_TIME, params.SYMBOL_CYCLE)
            if len(data) > 0:
                df_list.append(data)
        
        if len(df_list) == 0:
            raise ValueError("训练集没有有效数据")
        
        df = pd.concat(df_list, ignore_index=True)
        
        for col in features:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        df.drop_duplicates(subset=["asset", "date"], keep="last", inplace=True)
        df.sort_values(["asset", "date"], ignore_index=True, inplace=True)
        
        print(f"训练集数据预处理完成，共 {len(df)} 条记录")
        
        df = df.sort_values(["asset", "date"]).reset_index(drop=True)
        df["future_return"] = df.groupby("asset")["close"].shift(-params.Y_PERIOD) / df["close"] - 1
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
        
        update_task(task_id, message="正在加载测试数据...", progress=30)
        
        df_list_test = []
        for symbol in SYMBOLS:
            data = get_futures_data(symbol, params.BEGIN_TIME_TEST, params.END_TIME_TEST, params.SYMBOL_CYCLE)
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
        df_test["future_return"] = df_test.groupby("asset")["close"].shift(-params.Y_PERIOD) / df_test["close"] - 1
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
        
        df_list_now = []
        for symbol in SYMBOLS:
            data = get_futures_data(symbol, params.BEGIN_TIME_NOW, None, params.SYMBOL_CYCLE)
            if len(data) > 0:
                df_list_now.append(data)
        
        if len(df_list_now) > 0:
            df_now = pd.concat(df_list_now, ignore_index=True)
            for col in features:
                df_now[col] = pd.to_numeric(df_now[col], errors="coerce")
            df_now.drop_duplicates(subset=["asset", "date"], keep="last", inplace=True)
            df_now.sort_values(["asset", "date"], ignore_index=True, inplace=True)
            
            df_now = df_now.sort_values(["asset", "date"]).reset_index(drop=True)
            df_now["future_return"] = df_now.groupby("asset")["close"].shift(-params.Y_PERIOD) / df_now["close"] - 1
            df_now = df_now.dropna(subset=["future_return"]).reset_index(drop=True)
            
            df_now["time"] = df_now["date"]
            df_now["code"] = df_now["asset"]
            
            pivoted_now = {}
            for col in features + [target]:
                try:
                    pivoted_now[col] = df_now.pivot(index="time", columns="code", values=col)
                except:
                    unique_dates = pd.Series(df_now["time"]).unique()
                    unique_codes = pd.Series(df_now["code"]).unique()
                    pivoted_now[col] = pd.DataFrame(0, index=unique_dates, columns=unique_codes)
            
            X_dict_now = {f: pivoted_now[f].values for f in features}
            y_now = pivoted_now[target].values
        else:
            X_dict_now = None
            y_now = None
        
        update_task(task_id, message="正在评估因子...", progress=50)
        
        print(f"\n因子表达式: {params.formula}")
        print(f"品种合约: {SYMBOLS}")
        print(f"预测周期: {params.Y_PERIOD}")
        
        my_cls = sfa.My
        exprs = sfa.eval_factors(params.formula, my_cls, X_dict, features)
        
        factor = exprs["factor"]
        
        analysis_train = sfa.panel_to_long_factor_df(factor, y, pivoted, "训练集")
        
        if X_dict_test is not None:
            exprs_test = sfa.eval_factors(params.formula, my_cls, X_dict_test, features)
            factor_test = exprs_test["factor"]
            analysis_test = sfa.panel_to_long_factor_df(factor_test, y_test, pivoted_test, "测试集")
        else:
            analysis_test = None
        
        if X_dict_now is not None:
            exprs_now = sfa.eval_factors(params.formula, my_cls, X_dict_now, features)
            factor_now = exprs_now["factor"]
            analysis_now = sfa.panel_to_long_factor_df(factor_now, y_now, pivoted_now, "验证集")
        else:
            analysis_now = None
        
        update_task(task_id, message="正在计算IC统计...", progress=70)
        
        ic_train = calc_ic_stats(factor, y)
        ic_test = calc_ic_stats(factor_test, y_test) if analysis_test is not None else None
        ic_now = calc_ic_stats(factor_now, y_now) if analysis_now is not None else None
        
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
            
            ic_s = ic_curve_from_long(analysis_train)
            if not ic_s.empty:
                axes[0, 0].plot(ic_s.values)
                axes[0, 0].set_title('训练集 IC序列')
                axes[0, 0].axhline(0, color='r', linestyle='--')
            
            qret, ls = build_quantile_report(analysis_train, quantiles=params.quantiles)
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
            
            ic_s = ic_curve_from_long(analysis_test)
            if not ic_s.empty:
                axes[0, 0].plot(ic_s.values)
                axes[0, 0].set_title('测试集 IC序列')
                axes[0, 0].axhline(0, color='r', linestyle='--')
            
            qret, ls = build_quantile_report(analysis_test, quantiles=params.quantiles)
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
            
            ic_s = ic_curve_from_long(analysis_now)
            if not ic_s.empty:
                axes[0, 0].plot(ic_s.values)
                axes[0, 0].set_title('验证集 IC序列')
                axes[0, 0].axhline(0, color='r', linestyle='--')
            
            qret, ls = build_quantile_report(analysis_now, quantiles=params.quantiles)
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
        output_text = capture.get_output() + f"\n错误: {str(e)}\n{traceback.format_exc()}"
        
        update_task(
            task_id,
            status="failed",
            message=f"任务失败: {str(e)}",
            result={"output_text": output_text, "error": str(e)},
            completed_at=datetime.now().isoformat()
        )


import sys


@app.get("/")
async def root():
    return {
        "message": "单因子分析API服务", 
        "version": "1.0.0",
        "data_source_available": DATA_SOURCE_AVAILABLE
    }


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "data_source_available": DATA_SOURCE_AVAILABLE
    }


@app.post("/api/analysis/start", response_model=TaskStatus)
async def start_analysis(params: FactorAnalysisParams, background_tasks: BackgroundTasks):
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
        "formula": "RANK(WR(2), BOLL_UPPER(24, 2.26))",
        "SELECTED_SECTOR": ["时间分类", "有色金属"],
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
    uvicorn.run(app, host="0.0.0.0", port=8001)
