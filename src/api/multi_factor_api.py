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
from datetime import datetime
import threading
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = "multi_factor_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class MultiFactorAnalysisParams(BaseModel):
    formula: List[str] = Field(default=["RANK(WR(2), BOLL_UPPER(24, 2.26))"], description="因子表达式列表")
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
    def __init__(self):
        self.outputs = []
        
    def write(self, text):
        if text.strip():
            self.outputs.append(str(text))
            
    def flush(self):
        pass
        
    def get_output(self):
        return "\n".join(self.outputs)


def run_analysis_task(task_id: str, params: MultiFactorAnalysisParams):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import sys
        import multi_factor_analysis as mfa
        
        start_time = time.time()
        update_task(task_id, status="running", message="正在初始化...")
        
        capture = OutputCapture()
        old_stdout = sys.stdout
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
        "formula": ["RANK(WR(2), BOLL_UPPER(24, 2.26))"],
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
