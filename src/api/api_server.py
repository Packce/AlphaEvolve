from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import uuid
import time
from datetime import datetime, timedelta
import threading
import warnings
warnings.filterwarnings('ignore')

try:
    from factor_mining import (
        GeneticProgrammer, 
        FunctionSet, 
        fitness_func,
        get_symbols_by_sector,
        FUTURES_SECTORS,
    )
    DATA_SOURCE_AVAILABLE = True
except ImportError as e:
    print(f"警告: 无法导入 factor_mining: {e}")
    print("将使用模拟数据模式运行")
    DATA_SOURCE_AVAILABLE = False
    GeneticProgrammer = None
    FunctionSet = None
    fitness_func = None
    get_symbols_by_sector = None
    FUTURES_SECTORS = {}


def create_mock_genetic_programmer(features):
    """创建模拟的遗传编程器类和函数（用于测试）"""
    
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
            """解析表达式字符串为Node树"""
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

app = FastAPI(
    title="因子挖掘API",
    description="遗传编程因子挖掘服务 - 提供因子挖掘、因子评估等功能",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MiningParams(BaseModel):
    # 数据范围参数
    begin_time: str = Field(default="2025-06-01", description="训练集开始时间，格式：YYYY-MM-DD")
    end_time: str = Field(default="2025-08-31", description="训练集结束时间，格式：YYYY-MM-DD")
    begin_time_test: str = Field(default="2025-09-01", description="测试集开始时间，格式：YYYY-MM-DD")
    end_time_test: str = Field(default="2025-12-31", description="测试集结束时间，格式：YYYY-MM-DD")
    symbol_cycle: str = Field(default="15分钟", description="数据周期")
    selected_sector: str = Field(default="有色金属", description="选择的行业板块")
    symbols: Optional[List[str]] = Field(default=None, description="指定的期货代码列表")
    # 遗传编程参数
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


def generate_mock_data(symbols: List[str], features: List[str], n_timepoints: int = 500):
    """生成模拟的期货数据"""
    np.random.seed(42)
    n_contracts = len(symbols)
    
    X_dict = {}
    for feature in features:
        X_dict[feature] = np.random.randn(n_timepoints, n_contracts) * 100 + 1000
    
    close_prices = X_dict["close"]
    future_return = np.roll(close_prices, -20, axis=0) / close_prices - 1
    future_return = np.nan_to_num(future_return, nan=0.0)
    
    return X_dict, future_return


def prepare_data(
    symbols: List[str], 
    features: List[str], 
    ic_period: int, 
    train: bool = True,
    begin_time: str = None,
    end_time: str = None,
    symbol_cycle: str = "15分钟"
):
    """准备训练或测试数据"""
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


def run_mining_task(task_id: str, params: MiningParams):
    try:
        update_task(task_id, status="running", message="正在初始化数据...")
        
        if params.use_mock_data or not DATA_SOURCE_AVAILABLE:
            symbols = params.symbols if params.symbols else ["au888", "ag888", "cu888"]
            features = params.features
            
            update_task(task_id, message="正在生成模拟数据...")
            X_dict, y = generate_mock_data(symbols, features, n_timepoints=500)
            X_dict_test, y_test = generate_mock_data(symbols, features, n_timepoints=200)
        else:
            if params.symbols:
                symbols = params.symbols
            else:
                from factor_mining import get_symbols_by_sector
                symbols = get_symbols_by_sector(params.selected_sector, FUTURES_SECTORS, None)
            
            features = params.features
            
            update_task(task_id, message="正在加载训练数据...")
            X_dict, y = prepare_data(
                symbols, features, params.ic_period, train=True,
                begin_time=params.begin_time,
                end_time=params.end_time,
                symbol_cycle=params.symbol_cycle
            )
            
            update_task(task_id, message="正在加载测试数据...")
            X_dict_test, y_test = prepare_data(
                symbols, features, params.ic_period, train=False,
                begin_time=params.begin_time_test,
                end_time=params.end_time_test,
                symbol_cycle=params.symbol_cycle
            )
        
        update_task(task_id, message="正在初始化遗传编程器...")
        
        is_mock_mode = params.use_mock_data or not DATA_SOURCE_AVAILABLE
        
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
            completed_at=datetime.now().isoformat()
        )


@app.get("/")
async def root():
    return {
        "message": "因子挖掘API服务", 
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


@app.post("/api/mining/start", response_model=TaskStatus)
async def start_mining(params: MiningParams, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    create_task(task_id, "任务已创建，等待执行")
    
    background_tasks.add_task(run_mining_task, task_id, params)
    
    return get_task(task_id)


@app.get("/api/mining/status/{task_id}", response_model=TaskStatus)
async def get_mining_status(task_id: str):
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return task


@app.get("/api/mining/result/{task_id}")
async def get_mining_result(task_id: str):
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    if task.status != "completed":
        return {
            "status": task.status,
            "progress": task.progress,
            "message": task.message,
        }
    
    return {
        "status": task.status,
        "result": task.result,
        "elapsed_time": task.elapsed_time,
    }


@app.get("/api/mining/list")
async def list_mining_tasks(limit: int = 10):
    with tasks_lock:
        tasks = list(tasks_storage.values())
    tasks.sort(key=lambda x: x.created_at, reverse=True)
    return {"tasks": tasks[:limit], "total": len(tasks)}


@app.delete("/api/mining/task/{task_id}")
async def delete_task(task_id: str):
    with tasks_lock:
        if task_id in tasks_storage:
            del tasks_storage[task_id]
            return {"message": "任务已删除"}
    raise HTTPException(status_code=404, detail="任务不存在")


@app.get("/api/config/default")
async def get_default_config():
    return {
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
        "use_mock_data": False,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
