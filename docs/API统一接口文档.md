# 量化分析统一API文档

## 概述

`unified_api.py` 是一个整合了单因子分析、多因子分析和因子挖掘三大功能的统一API服务。基于 FastAPI 框架开发，提供异步任务处理、实时进度追踪和统一的接口规范。

### 核心功能

| 模块 | 功能描述 |
|------|----------|
| 单因子分析 | 对单个因子表达式进行 IC/IR 分析、分位数收益分析、多空策略回测 |
| 多因子分析 | 对多个因子进行相关性分析、因子合成、模型分析 |
| 因子挖掘 | 基于遗传编程自动挖掘有效因子表达式 |

### 技术特性

- **异步任务处理**：使用 FastAPI BackgroundTasks 实现后台任务执行
- **实时进度追踪**：支持任务进度实时查询
- **统一错误处理**：规范的 HTTP 错误码和错误信息
- **跨域支持**：默认支持 CORS，方便前端集成

---

## 快速开始

### 1. 安装依赖

```bash
pip install fastapi uvicorn pandas numpy matplotlib pydantic
```

### 2. 启动服务

```bash
cd D:\PythonCode\AlphaEvovle\back_end
D:\软件\期魔方\coder\python3116\python.exe d:\PythonCode\AlphaEvovle\back_end\src\api\unified_api.py
``` 

服务默认运行在 `http://0.0.0.0:8000`

### 3. 访问 API 文档

启动服务后，访问以下地址查看交互式 API 文档：

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## 基础接口

### 健康检查

**GET** `/api/health`

检查服务状态和各模块可用性。

**响应示例：**
```json
{
  "status": "healthy",
  "timestamp": "2026-04-08T10:30:00.000000",
  "services": {
    "single_factor": true,
    "multi_factor": true,
    "mining": true
  }
}
```

### 服务信息

**GET** `/`

获取服务基本信息。

**响应示例：**
```json
{
  "message": "量化分析统一API服务",
  "version": "2.0.0",
  "services": {
    "single_factor": true,
    "multi_factor": true,
    "mining": true
  }
}
```

---

## 单因子分析接口

### 启动分析任务

**POST** `/api/single-factor/start`

对单个因子表达式进行完整的 IC/IR 分析。

**请求体 (SingleFactorParams)：**

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `formula` | string | `RANK(WR(2), BOLL_UPPER(24, 2.26))` | 因子表达式 |
| `selected_sector` | List[str] | `["时间分类", "有色金属"]` | 行业板块 |
| `begin_time` | string | `2025-06-01` | 训练集开始时间 |
| `end_time` | string | `2025-08-31` | 训练集结束时间 |
| `begin_time_test` | string | `2025-09-01` | 测试集开始时间 |
| `end_time_test` | string | `2025-12-31` | 测试集结束时间 |
| `begin_time_now` | string | `2026-01-01` | 验证集开始时间 |
| `symbol_cycle` | string | `15分钟` | 数据周期 |
| `y_period` | int | `1` | 预测周期 (1-20) |
| `symbols` | List[str] | `null` | 指定期货代码列表 |
| `quantiles` | int | `5` | 分位数数量 (2-10) |

**请求示例：**
```json
{
  "formula": "RANK(WR(2), BOLL_UPPER(24, 2.26))",
  "selected_sector": ["时间分类", "有色金属"],
  "begin_time": "2025-06-01",
  "end_time": "2025-08-31",
  "begin_time_test": "2025-09-01",
  "end_time_test": "2025-12-31",
  "begin_time_now": "2026-01-01",
  "symbol_cycle": "15分钟",
  "y_period": 1,
  "quantiles": 5
}
```

**响应示例：**
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "task_type": "single_factor",
  "status": "pending",
  "progress": 0.0,
  "message": "任务已创建，等待执行",
  "result": null,
  "created_at": "2026-04-08T10:30:00.000000",
  "started_at": null,
  "completed_at": null,
  "elapsed_time": null
}
```

### 获取默认配置

**GET** `/api/config/single-factor/default`

获取单因子分析的默认参数配置。

---

## 多因子分析接口

### 启动分析任务

**POST** `/api/multi-factor/start`

对多个因子进行相关性分析和因子合成。

**请求体 (MultiFactorParams)：**

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `formula` | List[str] | `["RANK(WR(2), BOLL_UPPER(24, 2.26))"]` | 因子表达式列表 |
| `use_lightgbm` | bool | `true` | 是否使用 LightGBM 模型 |
| `use_elastic_net` | bool | `true` | 是否使用 Elastic Net 模型 |
| `use_instashap` | bool | `true` | 是否使用 InstaSHAP 模型 |
| `selected_sector` | List[str] | `["时间分类", "有色金属"]` | 行业板块 |
| `symbols` | List[str] | `null` | 指定期货代码列表 |
| `begin_time` | string | `2025-06-01` | 训练集开始时间 |
| `end_time` | string | `2025-08-31` | 训练集结束时间 |
| `begin_time_test` | string | `2025-09-01` | 测试集开始时间 |
| `end_time_test` | string | `2025-12-31` | 测试集结束时间 |
| `begin_time_now` | string | `2026-01-01` | 验证集开始时间 |
| `symbol_cycle` | string | `15分钟` | 数据周期 |
| `y_period` | int | `1` | 预测周期 (1-20) |
| `quantiles` | int | `5` | 分位数数量 (2-10) |

**请求示例：**
```json
{
  "formula": [
    "RANK(WR(2), BOLL_UPPER(24, 2.26))",
    "RANK(MA(5), MA(20))"
  ],
  "use_lightgbm": true,
  "use_elastic_net": true,
  "use_instashap": true,
  "selected_sector": ["时间分类", "有色金属"],
  "begin_time": "2025-06-01",
  "end_time": "2025-08-31",
  "symbol_cycle": "15分钟",
  "y_period": 1
}
```

### 获取默认配置

**GET** `/api/config/multi-factor/default`

---

## 因子挖掘接口

### 启动挖掘任务

**POST** `/api/mining/start`

基于遗传编程算法自动挖掘有效因子。

**请求体 (MiningParams)：**

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| **数据参数** ||||
| `begin_time` | string | `2025-06-01` | 训练集开始时间 |
| `end_time` | string | `2025-08-31` | 训练集结束时间 |
| `begin_time_test` | string | `2025-09-01` | 测试集开始时间 |
| `end_time_test` | string | `2025-12-31` | 测试集结束时间 |
| `symbol_cycle` | string | `15分钟` | 数据周期 |
| `selected_sector` | List[str] | `["有色金属"]` | 行业板块 |
| `symbols` | List[str] | `null` | 指定期货代码列表 |
| `features` | List[str] | 全部 | 使用的特征列表 |
| `ic_period` | int | `20` | IC计算周期 |
| **遗传编程参数** ||||
| `generations` | int | `15` | 进化代数 (1-100) |
| `population_size` | int | `120` | 种群规模 (10-1000) |
| `tournament_size` | int | `4` | 锦标赛规模 (2-20) |
| `n_components` | int | `5` | 保留的最优个体数量 (1-20) |
| `hall_of_fame` | int | `6` | 精英保留数量 (1-20) |
| `ts_window` | int | `20` | 时间窗口范围 (5-250) |
| `const_range` | tuple | `(-2, 120)` | 常数范围 |
| `p_crossover` | float | `0.30` | 交叉概率 (0-1) |
| `p_subtree_mutation` | float | `0.30` | 子树变异概率 (0-1) |
| `p_hoist_mutation` | float | `0.10` | 提升变异概率 (0-1) |
| `p_point_mutation` | float | `0.20` | 点变异概率 (0-1) |
| `immigration_rate` | float | `0.20` | 每代注入随机个体比例 (0-1) |
| `parsimony_coefficient` | float | `0.002` | 简约系数 (0-1) |
| `init_depth` | tuple | `(3, 8)` | 初始树深度范围 |
| `suit_size` | tuple | `(4, 14)` | 表达树节点数上下界 |
| `stagnation_threshold` | int | `6` | 停滞检测阈值 |
| `min_improvement` | float | `0.001` | 最小显著提升阈值 |
| `max_restarts` | int | `3` | 最大自动重启次数 |
| `max_program_size` | int | `24` | 进化过程最大节点数限制 |
| `max_best_program_size` | int | `24` | 最终最优个体最大节点数限制 |
| `ic_objective` | string | `max` | IC优化方向 (`max` 或 `min`) |
| `fitness_w_train` | float | `0.6` | 训练集适应度权重 (0-1) |
| `fitness_w_test` | float | `0.4` | 测试集适应度权重 (0-1) |
| `random_state` | int | `null` | 随机数种子 |
| `use_mock_data` | bool | `false` | 是否使用模拟数据（仅用于测试） |

**请求示例：**
```json
{
  "begin_time": "2025-06-01",
  "end_time": "2025-08-31",
  "begin_time_test": "2025-09-01",
  "end_time_test": "2025-12-31",
  "symbol_cycle": "15分钟",
  "selected_sector": ["有色金属"],
  "generations": 15,
  "population_size": 120,
  "features": ["open", "close", "high", "low", "volume", "open_interest"],
  "ic_period": 20,
  "use_mock_data": false
}
```

### 获取默认配置

**GET** `/api/config/mining/default`

---

## 统一任务管理接口

### 查询任务状态

**GET** `/api/task/status/{task_id}`

获取指定任务的当前状态。

**路径参数：**
- `task_id` (string, required): 任务ID

**响应示例：**
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "task_type": "single_factor",
  "status": "running",
  "progress": 50.0,
  "message": "正在评估因子...",
  "result": null,
  "created_at": "2026-04-08T10:30:00.000000",
  "started_at": "2026-04-08T10:30:01.000000",
  "completed_at": null,
  "elapsed_time": 5.23
}
```

### 获取任务结果

**GET** `/api/task/result/{task_id}`

获取已完成任务的结果。

**响应示例（单因子分析）：**
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "task_type": "single_factor",
  "status": "completed",
  "result": {
    "formula": "RANK(WR(2), BOLL_UPPER(24, 2.26))",
    "symbols": ["au888", "ag888", "cu888"],
    "ic_stats": [
      {
        "数据集": "训练集",
        "IC均值": 0.035,
        "IC标准差": 0.12,
        "ICIR": 0.29,
        "IC>0比例": 0.65,
        "Rank IC均值": 0.042,
        "Rank ICIR": 0.35
      }
    ],
    "charts": [
      {
        "name": "训练集分析",
        "image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAA..."
      }
    ],
    "output_text": "当前使用的合约列表（共 3 个合约）:\n...",
    "elapsed_time": 12.5
  },
  "elapsed_time": 12.5
}
```

### 任务列表

**GET** `/api/task/list`

获取任务列表。

**查询参数：**
- `limit` (int, optional): 返回任务数量，默认 10
- `task_type` (string, optional): 按任务类型过滤 (`single_factor`, `multi_factor`, `mining`)

**响应示例：**
```json
{
  "tasks": [...],
  "total": 25
}
```

### 删除任务

**DELETE** `/api/task/{task_id}`

删除指定任务。

**响应示例：**
```json
{
  "message": "任务已删除"
}
```

---

## 任务状态说明

| 状态 | 描述 |
|------|------|
| `pending` | 任务已创建，等待执行 |
| `running` | 任务正在执行中 |
| `completed` | 任务已完成 |
| `failed` | 任务执行失败 |

---

## 完整使用流程

### 1. 启动任务

```bash
curl -X POST "http://localhost:8000/api/single-factor/start" \
  -H "Content-Type: application/json" \
  -d '{"formula": "RANK(WR(2), BOLL_UPPER(24, 2.26))", "selected_sector": ["有色金属"]}'
```

### 2. 获取任务ID

响应中的 `task_id` 即为任务ID。

### 3. 轮询任务状态

```bash
curl "http://localhost:8000/api/task/status/{task_id}"
```

### 4. 获取结果

```bash
curl "http://localhost:8000/api/task/result/{task_id}"
```

---

## 错误处理

### HTTP 状态码

| 状态码 | 说明 |
|--------|------|
| `200` | 请求成功 |
| `404` | 资源不存在（任务ID不存在） |
| `503` | 服务不可用（对应模块未安装） |

### 错误响应示例

```json
{
  "detail": "任务不存在"
}
```

---

## 数据说明

### 数据来源

API 依赖以下数据模块：
- `single_factor_analysis` - 单因子分析模块
- `multi_factor_analysis` - 多因子分析模块
- `factor_mining` - 因子挖掘模块

确保这些模块已正确安装并配置了本地数据源。

### 数据周期

支持的期货数据周期：
- `1分钟`、`5分钟`、`15分钟`、`30分钟`
- `1小时`、`2小时`、`4小时`
- `日线`、`周线`

### 因子表达式

因子表达式支持多种函数组合，示例：
```
RANK(WR(2), BOLL_UPPER(24, 2.26))
RANK(MA(5), MA(20))
DIV(CLOSE, OPEN)
MUL(SUB(CLOSE, OPEN), VOLUME)
```

---

## 启动端口

默认情况下，服务运行在端口 `8000`。

如需修改，编辑 `unified_api.py` 最后一行：

```python
uvicorn.run(app, host="0.0.0.0", port=你想要的端口)
```

---

## 注意事项

1. **数据依赖**：确保 `qmf_data` 等数据源模块已正确配置
2. **长时间任务**：因子挖掘等任务可能耗时较长，请耐心等待
3. **并发限制**：建议同一时间只运行一个分析任务
4. **内存使用**：大参数（如大种群规模）会占用较多内存
