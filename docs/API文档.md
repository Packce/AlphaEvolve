# API接口文档

本文档包含三个API服务的接口说明：
1. **因子挖掘API** (`api_server.py`) - 基于遗传编程的因子挖掘，服务端口 **8000**
2. **单因子分析API** (`single_factor_api.py`) - 单因子分析评估，服务端口 **8001**
3. **多因子分析API** (`multi_factor_api.py`) - 多因子合成和模型分析，服务端口 **8002**

---

## 1. 概述

因子挖掘API是一个基于遗传编程的金融因子挖掘服务。
单因子分析API提供对单个因子表达式的评估分析功能。
多因子分析API支持多因子合成和模型分析。

- **因子挖掘API地址**: `http://your-server:8000`
- **单因子分析API地址**: `http://your-server:8001`
- **多因子分析API地址**: `http://your-server:8002`
- **版本**: 2.0.0
- **交互式文档**: 
  - 因子挖掘: `http://your-server:8000/docs`
  - 单因子分析: `http://your-server:8001/docs`
  - 多因子分析: `http://your-server:8002/docs`

---

## 2. 启动服务

```bash
# 安装依赖
pip install -r requirements_api.txt

# 启动因子挖掘服务 (端口 8000)
python api_server.py

# 启动单因子分析服务 (端口 8001)
python single_factor_api.py

# 启动多因子分析服务 (端口 8002)
python multi_factor_api.py
```

服务默认运行在：
- 因子挖掘API: `http://0.0.0.0:8000`
- 单因子分析API: `http://0.0.0.0:8001`
- 多因子分析API: `http://0.0.0.0:8002`

---

## 3. 接口列表

### 3.1 健康检查

检查服务是否正常运行。

**因子挖掘API请求**

```http
GET /api/health
```

**单因子分析API请求**

```http
GET http://localhost:8001/api/health
```

**响应**

```json
{
  "status": "healthy",
  "timestamp": "2026-04-06T21:42:43.989568",
  "data_source_available": true
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| status | string | 服务状态 |
| timestamp | string | 时间戳 |
| data_source_available | boolean | 数据源是否可用 |

---

### 3.2 获取默认配置

获取因子挖掘的默认参数配置。

**因子挖掘API请求**

```http
GET /api/config/default
```

**单因子分析API请求**

```http
GET http://localhost:8001/api/config/default
```

**因子挖掘API响应**

```json
{
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
  "use_mock_data": false
}
```

**单因子分析API响应**

```json
{
  "formula": "RANK(WR(2), BOLL_UPPER(24, 2.26))",
  "SELECTED_SECTOR": ["时间分类", "有色金属"],
  "BEGIN_TIME": "2025-06-01",
  "END_TIME": "2025-08-31",
  "BEGIN_TIME_TEST": "2025-09-01",
  "END_TIME_TEST": "2025-12-31",
  "BEGIN_TIME_NOW": "2026-01-01",
  "SYMBOL_CYCLE": "15分钟",
  "Y_PERIOD": 1,
  "quantiles": 5
}
```

---

### 3.3 启动挖掘任务 (因子挖掘API)

启动一个因子挖掘任务。

**请求**

```http
POST /api/mining/start
Content-Type: application/json
```

**请求体参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| generations | integer | 否 | 15 | 进化代数，范围: 1-100 |
| population_size | integer | 否 | 120 | 种群规模，范围: 10-1000 |
| tournament_size | integer | 否 | 4 | 锦标赛规模，范围: 2-20 |
| n_components | integer | 否 | 5 | 保留的最优个体数量，范围: 1-20 |
| hall_of_fame | integer | 否 | 6 | 精英保留数量，范围: 1-20 |
| ts_window | integer | 否 | 20 | 时间窗口范围，范围: 5-250 |
| const_range | array | 否 | [-2, 120] | 常数范围 |
| p_crossover | float | 否 | 0.30 | 交叉概率，范围: 0-1 |
| p_subtree_mutation | float | 否 | 0.30 | 子树变异概率，范围: 0-1 |
| p_hoist_mutation | float | 否 | 0.10 | 提升变异概率，范围: 0-1 |
| p_point_mutation | float | 否 | 0.20 | 点变异概率，范围: 0-1 |
| immigration_rate | float | 否 | 0.20 | 每代注入随机个体比例，范围: 0-1 |
| parsimony_coefficient | float | 否 | 0.002 | 简约系数，范围: 0-1 |
| init_depth | array | 否 | [3, 8] | 初始树深度范围 |
| suit_size | array | 否 | [4, 14] | 表达树节点数上下界 |
| stagnation_threshold | integer | 否 | 6 | 停滞检测阈值，范围: 1-50 |
| min_improvement | float | 否 | 0.001 | 最小显著提升阈值，范围: 0-1 |
| max_restarts | integer | 否 | 3 | 最大自动重启次数，范围: 0-10 |
| max_program_size | integer | 否 | 24 | 进化过程最大节点数限制，范围: 5-100 |
| max_best_program_size | integer | 否 | 24 | 最终最优个体最大节点数限制，范围: 5-100 |
| ic_objective | string | 否 | "max" | IC优化方向: "max" 或 "min" |
| features | array | 否 | 见下方 | 使用的特征列表 |
| symbols | array | 否 | null | 指定的期货代码列表，如: ["au888", "ag888"] |
| selected_sector | string | 否 | "黑色金属" | 选择的行业板块 |
| ic_period | integer | 否 | 20 | IC计算周期，范围: 5-250 |
| fitness_w_train | float | 否 | 0.6 | 训练集适应度权重，范围: 0-1 |
| fitness_w_test | float | 否 | 0.4 | 测试集适应度权重，范围: 0-1 |
| random_state | integer | 否 | 42 | 随机数种子 |
| use_mock_data | boolean | 否 | false | 是否使用模拟数据（仅用于测试） |

**features 默认值**

```json
["open", "close", "high", "low", "volume", "open_interest"]
```

**请求示例**

```json
{
  "generations": 15,
  "population_size": 120,
  "ts_window": 20,
  "use_mock_data": true,
  "random_state": 42
}
```

**响应**

```json
{
  "task_id": "5b9841bf-ef40-4b6e-bf86-75eefdd70e68",
  "status": "pending",
  "progress": 0.0,
  "message": "任务已创建，等待执行",
  "result": null,
  "created_at": "2026-04-06T21:42:53.510268",
  "started_at": null,
  "completed_at": null,
  "elapsed_time": null
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| task_id | string | 任务ID，用于后续查询 |
| status | string | 任务状态: pending/running/completed/failed |
| progress | float | 进度百分比 (0-100) |
| message | string | 状态消息 |
| created_at | string | 创建时间 |

---

### 3.4 启动分析任务 (单因子分析API)

启动一个单因子分析任务。

**请求**

```http
POST http://localhost:8001/api/analysis/start
Content-Type: application/json
```

**请求体参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| formula | string | 是 | - | 因子表达式 |
| SELECTED_SECTOR | array | 否 | ["时间分类", "有色金属"] | 行业板块选择 |
| BEGIN_TIME | string | 否 | "2025-06-01" | 训练集开始时间 |
| END_TIME | string | 否 | "2025-08-31" | 训练集结束时间 |
| BEGIN_TIME_TEST | string | 否 | "2025-09-01" | 测试集开始时间 |
| END_TIME_TEST | string | 否 | "2025-12-31" | 测试集结束时间 |
| BEGIN_TIME_NOW | string | 否 | "2026-01-01" | 验证集开始时间 |
| SYMBOL_CYCLE | string | 否 | "15分钟" | 数据周期 |
| Y_PERIOD | int | 否 | 1 | 预测周期 |
| symbols | array | 否 | null | 指定期货代码 |
| quantiles | int | 否 | 5 | 分位数数量 |

**请求示例**

```json
{
  "formula": "RANK(WR(2), BOLL_UPPER(24, 2.26))",
  "SELECTED_SECTOR": ["时间分类", "有色金属"],
  "BEGIN_TIME": "2025-06-01",
  "END_TIME": "2025-08-31",
  "BEGIN_TIME_TEST": "2025-09-01",
  "END_TIME_TEST": "2025-12-31",
  "BEGIN_TIME_NOW": "2026-01-01",
  "SYMBOL_CYCLE": "15分钟",
  "Y_PERIOD": 1,
  "quantiles": 5
}
```

**响应**

```json
{
  "task_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "pending",
  "progress": 0.0,
  "message": "任务已创建，等待执行",
  "result": null,
  "created_at": "2026-04-07T12:00:00.000000",
  "started_at": null,
  "completed_at": null,
  "elapsed_time": null
}
```

---

### 3.5 查询任务状态 (因子挖掘API)

查询指定任务的状态。

**请求**

```http
GET /api/mining/status/{task_id}
```

**路径参数**

| 参数 | 类型 | 说明 |
|------|------|------|
| task_id | string | 任务ID |

**响应**

```json
{
  "task_id": "5b9841bf-ef40-4b6e-bf86-75eefdd70e68",
  "status": "running",
  "progress": 50.0,
  "message": "正在进化第 8/15 代 | 平均适应度: 0.023456 | 最优适应度: 0.034567",
  "result": null,
  "created_at": "2026-04-06T21:42:53.510268",
  "started_at": "2026-04-06T21:42:53.515561",
  "completed_at": null,
  "elapsed_time": 15.234
}
```

---

### 3.6 查询任务状态 (单因子分析API)

查询指定分析任务的状态。

**请求**

```http
GET http://localhost:8001/api/analysis/status/{task_id}
```

**响应**

```json
{
  "task_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "running",
  "progress": 50.0,
  "message": "正在评估因子...",
  "result": null,
  "created_at": "2026-04-07T12:00:00.000000",
  "started_at": "2026-04-07T12:00:00.100000",
  "completed_at": null,
  "elapsed_time": 5.5
}
```

**status 状态说明**
- `pending`: 等待执行
- `running`: 执行中
- `completed`: 完成
- `failed`: 失败

---

### 3.7 获取任务结果 (因子挖掘API)

获取已完成任务的结果。

**请求**

```http
GET /api/mining/result/{task_id}
```

**路径参数**

| 参数 | 类型 | 说明 |
|------|------|------|
| task_id | string | 任务ID |

**响应**

```json
{
  "status": "completed",
  "result": {
    "best_factors": [
      {
        "rank": 1,
        "expression": "MUL(CLOSE, VOLUME)",
        "depth": 2,
        "size": 3,
        "fitness": 0.48719581456328864,
        "train_ic": -0.48719581456328864,
        "train_ir": -0.38975665165063095,
        "test_ic": -0.43847623310695977,
        "test_ir": -0.341037070194302,
        "valid_ts": 480,
        "total_ts": 500
      },
      {
        "rank": 2,
        "expression": "ADD(CLOSE, VOLUME)",
        "depth": 2,
        "size": 3,
        "fitness": 0.4896889611171462,
        "train_ic": -0.4896889611171462,
        "train_ir": -0.391751168893717,
        "test_ic": -0.4407200650054316,
        "test_ir": -0.34278227278200235,
        "valid_ts": 480,
        "total_ts": 500
      }
    ],
    "total_generations": 15,
    "population_size": 120,
    "symbols_count": 3,
    "features": ["open", "close", "high", "low", "volume", "open_interest"]
  },
  "elapsed_time": 125.5
}
```

**result.best_factors 字段说明**

| 字段 | 类型 | 说明 |
|------|------|------|
| rank | integer | 排名 |
| expression | string | 因子表达式 |
| depth | integer | 树深度 |
| size | integer | 节点数 |
| fitness | float | 适应度值 |
| train_ic | float | 训练集IC |
| train_ir | float | 训练集IR |
| test_ic | float | 测试集IC |
| test_ir | float | 测试集IR |
| valid_ts | integer | 有效时间点数 |
| total_ts | integer | 总时间点数 |

---

### 3.8 获取分析结果 (单因子分析API)

获取已完成分析任务的结果。

**请求**

```http
GET http://localhost:8001/api/analysis/result/{task_id}
```

**响应**

```json
{
  "status": "completed",
  "result": {
    "formula": "RANK(WR(2), BOLL_UPPER(24, 2.26))",
    "symbols": ["au888", "ag888", "cu888"],
    "ic_stats": [
      {
        "数据集": "训练集",
        "IC均值": 0.025,
        "IC标准差": 0.15,
        "ICIR": 0.167,
        "IC>0比例": 0.52,
        "Rank IC均值": 0.032,
        "Rank ICIR": 0.21
      },
      {
        "数据集": "测试集",
        "IC均值": 0.018,
        "IC标准差": 0.16,
        "ICIR": 0.112,
        "IC>0比例": 0.48,
        "Rank IC均值": 0.025,
        "Rank ICIR": 0.15
      }
    ],
    "charts": [
      {
        "name": "训练集分析",
        "image": "iVBORw0KGgoAAAANSUhEUgAA..."
      }
    ],
    "output_text": "当前使用的合约列表（共 3 个合约）:\n['au888', 'ag888', 'cu888']\n训练集数据预处理完成，共 15000 条记录\n...",
    "elapsed_time": 125.5
  }
}
```

**result 字段说明**

| 字段 | 类型 | 说明 |
|------|------|------|
| formula | string | 因子表达式 |
| symbols | array | 使用的合约列表 |
| ic_stats | array | IC统计信息 |
| charts | array | 图表数据（Base64编码） |
| output_text | string | 控制台输出文本 |
| elapsed_time | float | 耗时（秒） |

**ic_stats 字段说明**

| 字段 | 说明 |
|------|------|
| 数据集 | 数据集名称 |
| IC均值 | IC平均值 |
| IC标准差 | IC标准差 |
| ICIR | IC信息比率 |
| IC>0比例 | IC为正的比例 |
| Rank IC均值 | Rank IC平均值 |
| Rank ICIR | Rank IC信息比率 |

**charts 字段说明**

| 字段 | 说明 |
|------|------|
| name | 图表名称 |
| image | Base64编码的PNG图片 |

---

### 3.9 任务列表 (因子挖掘API)

获取所有任务列表。

**请求**

```http
GET /api/mining/list?limit=10
```

**查询参数**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| limit | integer | 10 | 返回任务数量限制 |

**响应**

```json
{
  "tasks": [
    {
      "task_id": "5b9841bf-ef40-4b6e-bf86-75eefdd70e68",
      "status": "completed",
      "progress": 100.0,
      "message": "因子挖掘完成",
      "result": {...},
      "created_at": "2026-04-06T21:42:53.510268"
    }
  ],
  "total": 5
}
```

---

### 3.10 任务列表 (单因子分析API)

获取所有分析任务列表。

**请求**

```http
GET http://localhost:8001/api/analysis/list?limit=10
```

---

### 3.11 删除任务 (因子挖掘API)

删除指定任务。

**请求**

```http
DELETE /api/mining/task/{task_id}
```

**路径参数**

| 参数 | 类型 | 说明 |
|------|------|------|
| task_id | string | 任务ID |

**响应**

```json
{
  "message": "任务已删除"
}
```

---

### 3.12 删除任务 (单因子分析API)

删除指定分析任务。

**请求**

```http
DELETE http://localhost:8001/api/analysis/task/{task_id}
```

---

## 3.13 多因子分析API健康检查

检查多因子分析服务是否正常运行。

**请求**

```http
GET http://localhost:8002/api/health
```

**响应**

```json
{
  "status": "healthy",
  "timestamp": "2026-04-07T20:23:50.108968"
}
```

---

## 3.14 多因子分析API获取默认配置

获取多因子分析的默认参数配置。

**请求**

```http
GET http://localhost:8002/api/config/default
```

**响应**

```json
{
  "formula": ["RANK(WR(2), BOLL_UPPER(24, 2.26))"],
  "USE_LIGHTGBM": true,
  "USE_ELASTIC_NET": true,
  "USE_INSTASHAP": true,
  "SELECTED_SECTOR": ["时间分类", "有色金属"],
  "MANUAL_SYMBOLS": ["au888", "ag888"],
  "BEGIN_TIME": "2025-06-01",
  "END_TIME": "2025-08-31",
  "BEGIN_TIME_TEST": "2025-09-01",
  "END_TIME_TEST": "2025-12-31",
  "BEGIN_TIME_NOW": "2026-01-01",
  "SYMBOL_CYCLE": "15分钟",
  "Y_PERIOD": 1,
  "quantiles": 5
}
```

---

## 3.15 多因子分析API启动任务

启动一个多因子分析任务。

**请求**

```http
POST http://localhost:8002/api/analysis/start
Content-Type: application/json
```

**请求参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| formula | array | 是 | - | 因子表达式列表 |
| USE_LIGHTGBM | boolean | 否 | true | 是否使用LightGBM模型 |
| USE_ELASTIC_NET | boolean | 否 | true | 是否使用Elastic Net模型 |
| USE_INSTASHAP | boolean | 否 | true | 是否使用InstaSHAP模型 |
| SELECTED_SECTOR | array | 否 | null | 选择的行业板块 |
| MANUAL_SYMBOLS | array | 否 | null | 手动指定的合约列表 |
| BEGIN_TIME | string | 否 | "2025-06-01" | 训练集开始时间 |
| END_TIME | string | 否 | "2025-08-31" | 训练集结束时间 |
| BEGIN_TIME_TEST | string | 否 | "2025-09-01" | 测试集开始时间 |
| END_TIME_TEST | string | 否 | "2025-12-31" | 测试集结束时间 |
| BEGIN_TIME_NOW | string | 否 | "2026-01-01" | 验证集开始时间 |
| SYMBOL_CYCLE | string | 否 | "15分钟" | 数据周期 |
| Y_PERIOD | int | 否 | 1 | 预测周期 |
| quantiles | int | 否 | 5 | 分位数数量 |

**请求示例**

```json
{
  "formula": ["RANK(WR(2), BOLL_UPPER(24, 2.26))", "MA(close, 10)"],
  "USE_LIGHTGBM": true,
  "USE_ELASTIC_NET": true,
  "USE_INSTASHAP": false,
  "SELECTED_SECTOR": ["时间分类", "有色金属"],
  "BEGIN_TIME": "2025-06-01",
  "END_TIME": "2025-08-31",
  "BEGIN_TIME_TEST": "2025-09-01",
  "END_TIME_TEST": "2025-12-31",
  "BEGIN_TIME_NOW": "2026-01-01",
  "SYMBOL_CYCLE": "15分钟",
  "Y_PERIOD": 1,
  "quantiles": 5
}
```

**响应**

```json
{
  "task_id": "c1d2e3f4-5678-90ab-cdef-1234567890ab",
  "status": "pending",
  "progress": 0.0,
  "message": "任务已创建，等待执行",
  "result": null,
  "created_at": "2026-04-07T20:30:00.000000",
  "started_at": null,
  "completed_at": null,
  "elapsed_time": null
}
```

---

## 3.16 多因子分析API查询状态

查询指定分析任务的状态。

**请求**

```http
GET http://localhost:8002/api/analysis/status/{task_id}
```

**响应**

```json
{
  "task_id": "c1d2e3f4-5678-90ab-cdef-1234567890ab",
  "status": "running",
  "progress": 50.0,
  "message": "正在评估因子...",
  "result": null,
  "created_at": "2026-04-07T20:30:00.000000",
  "started_at": "2026-04-07T20:30:00.100000",
  "completed_at": null,
  "elapsed_time": 5.5
}
```

---

## 3.17 多因子分析API获取结果

获取已完成分析任务的结果。

**请求**

```http
GET http://localhost:8002/api/analysis/result/{task_id}
```

**响应**

```json
{
  "status": "completed",
  "result": {
    "formula": ["RANK(WR(2), BOLL_UPPER(24, 2.26))"],
    "symbols": ["au888", "ag888"],
    "ic_stats": [
      {
        "因子": "因子1",
        "表达式": "RANK(WR(2), BOLL_UPPER(24, 2.26))",
        "IC均值": 0.025,
        "ICIR": 0.167
      }
    ],
    "charts": [
      {
        "name": "因子IC分析",
        "image": "iVBORw0KGgoAAAANSUhEUgAA..."
      },
      {
        "name": "因子相关性",
        "image": "iVBORw0KGgoAAAANSUhEUgAA..."
      }
    ],
    "output_text": "当前使用的合约列表（共 2 个合约）:\n['au888', 'ag888']\n...",
    "elapsed_time": 125.5
  }
}
```

**result 字段说明**

| 字段 | 类型 | 说明 |
|------|------|------|
| formula | array | 因子表达式列表 |
| symbols | array | 使用的合约列表 |
| ic_stats | array | IC统计信息 |
| charts | array | 图表数据（Base64编码） |
| output_text | string | 控制台输出文本 |
| elapsed_time | float | 耗时（秒） |

---

## 3.18 多因子分析API任务列表

获取所有分析任务列表。

**请求**

```http
GET http://localhost:8002/api/analysis/list?limit=10
```

---

## 3.19 多因子分析API删除任务

删除指定分析任务。

**请求**

```http
DELETE http://localhost:8002/api/analysis/task/{task_id}
```

---

## 4. 使用示例

### 4.1 因子挖掘 Python 示例

```python
import requests

BASE_URL = "http://localhost:8000"

# 1. 启动任务
params = {
    "generations": 15,
    "population_size": 120,
    "use_mock_data": True,
    "random_state": 42
}
response = requests.post(f"{BASE_URL}/api/mining/start", json=params)
task_id = response.json()["task_id"]
print(f"Task ID: {task_id}")

# 2. 查询状态
import time
while True:
    response = requests.get(f"{BASE_URL}/api/mining/status/{task_id}")
    status = response.json()
    print(f"Status: {status['status']}, Progress: {status['progress']}%")
    if status["status"] in ["completed", "failed"]:
        break
    time.sleep(2)

# 3. 获取结果
response = requests.get(f"{BASE_URL}/api/mining/result/{task_id}")
result = response.json()
print(result)
```

### 4.2 单因子分析 Python 示例

```python
import requests
import time
import base64

BASE_URL = "http://localhost:8001"

# 1. 启动分析任务
params = {
    "formula": "RANK(WR(2), BOLL_UPPER(24, 2.26))",
    "SELECTED_SECTOR": ["时间分类", "有色金属"],
    "BEGIN_TIME": "2025-06-01",
    "END_TIME": "2025-08-31",
    "BEGIN_TIME_TEST": "2025-09-01",
    "END_TIME_TEST": "2025-12-31",
    "BEGIN_TIME_NOW": "2026-01-01",
    "SYMBOL_CYCLE": "15分钟",
    "Y_PERIOD": 1,
    "quantiles": 5
}

response = requests.post(f"{BASE_URL}/api/analysis/start", json=params)
task_id = response.json()["task_id"]
print(f"Task ID: {task_id}")

# 2. 轮询等待结果
while True:
    response = requests.get(f"{BASE_URL}/api/analysis/status/{task_id}")
    status = response.json()
    print(f"Status: {status['status']}, Progress: {status['progress']}%")
    
    if status["status"] in ["completed", "failed"]:
        break
    time.sleep(2)

# 3. 获取结果
response = requests.get(f"{BASE_URL}/api/analysis/result/{task_id}")
result = response.json()["result"]

# 打印控制台输出
print(result["output_text"])

# 保存图表
for chart in result["charts"]:
    image_data = base64.b64decode(chart["image"])
    with open(f"{chart['name']}.png", "wb") as f:
        f.write(image_data)
    print(f"已保存: {chart['name']}.png")

# 打印IC统计
for ic_stat in result["ic_stats"]:
    print(ic_stat)
```

### 4.3 因子挖掘 JavaScript 示例

```javascript
const BASE_URL = 'http://localhost:8000';

// 1. 启动任务
async function startMining() {
  const response = await fetch(`${BASE_URL}/api/mining/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      generations: 15,
      population_size: 120,
      use_mock_data: true,
      random_state: 42
    })
  });
  const data = await response.json();
  return data.task_id;
}

// 2. 查询状态
async function checkStatus(taskId) {
  const response = await fetch(`${BASE_URL}/api/mining/status/${taskId}`);
  return await response.json();
}

// 3. 获取结果
async function getResult(taskId) {
  const response = await fetch(`${BASE_URL}/api/mining/result/${taskId}`);
  return await response.json();
}

// 使用
const taskId = await startMining();
console.log('Task ID:', taskId);
```

### 4.4 单因子分析 JavaScript 示例

```javascript
const BASE_URL = 'http://localhost:8001';

async function runAnalysis(params) {
  // 1. 启动任务
  const response = await fetch(`${BASE_URL}/api/analysis/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params)
  });
  const { task_id } = await response.json();
  console.log('Task ID:', task_id);

  // 2. 轮询状态
  while (true) {
    const statusResponse = await fetch(`${BASE_URL}/api/analysis/status/${task_id}`);
    const status = await statusResponse.json();
    console.log(`Status: ${status.status}, Progress: ${status.progress}%`);
    
    if (status.status === 'completed' || status.status === 'failed') {
      break;
    }
    await new Promise(r => setTimeout(r, 2000));
  }

  // 3. 获取结果
  const resultResponse = await fetch(`${BASE_URL}/api/analysis/result/${task_id}`);
  const result = await resultResponse.json();
  
  console.log('Output:', result.result.output_text);
  
  // 显示图表
  result.result.charts.forEach(chart => {
    const img = document.createElement('img');
    img.src = `data:image/png;base64,${chart.image}`;
    document.body.appendChild(img);
  });
}

runAnalysis({
  formula: "RANK(WR(2), BOLL_UPPER(24, 2.26))",
  SELECTED_SECTOR: ["时间分类", "有色金属"],
  BEGIN_TIME: "2025-06-01",
  END_TIME: "2025-08-31",
  BEGIN_TIME_TEST": "2025-09-01",
  END_TIME_TEST": "2025-12-31",
  BEGIN_TIME_NOW": "2026-01-01",
  SYMBOL_CYCLE: "15分钟",
  Y_PERIOD": 1,
  quantiles: 5
});
```

### 4.5 多因子分析 Python 示例

```python
import requests
import time
import base64

BASE_URL = "http://localhost:8002"

# 1. 启动分析任务
params = {
    "formula": ["RANK(WR(2), BOLL_UPPER(24, 2.26))", "MA(close, 10)"],
    "USE_LIGHTGBM": True,
    "USE_ELASTIC_NET": True,
    "USE_INSTASHAP": False,
    "SELECTED_SECTOR": ["时间分类", "有色金属"],
    "BEGIN_TIME": "2025-06-01",
    "END_TIME": "2025-08-31",
    "BEGIN_TIME_TEST": "2025-09-01",
    "END_TIME_TEST": "2025-12-31",
    "BEGIN_TIME_NOW": "2026-01-01",
    "SYMBOL_CYCLE": "15分钟",
    "Y_PERIOD": 1,
    "quantiles": 5
}

response = requests.post(f"{BASE_URL}/api/analysis/start", json=params)
task_id = response.json()["task_id"]
print(f"Task ID: {task_id}")

# 2. 轮询等待结果
while True:
    response = requests.get(f"{BASE_URL}/api/analysis/status/{task_id}")
    status = response.json()
    print(f"Status: {status['status']}, Progress: {status['progress']}%")
    
    if status["status"] in ["completed", "failed"]:
        break
    time.sleep(2)

# 3. 获取结果
response = requests.get(f"{BASE_URL}/api/analysis/result/{task_id}")
result = response.json()["result"]

# 打印控制台输出
print(result["output_text"])

# 保存图表
for chart in result["charts"]:
    image_data = base64.b64decode(chart["image"])
    with open(f"{chart['name']}.png", "wb") as f:
        f.write(image_data)
    print(f"已保存: {chart['name']}.png")

# 打印IC统计
for ic_stat in result["ic_stats"]:
    print(ic_stat)
```

### 4.6 多因子分析 JavaScript 示例

```javascript
const BASE_URL = 'http://localhost:8002';

async function runMultiFactorAnalysis(params) {
  // 1. 启动任务
  const response = await fetch(`${BASE_URL}/api/analysis/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params)
  });
  const { task_id } = await response.json();
  console.log('Task ID:', task_id);

  // 2. 轮询状态
  while (true) {
    const statusResponse = await fetch(`${BASE_URL}/api/analysis/status/${task_id}`);
    const status = await statusResponse.json();
    console.log(`Status: ${status.status}, Progress: ${status.progress}%`);
    
    if (status.status === 'completed' || status.status === 'failed') {
      break;
    }
    await new Promise(r => setTimeout(r, 2000));
  }

  // 3. 获取结果
  const resultResponse = await fetch(`${BASE_URL}/api/analysis/result/${task_id}`);
  const result = await resultResponse.json();
  
  console.log('Output:', result.result.output_text);
  
  // 显示图表
  result.result.charts.forEach(chart => {
    const img = document.createElement('img');
    img.src = `data:image/png;base64,${chart.image}`;
    document.body.appendChild(img);
  });
}

runMultiFactorAnalysis({
  formula: ["RANK(WR(2), BOLL_UPPER(24, 2.26))", "MA(close, 10)"],
  USE_LIGHTGBM: true,
  USE_ELASTIC_NET: true,
  USE_INSTASHAP: false,
  SELECTED_SECTOR: ["时间分类", "有色金属"],
  BEGIN_TIME: "2025-06-01",
  END_TIME: "2025-08-31",
  BEGIN_TIME_TEST": "2025-09-01",
  END_TIME_TEST": "2025-12-31",
  BEGIN_TIME_NOW": "2026-01-01",
  SYMBOL_CYCLE: "15分钟",
  Y_PERIOD: 1,
  quantiles: 5
});
```

---

## 5. 错误响应

### 404 任务不存在

```json
{
  "detail": "任务不存在"
}
```

### 400 参数错误

```json
{
  "detail": "任务失败: 训练集没有有效数据"
}
```

---

## 6. 注意事项

1. 所有任务均为异步执行，启动任务后会立即返回 task_id
2. 需要轮询状态接口获取进度和结果
3. `use_mock_data` 参数仅用于测试，生产环境应设置为 false
4. 真实数据模式需要确保数据源模块可用
5. 单因子分析的图表以 Base64 格式返回，前端需要解码后显示

---

## 7. 字段说明

### IC (Information Coefficient)

信息系数，表示因子预测值与真实收益之间的相关系数。IC越高，因子预测能力越强。

### IR (Information Ratio)

信息比率，IC的均值与标准差之比，反映因子预测能力的稳定性。

### 适应度 (Fitness)

综合考虑训练集和测试集IC的加权得分，用于遗传算法选择优秀个体。

---

## 8. 因子表达式语法 (单因子分析)

单因子分析支持以下函数：

### 基础运算
- `ADD(x, y)` - 加法
- `SUB(x, y)` - 减法
- `MUL(x, y)` - 乘法
- `DIV(x, y)` - 除法

### 技术指标
- `MA(x, d)` - 简单移动平均
- `EMA(x, d)` - 指数移动平均
- `SMA(x, d, m)` - 中国式SMA
- `WMA(x, d)` - 加权移动平均
- `BOLL_UPPER(x, d, k)` - 布林线上轨
- `BOLL_LOWER(x, d, k)` - 布林线下轨
- `WR(n)` - 威廉指标
- `RSI(x, d)` - RSI指标
- `MACD(x, fast, slow, signal)` - MACD

### 排名函数
- `RANK(x)` - 排名
- `TS_RANK(x, d)` - 时序排名

### 其他
- `REF(x, n)` - 引用n天前数据
- `DIFF(x, n)` - 差分
- `HHV(x, d)` - 最高值
- `LLV(x, d)` - 最低值
- `CORR(x, y, d)` - 相关系数

---

*文档版本: 2.0.0*
*更新时间: 2026-04-07*
