# 智能旅行规划助手 - 实验三

基于 Qwen3 的 Agent 应用，包含自动化测试与可视化功能。

## 功能特性

- **智能对话**：使用通义千问 API 进行自然语言交互
- **工具调用**：支持天气查询、景点搜索、汇率换算、攻略检索
- **记忆管理**：SQLite 存储用户偏好和历史对话
- **RAG 知识库**：基于 FAISS + BGE 嵌入的旅行文档检索
- **自动重试**：低置信度回答自动补充信息重试
- **自动化测试**：预设测试用例，评估工具调用准确率
- **可视化报告**：生成性能指标图表

## 环境要求

```bash
pip install dashscope langchain langchain-community langchain-huggingface faiss-cpu matplotlib numpy
```

## API 配置

设置 DashScope API Key：

```python
DASHSCOPE_API_KEY = "your-api-key"
DASHSCOPE_MODEL = "qwen-plus"
```

## 项目结构

```
.
├── ceshi.py              # 主程序
├── travel_docs/          # 示例旅行文档
│   ├── tokyo_guide.md
│   ├── paris_guide.md
│   └── ny_guide.md
├── travel_vector_db/     # FAISS 向量库（自动生成）
├── travel_memory.db      # SQLite 记忆库（自动生成）
├── experiment_results.png # 结果图表（运行后生成）
└── metrics.json         # 指标数据（运行后生成）
```

## 可用工具

| 工具名称              | 功能         | 参数                       |
| --------------------- | ------------ | -------------------------- |
| `get_weather`         | 查询城市天气 | city                       |
| `search_attractions`  | 搜索景点     | city, keyword(可选)        |
| `search_travel_guide` | 检索旅行攻略 | query                      |
| `get_exchange_rate`   | 货币汇率换算 | from_currency, to_currency |
| `get_current_time`    | 获取当前时间 | 无                         |

## 运行测试

```bash
python ceshi.py
```

## 测试用例

| 问题                                                 | 预期工具                                | 类型 |
| ---------------------------------------------------- | --------------------------------------- | ---- |
| 我想去东京旅行，有什么推荐？                         | search_travel_guide                     | 单步 |
| 巴黎现在天气怎么样？                                 | get_weather                             | 单步 |
| 纽约有哪些必去景点？                                 | search_attractions                      | 单步 |
| 100美元能换多少人民币？                              | get_exchange_rate                       | 单步 |
| 帮我规划一个3天北京行程，预算3000元                  | search_travel_guide, search_attractions | 多跳 |
| 我想去日本玩，但只有5000预算，能去东京吗？天气如何？ | search_travel_guide, get_weather        | 多跳 |

## 输出指标

- **平均置信度**：回答质量评分
- **平均响应时间**：处理耗时
- **工具调用准确率**：是否正确使用预期工具
- **多跳问题解决率**：复杂问题的处理能力
- **重试次数统计**

## 输出文件

- `experiment_results.png`：包含4个子图的性能可视化
- `metrics.json`：JSON 格式的详细指标数据

## 置信度评估规则

- 包含景点/天气/推荐等关键词 +0.2
- 包含数字/温度/货币符号 +0.2
- 包含"不确定"/"可能" -0.2
- 阈值：0.7（低于此值触发重试）

## 注意事项

1. 需要有效的 DashScope API Key
2. 首次运行会自动构建向量库（需下载 BGE 模型）
3. 支持中文图表显示（已配置 SimHei 字体）