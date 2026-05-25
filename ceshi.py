"""
实验三：基于Qwen3的Agent应用 - 智能旅行规划助手（含自动化测试与可视化）
- 支持中文图表
- 纯自动运行模式
"""

import os
import json
import sqlite3
import time
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import numpy as np

# ====== 解决 matplotlib 中文显示问题 ======
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import dashscope
from dashscope import Generation
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ==================== 配置 ====================
DASHSCOPE_API_KEY = "sk-4a2b1cb5cf864435af9b1f7df3a4bfee"
DASHSCOPE_MODEL = "qwen-plus"
VECTOR_DB_PATH = Path("./travel_vector_db")
MEMORY_DB_PATH = Path("./travel_memory.db")
CONFIDENCE_THRESHOLD = 0.7

# 创建示例文档
SAMPLE_DOCS_DIR = Path("./travel_docs")
SAMPLE_DOCS_DIR.mkdir(exist_ok=True)
(SAMPLE_DOCS_DIR / "tokyo_guide.md").write_text("""
# 东京旅行攻略
- 最佳季节：春季（樱花）和秋季（红叶）
- 推荐景点：浅草寺、东京塔、涩谷、迪士尼乐园
- 美食：寿司、拉面、天妇罗
- 交通：购买Suica卡，地铁便利
""", encoding="utf-8")
(SAMPLE_DOCS_DIR / "paris_guide.md").write_text("""
# 巴黎旅行攻略
- 最佳季节：春秋两季，气候宜人
- 推荐景点：埃菲尔铁塔、卢浮宫、圣母院
- 美食：可颂、马卡龙、法式蜗牛
- 注意：热门景点需提前预约
""", encoding="utf-8")
(SAMPLE_DOCS_DIR / "ny_guide.md").write_text("""
# 纽约旅行攻略
- 最佳季节：5-6月或9-10月
- 推荐景点：自由女神像、时代广场、中央公园
- 美食：汉堡、披萨、百吉饼
- 建议：住曼哈顿中城，地铁24小时运行
""", encoding="utf-8")

# ==================== 记忆模块 ====================
class MemoryManager:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(str(db_path))
        self._init_db()
    def _init_db(self):
        self.conn.execute('CREATE TABLE IF NOT EXISTS user_prefs (user_id TEXT PRIMARY KEY, budget INTEGER, duration INTEGER, preferred_theme TEXT, last_city TEXT, updated_at TIMESTAMP)')
        self.conn.execute('CREATE TABLE IF NOT EXISTS history (user_id TEXT, question TEXT, answer TEXT, timestamp TIMESTAMP)')
        self.conn.commit()
    def save_preference(self, user_id, budget=None, duration=None, theme=None, city=None):
        cur = self.conn.execute("SELECT * FROM user_prefs WHERE user_id=?", (user_id,))
        if cur.fetchone():
            self.conn.execute("UPDATE user_prefs SET budget=?, duration=?, preferred_theme=?, last_city=?, updated_at=? WHERE user_id=?", (budget, duration, theme, city, datetime.now(), user_id))
        else:
            self.conn.execute("INSERT INTO user_prefs VALUES (?,?,?,?,?,?)", (user_id, budget, duration, theme, city, datetime.now()))
        self.conn.commit()
    def get_preference(self, user_id):
        cur = self.conn.execute("SELECT budget, duration, preferred_theme, last_city FROM user_prefs WHERE user_id=?", (user_id,))
        row = cur.fetchone()
        return {"budget": row[0], "duration": row[1], "theme": row[2], "last_city": row[3]} if row else {}
    def save_history(self, user_id, question, answer):
        self.conn.execute("INSERT INTO history VALUES (?,?,?,?)", (user_id, question, answer, datetime.now()))
        self.conn.commit()
    def get_recent_history(self, user_id, limit=3):
        rows = self.conn.execute("SELECT question, answer FROM history WHERE user_id=? ORDER BY timestamp DESC LIMIT ?", (user_id, limit)).fetchall()
        return "\n".join([f"用户问：{q}\n助手答：{a}" for q, a in reversed(rows)])

# ==================== RAG知识库 ====================
def build_vectorstore():
    if VECTOR_DB_PATH.exists():
        try:
            embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5")
            return FAISS.load_local(str(VECTOR_DB_PATH), embeddings, allow_dangerous_deserialization=True)
        except:
            pass
    print("构建向量库...")
    docs = []
    for md_file in SAMPLE_DOCS_DIR.glob("*.md"):
        loader = TextLoader(str(md_file), encoding='utf-8')
        docs.extend(loader.load())
    if not docs:
        return None
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(VECTOR_DB_PATH))
    return vectorstore

vectorstore = build_vectorstore()

def search_travel_guide(query: str) -> str:
    if vectorstore is None:
        return "知识库未就绪"
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    docs = retriever.invoke(query)
    if not docs:
        return "未找到相关攻略"
    return "\n".join([f"【{doc.metadata.get('source','')}】\n{doc.page_content}" for doc in docs])

# ==================== 工具函数 ====================
def get_weather(city: str) -> str:
    weather_db = {"东京": "晴天 18°C", "巴黎": "多云 15°C", "纽约": "小雨 12°C", "北京": "晴 25°C", "上海": "阴 22°C"}
    return weather_db.get(city, f"{city}：晴 20°C")

def search_attractions(city: str, keyword: str = "") -> str:
    attractions = {
        "东京": ["浅草寺", "东京塔", "涩谷", "迪士尼"],
        "巴黎": ["埃菲尔铁塔", "卢浮宫", "凯旋门"],
        "纽约": ["自由女神像", "时代广场", "中央公园"],
        "北京": ["故宫", "长城", "天坛"],
        "上海": ["外滩", "东方明珠", "迪士尼"]
    }
    items = attractions.get(city, [f"{city}著名景点"])
    if keyword:
        items = [x for x in items if keyword.lower() in x.lower()]
    return f"{city}景点：\n" + "\n".join(f"- {x}" for x in items)

def get_exchange_rate(from_currency: str, to_currency: str = "CNY") -> str:
    rates = {"USD": 7.25, "EUR": 7.85, "JPY": 0.048, "GBP": 9.10, "CNY": 1}
    from_r = rates.get(from_currency.upper(), 1)
    to_r = rates.get(to_currency.upper(), 1)
    return f"1 {from_currency.upper()} = {from_r/to_r:.2f} {to_currency.upper()}"

def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

TOOLS = [
    {"type": "function", "function": {"name": "get_current_time", "description": "获取当前时间", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "search_travel_guide", "description": "搜索旅行攻略", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "get_weather", "description": "查询天气", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "search_attractions", "description": "查询景点", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "keyword": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "get_exchange_rate", "description": "汇率换算", "parameters": {"type": "object", "properties": {"from_currency": {"type": "string"}, "to_currency": {"type": "string", "default": "CNY"}}, "required": ["from_currency"]}}}
]

TOOL_FUNCTIONS = {
    "get_current_time": get_current_time,
    "search_travel_guide": search_travel_guide,
    "get_weather": get_weather,
    "search_attractions": search_attractions,
    "get_exchange_rate": get_exchange_rate,
}

SYSTEM_PROMPT = """你是旅行规划助手。你必须严格使用工具来回答问题，不要自己编造数据。

可用的工具：
- search_travel_guide(query): 查询攻略（景点、美食、最佳季节）
- get_weather(city): 查询天气
- search_attractions(city, keyword): 查询景点列表
- get_exchange_rate(from_currency, to_currency): 汇率换算

规则：
1. 用户问“XX有什么推荐/攻略” → 调用 search_travel_guide
2. 用户问“XX天气” → 调用 get_weather
3. 用户问“XX景点” → 调用 search_attractions
4. 用户问“XX货币换人民币” → 调用 get_exchange_rate
5. 如果用户同时问多个问题，依次调用对应的工具。

不要直接回答，必须通过工具获取信息后再组织答案。"""

def estimate_confidence(answer: str) -> float:
    score = 0.5
    if any(k in answer for k in ["景点", "天气", "推荐", "攻略", "温度"]):
        score += 0.2
    if any(c.isdigit() for c in answer) or "℃" in answer or "¥" in answer or "=" in answer:
        score += 0.2
    if "不确定" in answer or "可能" in answer:
        score -= 0.2
    return min(max(score, 0.0), 1.0)

def retry_with_adjusted_params(query: str) -> str:
    if vectorstore:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        docs = retriever.invoke(query)
        if docs:
            return "\n".join([d.page_content for d in docs])
    return ""

def ask_agent(question: str, memory: MemoryManager, user_id: str, enable_retry=True) -> Dict[str, Any]:
    start_time = time.time()
    prefs = memory.get_preference(user_id)
    history = memory.get_recent_history(user_id, 2)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"偏好：预算{prefs.get('budget','未知')}元，天数{prefs.get('duration','未知')}天，上次城市{prefs.get('last_city','无')}。历史：{history}"},
        {"role": "assistant", "content": "明白，我会使用工具。"},
        {"role": "user", "content": question}
    ]
    response = Generation.call(
        model=DASHSCOPE_MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        result_format="message"
    )
    assistant = response.output.choices[0].message
    tool_calls_made = []
    if assistant.get("tool_calls"):
        for tc in assistant["tool_calls"]:
            name = tc["function"]["name"]
            args = json.loads(tc["function"]["arguments"])
            tool_calls_made.append((name, args))
            func = TOOL_FUNCTIONS.get(name)
            if func:
                result = func(**args)
            else:
                result = f"未知工具 {name}"
            messages.append(assistant)
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
        final = Generation.call(model=DASHSCOPE_MODEL, messages=messages, result_format="message")
        answer = final.output.choices[0].message.get("content", "")
    else:
        answer = assistant.get("content", "")
    confidence = estimate_confidence(answer)
    retried = False
    if enable_retry and confidence < CONFIDENCE_THRESHOLD:
        retried = True
        new_info = retry_with_adjusted_params(question)
        if new_info:
            messages.append({"role": "user", "content": f"补充信息：{new_info}，请重新回答"})
            retry_resp = Generation.call(model=DASHSCOPE_MODEL, messages=messages, result_format="message")
            answer = retry_resp.output.choices[0].message.get("content", answer)
            confidence = estimate_confidence(answer)
    elapsed = time.time() - start_time
    memory.save_history(user_id, question, answer)
    for city in ["东京","巴黎","纽约","北京","上海"]:
        if city in question:
            memory.save_preference(user_id, city=city)
    return {
        "question": question,
        "answer": answer,
        "confidence": confidence,
        "response_time": elapsed,
        "tool_calls": tool_calls_made,
        "retried": retried
    }

TEST_CASES = [
    {"question": "我想去东京旅行，有什么推荐？", "expected_tools": ["search_travel_guide"], "type": "单步"},
    {"question": "巴黎现在天气怎么样？", "expected_tools": ["get_weather"], "type": "单步"},
    {"question": "纽约有哪些必去景点？", "expected_tools": ["search_attractions"], "type": "单步"},
    {"question": "100美元能换多少人民币？", "expected_tools": ["get_exchange_rate"], "type": "单步"},
    {"question": "帮我规划一个3天北京行程，预算3000元", "expected_tools": ["search_travel_guide", "search_attractions"], "type": "多跳"},
    {"question": "我想去日本玩，但只有5000预算，能去东京吗？天气如何？", "expected_tools": ["search_travel_guide", "get_weather"], "type": "多跳"},
]

def evaluate_tool_accuracy(result: Dict, expected_tools: List[str]) -> bool:
    called = set([name for name, _ in result["tool_calls"]])
    expected = set(expected_tools)
    return expected.issubset(called)

def run_experiment():
    dashscope.api_key = DASHSCOPE_API_KEY
    memory = MemoryManager(MEMORY_DB_PATH)
    user_id = "test_user"
    results = []
    print("开始自动化测试...\n")
    for case in TEST_CASES:
        print(f"测试: {case['question']}")
        res = ask_agent(case["question"], memory, user_id, enable_retry=True)
        tool_ok = evaluate_tool_accuracy(res, case["expected_tools"])
        res["tool_correct"] = tool_ok
        results.append(res)
        print(f"  置信度: {res['confidence']:.2f}, 响应时间: {res['response_time']:.2f}s, 工具正确: {tool_ok}, 重试: {res['retried']}")
        print(f"  实际调用: {[name for name,_ in res['tool_calls']]}")
    # 计算指标
    avg_conf = np.mean([r["confidence"] for r in results])
    avg_time = np.mean([r["response_time"] for r in results])
    tool_acc = np.mean([1 if r["tool_correct"] else 0 for r in results])
    multi_indices = [i for i, case in enumerate(TEST_CASES) if case["type"] == "多跳"]
    multi_hop_solve_rate = np.mean([1 if results[i]["tool_correct"] else 0 for i in multi_indices]) if multi_indices else 0
    # 生成图表（中文已支持）
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0,0].bar(["工具调用准确率"], [tool_acc], color='skyblue')
    axes[0,0].set_ylim(0,1)
    axes[0,0].set_ylabel("准确率")
    axes[0,0].set_title(f"工具调用准确率: {tool_acc*100:.1f}%")
    times = [r["response_time"] for r in results]
    axes[0,1].bar(range(len(times)), times, color='lightgreen')
    axes[0,1].set_xticks(range(len(times)))
    axes[0,1].set_xticklabels([f"Q{i+1}" for i in range(len(times))], rotation=45)
    axes[0,1].set_ylabel("时间 (秒)")
    axes[0,1].set_title(f"平均响应时间: {avg_time:.2f}s")
    confs = [r["confidence"] for r in results]
    axes[1,0].hist(confs, bins=10, alpha=0.7, color='orange')
    axes[1,0].axvline(x=CONFIDENCE_THRESHOLD, color='r', linestyle='--', label=f"阈值 {CONFIDENCE_THRESHOLD}")
    axes[1,0].set_xlabel("置信度")
    axes[1,0].set_ylabel("频次")
    axes[1,0].set_title(f"置信度分布 (平均={avg_conf:.2f})")
    axes[1,0].legend()
    axes[1,1].bar(["整体工具准确率", "多跳问题解决率"], [tool_acc, multi_hop_solve_rate], color=['steelblue', 'darkorange'])
    axes[1,1].set_ylim(0,1)
    axes[1,1].set_ylabel("比率")
    axes[1,1].set_title("多跳推理能力对比")
    for i, v in enumerate([tool_acc, multi_hop_solve_rate]):
        axes[1,1].text(i, v+0.02, f"{v*100:.1f}%", ha='center')
    plt.tight_layout()
    plt.savefig("experiment_results.png", dpi=150)
    print("\n图表已保存为 experiment_results.png")
    metrics = {
        "avg_confidence": float(avg_conf),
        "avg_response_time": float(avg_time),
        "tool_accuracy": float(tool_acc),
        "multi_hop_solve_rate": float(multi_hop_solve_rate),
        "retried_cases_count": sum(1 for r in results if r["retried"]),
        "total_cases": len(results)
    }
    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print("指标已保存到 metrics.json")
    print("\n========== 详细结果 ==========")
    for i, r in enumerate(results):
        print(f"{i+1}. {r['question'][:30]}... 置信:{r['confidence']:.2f} 时间:{r['response_time']:.2f}s 工具正确:{r['tool_correct']} 重试:{r['retried']}")
    return results

if __name__ == "__main__":
    run_experiment()
    print("\n自动测试完成。如需查看图表，请打开 experiment_results.png")