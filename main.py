import os
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain.agents import create_agent
from dataclasses import dataclass
from langchain.tools import ToolRuntime
from langchain.chat_models import init_chat_model
from langchain.agents.structured_output import ToolStrategy
from langgraph.checkpoint.memory import InMemorySaver

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import json
from dataclasses import asdict
import time
import logging
from fastapi import Request

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ==================== Agent 配置 ====================

checkpointer = InMemorySaver()


@tool
def get_weather_for_location(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str


@dataclass
class ResponseFormat:
    """Response schema for the agent."""
    # A punny response (always required)
    punny_response: str
    # Any interesting information about the weather if available
    weather_conditions: str | None = None


@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """Retrieve user information based on user ID."""
    user_id = runtime.context.user_id
    return "Florida" if user_id == "1" else "SF"


# ==================== 初始化 Agent ====================

SYSTEM_PROMPT = """You are an expert weather forecaster, who speaks in puns.
You have access to two tools:

- get_weather_for_location: use this to get the weather for a specific location
- get_user_location: use this to get the user's location

If a user asks you for the weather, make sure you know the location. If you can tell from the question that they mean wherever they are, use the get_user_location tool to find their location."""

model = init_chat_model(
    "deepseek-chat",
    model_provider="openai",
    temperature=0.5,
    timeout=10,
    max_tokens=1000,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)

agent = create_agent(
    model,
    tools=[get_user_location, get_weather_for_location],
    system_prompt=SYSTEM_PROMPT,
    context_schema=Context,
    response_format=ToolStrategy(ResponseFormat),  # 结构化输出会增加一次 LLM 调用
    checkpointer=checkpointer
)

# ==================== FastAPI 应用 ====================

app = FastAPI(title="LangChain Agent API")

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源（生产环境应限制具体域名）
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """记录请求耗时"""
    start_time = time.time()

    response = await call_next(request)

    process_time = time.time() - start_time
    logger.info(f"[{request.method}] {request.url.path} - {response.status_code} - {process_time:.3f}s")

    return response


class AgentRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None


class AgentResponse(BaseModel):
    response: str
    session_id: str
    tools_used: List[str] = []
    structured_response: Optional[dict] = None


@app.post("/api/chat", response_model=AgentResponse)
async def chat_with_agent(request: AgentRequest):
    """与智能体对话的端点"""
    try:
        session_id = request.session_id or "default_session"
        user_id = request.user_id or "1"

        config = {"configurable": {"thread_id": session_id}}

        response = agent.invoke(
            {"messages": [{"role": "user", "content": request.message}]},
            config=config,
            context=Context(user_id=user_id)
        )

        # 提取工具调用信息
        tools_used = []
        for msg in response.get("messages", []):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for call in msg.tool_calls:
                    tools_used.append(call.get("name", "unknown"))

        # 转换 structured_response 为字典
        structured_resp = response.get("structured_response")
        structured_dict = asdict(structured_resp) if structured_resp else None

        return AgentResponse(
            response=response["messages"][-1].content,
            session_id=session_id,
            tools_used=tools_used,
            structured_response=structured_dict
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/stream")
async def chat_stream(request: AgentRequest):
    """流式响应端点 (SSE)"""
    from fastapi.responses import StreamingResponse

    session_id = request.session_id or "default_session"
    user_id = request.user_id or "1"

    async def generate():
        try:
            yield f"data: {json.dumps({'type': 'start', 'message': '正在处理'})}\n\n"

            # 记录实际处理时间（不包括流式传输）
            process_start = time.time()

            config = {"configurable": {"thread_id": session_id}}

            response = agent.invoke(
                {"messages": [{"role": "user", "content": request.message}]},
                config=config,
                context=Context(user_id=user_id)
            )

            process_time = time.time() - process_start
            content = response["messages"][-1].content

            # 发送处理时间
            yield f"data: {json.dumps({'type': 'process_time', 'time': f'{process_time:.3f}s'})}\n\n"

            # 模拟流式输出
            for char in content:
                yield f"data: {json.dumps({'type': 'token', 'content': char})}\n\n"

            structured_response = response.get("structured_response")
            if structured_response:
                structured_dict = asdict(structured_response)
                yield f"data: {json.dumps({'type': 'structured_response', 'data': structured_dict})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy"}


# ==================== 测试代码 ====================

def test_agent():
    """测试 agent 功能"""
    config = {"configurable": {"thread_id": "test_session"}}

    # 测试 1: 询问 "my location" 的天气
    print("=== Test 1: what is the weather at my location ===")
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "what is the weather at my location"}]},
        config=config,
        context=Context(user_id="1")
    )
    print(f"Response: {response['messages'][-1].content}")
    print(f"Structured Response: {response.get('structured_response')}")
    print()

    # 测试 2: 继续对话
    print("=== Test 2: thank you! ===")
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "thank you!"}]},
        config=config,
        context=Context(user_id="1")
    )
    print(f"Response: {response['messages'][-1].content}")
    print(f"Structured Response: {response.get('structured_response')}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # 运行测试
        test_agent()
    else:
        import uvicorn
        print("Starting FastAPI server on http://127.0.0.1:8000")
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
