import os
from typing import TypedDict, Annotated
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
import uvicorn

app = FastAPI(title="Maxwell DeepSeek Tool-Enabled Agent", version="1.0")

# Define tools for agent
# Python Decorator
@tool
def calculate_shipping_cost(weight_kg: float, distance_km: float) -> str:
    """Calculate shipping costs. Use when user asks about freight or logistics fees"""
    cost = weight_kg * distance_km * 0.5
    return f"result: weight {weight_kg}kg, distance {distance_km}km, cost ${cost:.2f}"

@tool
def get_system_status(service_name: str) -> str:
    """Check the health status of a backend service. Use when user asks about server or service health"""
    return f"Service [{service_name}] currently running on EKS cluster system-infra nodes works properly CPU usage 24%。"

tools = [calculate_shipping_cost, get_system_status]

# 初始化 DeepSeek 大模型并绑定工具
llm = ChatOpenAI(
    model="deepseek-v4-pro",
    openai_api_key=os.environ.get("DEEPSEEK_API_KEY"),  
    openai_api_base="https://api.deepseek.com",       
    temperature=0.1
)
llm_with_tools = llm.bind_tools(tools)

# 使用 LangGraph 标准的 MessagesState
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def call_model(state: AgentState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent")
agent_app = workflow.compile()

# FastAPI 
class ChatRequest(BaseModel):
    prompt: str = None
    contents: str = None

    def get_message(self) -> str:
        msg = self.prompt or self.contents
        if not msg:
            raise ValueError("Either 'prompt' or 'contents' must be provided.")
        return msg

@app.post("/chat")
def chat_with_agent(req: ChatRequest):
    try:
        user_input = req.get_message()
        initial_state = {"messages": [HumanMessage(content=user_input)]}
        result = agent_app.invoke(initial_state)
        last_message = result["messages"][-1]
        return {
            "status": "success",
            "response": last_message.content
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/healthz")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)