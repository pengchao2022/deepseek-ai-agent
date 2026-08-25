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
from kubernetes import client, config
import uvicorn

app = FastAPI(title="Maxwell DeepSeek Tool-Enabled Agent", version="1.0")

# Define tools for agent
@tool
def calculate_shipping_cost(weight_kg: float, distance_km: float) -> str:
    """Calculate shipping costs. Use when user asks about freight or logistics fees"""
    cost = weight_kg * distance_km * 0.5
    return f"result: weight {weight_kg}kg, distance {distance_km}km, cost ${cost:.2f}"

@tool
def get_system_status(service_name: str) -> str:
    """Check the real health status of a backend service or pods on the EKS cluster. 
    Use when user asks about server, service health, or Kubernetes deployment status.
    """
    try:
        try:
            config.load_incluster_config()
        except Exception:
            config.load_kube_config()

        v1 = client.CoreV1Api()
        namespace = os.environ.get("TARGET_NAMESPACE", "ai-agent")
        
        pods = v1.list_namespaced_pod(
            namespace=namespace,
            label_selector=f"app.kubernetes.io/name={service_name}"
        )
        
        if not pods.items:
            pods = v1.list_namespaced_pod(namespace=namespace)

        if not pods.items:
            return f"No pods found in namespace [{namespace}] for service [{service_name}]."

        status_reports = []
        for pod in pods.items:
            pod_name = pod.metadata.name
            phase = pod.status.phase  # Running, Pending, Failed 等
            restarts = 0
            if pod.status.container_statuses:
                restarts = pod.status.container_statuses[0].restart_count
                
            ip = pod.status.pod_ip or "N/A"
            status_reports.append(
                f"- Pod: {pod_name} | Status: {phase} | Restarts: {restarts} | IP: {ip}"
            )

        return f"Real-time EKS Cluster Status (Namespace: {namespace}):\n" + "\n".join(status_reports)

    except Exception as e:
        return f"Failed to fetch real-time EKS cluster status due to error: {str(e)}"

@tool
def get_node_memory_usage() -> str:
    """Check the CPU and Memory usage of all nodes in the EKS cluster.
    Use when user asks about node health, node memory, or cluster node capacity.
    """
    try:
        try:
            config.load_incluster_config()
        except Exception:
            config.load_kube_config()

        api = client.CustomObjectsApi()
        metrics = api.list_cluster_custom_object(
            group="metrics.k8s.io", 
            version="v1beta1", 
            plural="nodes"
        )

        reports = []
        for item in metrics.get("items", []):
            node_name = item["metadata"]["name"]
            usage = item["usage"]
            memory_usage = usage.get("memory", "N/A")
            cpu_usage = usage.get("cpu", "N/A")
            reports.append(
                f"- Node: {node_name} | CPU Usage: {cpu_usage} | Memory Usage: {memory_usage}"
            )

        if not reports:
            return "No node metrics found. Please ensure metrics-server is installed in the cluster."

        return "Real-time EKS Node Resource Usage:\n" + "\n".join(reports)

    except Exception as e:
        return f"Failed to fetch node metrics. Make sure metrics-server is running. Error: {str(e)}"

@tool
def get_pod_logs(pod_name: str, tail_lines: int = 50) -> str:
    """Fetch the recent standard output logs of a specific Kubernetes pod for debugging and troubleshooting.
    Use when user asks to check logs, review errors, or debug a specific pod.
    """
    try:
        try:
            config.load_incluster_config()
        except Exception:
            config.load_kube_config()

        v1 = client.CoreV1Api()
        namespace = os.environ.get("TARGET_NAMESPACE", "ai-agent")

        logs = v1.read_namespaced_pod_log(
            name=pod_name,
            namespace=namespace,
            tail_lines=tail_lines
        )

        if not logs:
            return f"Pod [{pod_name}] in namespace [{namespace}] has empty logs."

        return f"Recent logs for Pod [{pod_name}] (last {tail_lines} lines):\n{logs}"

    except Exception as e:
        return f"Failed to fetch logs for pod [{pod_name}]. Error: {str(e)}"

# 注册所有工具
tools = [calculate_shipping_cost, get_system_status, get_node_memory_usage, get_pod_logs]

# initialization deepseek with customer tools
llm = ChatOpenAI(
    model="deepseek-v4-pro", # 确认你的 DeepSeek 模型名称是否正确
    openai_api_key=os.environ.get("DEEPSEEK_API_KEY"),  
    openai_api_base="https://api.deepseek.com",       
    temperature=0.1
)
llm_with_tools = llm.bind_tools(tools)

# Use LangGraph standard MessagesState
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

# FastAPI define
class ChatRequest(BaseModel):
    prompt: str = None
    contents: str = None

    def get_message(self) -> str:
        msg = self.prompt or self.contents
        if not msg:
            raise ValueError("Either 'prompt' or 'contents' must be provided.")
        return msg

@app.get("/")
def read_root():
    """根路径处理，消除 ALB / 健康检查的 404 日志"""
    return {"status": "running", "service": "Maxwell DeepSeek Tool-Enabled Agent"}

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

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)