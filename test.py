from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()
print("Using key:", bool(os.getenv("GOOGLE_API_KEY")))

class LLMState(TypedDict):
    question: str
    answer: str

# replace with a supported model name you got from ListModels
llm = ChatGoogleGenerativeAI(model="gemini-flash-latest")

def llm_qa(state: LLMState) -> LLMState:
    question = state["question"]
    prompt = f"Answer the following question concisely:\n{question}"
    answer = llm.invoke(prompt).content
    state["answer"] = answer
    return state

graph = StateGraph(LLMState)
graph.add_node("llm_qa", llm_qa)
graph.add_edge(START, "llm_qa")
graph.add_edge("llm_qa", END)
workflow = graph.compile()

initial_state = {"question": "What is LangGraph?"}
final_state = workflow.invoke(initial_state)
print(final_state)
