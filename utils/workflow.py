# workflow.py

import os
from typing import Dict, Any

from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from dotenv import load_dotenv, find_dotenv
from langchain.prompts import PromptTemplate
from typing import Annotated, Dict, Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from typing import Dict, Any, Annotated
from assets.prompts import (
    CONTENT_SUMMARY_PROMPT,
    SYSTEM_PROMPT,
    SUMMARY_SYNTHESIS_PROMPT,
)
from utils.evaluator import Tools


load_dotenv(find_dotenv())

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# Define all the chains


class ContentSummarizer:
    def __init__(
        self, content, llm=ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    ):
        self.content_text = content["raw_text"]
        self.chunks = content["chunks"]
        self.docs = [Document(page_content=chunk) for chunk in content["chunks"]]
        self.content_type = content["content_type"]
        self.word_count = content["metadata"]["word_count"]
        self.llm = llm

    def get_prompts(self, modifiers):
        additions = f"Additional information from the user: {modifiers}"
        prompt = CONTENT_SUMMARY_PROMPT + (additions if modifiers else "")
        summary_prompt = PromptTemplate.from_template(prompt)

        summary_prompt = summary_prompt.partial(content_type=self.content_type)
        return summary_prompt

    def summarize(self, modifiers=""):
        summary_prompt = self.get_prompts(modifiers)
        if self.word_count < 100000:
            self.summary_chain = load_summarize_chain(
                self.llm, prompt=summary_prompt, chain_type="stuff"
            )

        else:
            self.summary_chain = load_summarize_chain(self.llm, chain_type="refine")
            self.summary_chain.initial_llm_chain.prompt = summary_prompt

        summary = self.summary_chain.invoke(self.docs)

        return {"summary": summary["output_text"]}


def evaluation_summarizer(state):
    messages = state["messages"]
    summary_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
    summary_prompt = PromptTemplate.from_template(SUMMARY_SYNTHESIS_PROMPT)
    summary_chain = summary_prompt | summary_llm
    eval_summary = summary_chain.invoke(messages)

    return {"summary": eval_summary.content}


path = "memory.sqlite"

memconn = sqlite3.connect(path, check_same_thread=False)

tools = [
    Tools.gen_scope,
    Tools.design_frameworks,
    Tools.perform_man_frameworks,
    Tools.transer_work_frameworks,
    Tools.synthesize_evalaution_summary,
    Tools.generate_downloadable_report,
    Tools.request_content,
]

model = llm.bind_tools(tools)


class CourseEvaluationState(TypedDict):
    messages: Annotated[list, add_messages]
    proceed: bool
    content: Dict[str, Any]
    content_type: str = ""
    summary: str = ""


def agent(state: CourseEvaluationState):
    if isinstance(state["messages"][0], SystemMessage):
        pass
    else:
        state["messages"].insert(0, SystemMessage(content=SYSTEM_PROMPT))

    # print("invoking the model")
    res = model.invoke(input=state["messages"])
    return state | {"messages": [res]}


memory = SqliteSaver(memconn)
# memory = MemorySaver()


def workflow_builder():
    graph_builder = StateGraph(CourseEvaluationState)
    graph_builder.add_node("agent", agent)
    graph_builder.add_edge(START, "agent")
    graph_builder.add_edge("agent", END)
    graph = graph_builder.compile(checkpointer=memory)
    return graph


# define the evaluations
