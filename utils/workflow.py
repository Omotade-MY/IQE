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
import streamlit as st
import sqlite3
from typing import Dict, Any, Annotated
from assets.prompts import (
    CONTENT_SUMMARY_PROMPT,
    SYSTEM_PROMPT,
    SUMMARY_SYNTHESIS_PROMPT,
)
from utils.evaluator import Tools


load_dotenv(find_dotenv())
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
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


def report_generator(state):
    messages = state["messages"]
    report_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
    report_prompt_template = """
    You are evaluating an instructional course content. Given the evaluation history below
    ---------------
    {history}
    ----------------
    Generate a summary report based on the summary and evaluation given to you. The summary report you have the following:
    Include the name of the course as the
    Summary Tables: Present relevant summary tables [Include the scores and explanation of the scores].
    Explanations: Briefly explain the key insights from each table.
    Recommendations: Include actionable suggestions based on the data.
    Keep the report focused and to the point.

    """
    report_prompt = PromptTemplate.from_template(report_prompt_template)
    report_chain = report_prompt | report_llm
    report = report_chain.invoke(messages)

    return {"report": report.content}


path = "memory.sqlite"

memconn = sqlite3.connect(path, check_same_thread=False)

tools = [
    Tools.gen_scope,
    Tools.design_frameworks,
    Tools.perform_man_frameworks,
    Tools.transer_work_frameworks,
    Tools.synthesize_evalaution_summary,
    # Tools.generate_downloadable_report,
    Tools.request_content,
]

model = llm.bind_tools(tools)


class CourseEvaluationState(TypedDict):
    messages: Annotated[list, add_messages]
    proceed: bool
    content: Dict[str, Any]
    content_type: str = ""
    steps: dict


class StepState:
    name: str
    number: int
    started: bool = False
    inprogess: bool = False
    completed: bool = False
    status: str = "not_stated"

    def update(self):
        if self.completed:
            self.status = "completed"
        elif self.inprogess:
            self.status = "inprogres"


steps_list = [
    "Content Intake & Validation",
    "Scope Confirmation",
    "Level of Critique",
    "First Round - DESIGN",
    "Second Round - TRANSFER",
    "Third Round - PERFORMANCE",
    "Synthesis & Summary",
    "Suggestions",
]


def agent(state: CourseEvaluationState):
    if isinstance(state["messages"][0], SystemMessage):
        pass
    else:
        state["messages"].insert(0, SystemMessage(content=SYSTEM_PROMPT))

    # print("invoking the model")

    step_num = state["steps"]["current_step"]
    # print(state["steps"])
    current_step = state["steps"]["steps"][step_num]
    instruct = state["steps"]["steps"].get("instruction", "")

    state_info = f"\nBelow is information about the current state of evaluation for your reminder\n-------\nStep Number: {step_num}\nCurrent Step: {current_step}\n{instruct}\n-------"
    state_info = SystemMessage(content=state_info)
    res = model.invoke(input=state["messages"] + [state_info])
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
