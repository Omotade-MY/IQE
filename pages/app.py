# app.py
__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import openai
import streamlit as st
from io import BytesIO
from markdown_pdf import MarkdownPdf, Section
import os
from utils.processors import (
    PDFProcessor,
    VideoProcessor,
    AudioProcessor,
    DummyProcessor,
)
from utils.evaluator import DesignEvaluator, TransferEvaluator, PerformanceEvaluator
from utils.workflow import workflow_builder, evaluation_summarizer, ContentSummarizer
from assets.prompts import (
    SYSTEM_PROMPT,
    GENERAL_EVAL_PROMPT,
    GENERAL_SLIDING_EVAL_PROMPT,
)

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

st.session_state["report_status"] = False
if "content_is_large" not in st.session_state:
    st.session_state["content_is_large"] = False

import uuid

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


def generate_unique_id():
    """
    Generates a unique ID using UUID4.

    Returns:
        str: A unique ID in string format.
    """
    return str(uuid.uuid4())


if not "thread_id" in st.session_state:
    st.session_state["thread_id"] = generate_unique_id()
# Configure a user thread
config = {"configurable": {"thread_id": st.session_state["thread_id"]}}
graph = workflow_builder()


# st.session_state['content_summary'] = None
class CourseEvaluatorApp:
    def __init__(self):
        self.setup_streamlit()
        self.initialize_processors()

    def setup_streamlit(self):
        st.set_page_config(
            page_title="Instructional Quality Prototype", layout="centered"
        )

    def initialize_processors(self):
        self.pdf_processor = PDFProcessor()
        self.video_processor = VideoProcessor()
        self.audio_processor = AudioProcessor()
        self.dummy_processor = DummyProcessor()

    def process_file(self, file):
        file_name = file.name
        file_ext = os.path.splitext(file_name)[-1].lower()
        if file_ext == ".pdf":
            st.write("PDF File detected")
            content = self.pdf_processor.process(file)
        elif file_ext in [".mp3", ".wav"]:
            st.write("Audio deteted")
            content = self.audio_processor.process(file)
        else:
            st.error("Unsupported file type")
            content = self.dummy_processor.process(file)

        return content

    def router(self, state):
        message = state["messages"][-1]
        if hasattr(message, "tool_calls") and len(message.tool_calls) > 0:
            available_tools = [
                "gen_scope",
                "design_frameworks",
                "transer_work_frameworks",
                "perform_man_frameworks",
                "synthesize_evalaution_summary",
                "generate_downloadable_report",
                "request_content",
            ]
            outbound_msgs = []
            for tool_call in message.tool_calls:
                if tool_call["name"] not in available_tools:
                    raise ValueError(
                        f"Model Called a Tool {tool_call['name']} That is not implement"
                    )

                if tool_call["name"] == "design_frameworks":
                    design = st.session_state["design_evaluator"]
                    ## Call the preliminary review chain
                    with st.spinner("Evaluating Design Frameworks"):
                        critique = tool_call["args"]
                        design.set_critique(**critique)
                        if st.session_state["content_is_large"]:
                            design_eval = design.eval_design(slide=True)

                        else:
                            design_eval = design.eval_design()

                        outbound_msgs.append(
                            ToolMessage(
                                content=str(design_eval),
                                name=tool_call["name"],
                                tool_call_id=tool_call["id"],
                            )
                        )

                elif tool_call["name"] == "transer_work_frameworks":
                    transfer = st.session_state["transfer_evaluator"]
                    with st.spinner("Evaluating Transfer Work Frameworks"):
                        critique = tool_call["args"]
                        transfer.set_critique(**critique)
                        if st.session_state["content_is_large"]:
                            tranaser_eval = transfer.eval_transfer(slide=True)
                        else:
                            tranaser_eval = transfer.eval_transfer()
                        outbound_msgs.append(
                            ToolMessage(
                                content=str(tranaser_eval),
                                name=tool_call["name"],
                                tool_call_id=tool_call["id"],
                            )
                        )

                elif tool_call["name"] == "perform_man_frameworks":
                    performance = st.session_state["performance_evaluator"]
                    with st.spinner("Evaluation Performance and Management Frameworks"):
                        critique = tool_call["args"]
                        performance.set_critique(**critique)
                        if st.session_state["content_is_large"]:
                            performance_eval = performance.eval_performance(slide=True)
                        else:
                            performance_eval = performance.eval_performance()
                        outbound_msgs.append(
                            ToolMessage(
                                content=str(performance_eval),
                                name=tool_call["name"],
                                tool_call_id=tool_call["id"],
                            )
                        )

                elif tool_call["name"] == "synthesize_evalaution_summary":
                    # print("Running summary TOOL")
                    with st.spinner("Running Evaluation Summary Synthesizer"):
                        eval_summary = evaluation_summarizer(state)
                        # st.markdown(eval_summary['summary'])
                        outbound_msgs.append(
                            ToolMessage(
                                content=eval_summary["summary"],
                                name=tool_call["name"],
                                tool_call_id=tool_call["id"],
                            )
                        )

                elif tool_call["name"] == "generate_downloadable_report":
                    # print("Generating Report")
                    report_statements = tool_call["args"]
                    st.session_state["report_status"] = True
                    report = self.save_to_pdf(**report_statements)
                    st.session_state["report"] = report
                    message = f"report saved"

                    outbound_msgs.append(
                        ToolMessage(
                            content=message,
                            name=tool_call["name"],
                            tool_call_id=tool_call["id"],
                        )
                    )

                elif tool_call["name"] == "gen_scope":
                    # print("Generating Scope")
                    # st.session_state["content_summary"] = None
                    additions = tool_call["args"]["info"]
                    summarizer = st.session_state.get("summarizer", None)
                    if summarizer is not None:
                        summary = summarizer.summarize(modifiers=additions)
                    else:
                        summary = {"summary": "Summarizer is not found"}

                    st.session_state["content_summary"] = summary
                    outbound_msgs.append(
                        ToolMessage(
                            content=summary["summary"],
                            name=tool_call["name"],
                            tool_call_id=tool_call["id"],
                        )
                    )

                elif tool_call["name"] == "request_content":
                    outbound_msgs.append(
                        ToolMessage(
                            content="Uploaded",
                            name=tool_call["name"],
                            tool_call_id=tool_call["id"],
                        )
                    )

                else:
                    message = "No result found. Invalid tool  or tool not implemented"
                    outbound_msgs.append(
                        ToolMessage(
                            content=message,
                            name=tool_call["name"],
                            tool_call_id=tool_call["id"],
                        )
                    )
            return outbound_msgs
        else:
            return False

    def save_to_pdf(self, report_statements):
        """
        Save raw string content to a PDF file.

        Args:
            text (str): The string content to save to the PDF.
            filename (str): The name of the output PDF file (e.g., 'report.pdf').

        Returns:
            bool
        """

        pdf = MarkdownPdf(toc_level=1)
        pdf.add_section(Section(report_statements))
        # Use an in-memory buffer
        pdf_buffer = BytesIO()
        pdf_buffer.seek(0)
        pdf.save(pdf_buffer)
        # st.download_button("Download Evaluation PDF", data=pdf_buffer, file_name="Evaluation_Summary_Report.pdf", mime="application/pdf")

        return pdf_buffer

    # Sidebar: File Upload
    def main(self):

        st.sidebar.title("Upload Your Course Material")
        uploaded_file = st.sidebar.file_uploader(
            "Upload your course materials. \n Note: with this prototype we have limited upload features and cannot attach zip files or non-YouTube URL links\n",
            type=["pdf", "mp3"],
        )
        youtube_url = st.sidebar.text_input(
            "Or provide a YouTube URL.\nFor any YouTube course, note that with v1, if the YouTube course does not have a transcript the course cannot be evaluated"
        )
        # Custom CSS to modify sidebar and page layout
        st.sidebar.markdown(
            """
            <style>
            [data-testid="stSidebar"] {
                background-color: #f0f2f6;
                width: 250px;
            }
            .main-title {
                text-align: center;
                font-size: 2.5em;
                color: #2c3e50;
                margin-bottom: 30px;
            }
            </style>
        """,
            unsafe_allow_html=True,
        )

        if "youtube_url" not in st.session_state:
            st.session_state["youtube_url"] = youtube_url
            st.session_state["instruction"] = True
        # Sidebar: Instructions

        # Welcome Message
        st.markdown(
            """
                <h1 class="main-title">Instructional Quality Agent (IQA)  Prototype</h1>

                Check out the upload steps on your left. <span class="arrow">←</span>
                """,
            unsafe_allow_html=True,
        )

        if youtube_url or uploaded_file:
            with st.columns(1)[0]:
                st.markdown(
                    """


                """
                )

        elif st.session_state.get("instruction", True):

            # Instructions Moved to Main Area
            st.markdown(
                """
                ## Instructions

                1. **Attach A Course Material**:
                    - Upload the course material you want evaluate (PDF, YouTube link, or audio file).
                    - If you provided a YouTube link, click on enter to apply.
                2. **Summary and Scope Confirmation**:
                    - I will verify the course structure and categories, and confirm if I capture the content of the course.
                3. **Critique Level**:
                    - Provide a critique level: (0 - 10).
                4. **Evaluation Frameworks**:
                    - I will evaluate your course using various proven frameworks.
                5. **Summarized Results**:
                    - At the end, you will receive a summary with ratings and actionable insights.
                6. **Suggestions**:
                    - Based on the evaluation, I will provide actionable insights and suggestions.
                7. **Downloadable Report**:
                    - You can download the report for further analysis or sharing.

                **Note**: The system may occasionally jump a step; if you need to go through that step, you can remind the model to go back to it.
                """
            )

        else:
            with st.columns(1)[0]:
                st.markdown(
                    """


                """
                )

        # Step 3.1: Confirmation Page
        if "content" not in st.session_state:
            if uploaded_file:
                st.success(f"File '{uploaded_file.name}' uploaded successfully!")
                content = self.process_file(uploaded_file)
                st.session_state["instruction"] = False
                if content["content_type"] == "pdf":
                    if content["metadata"]["pages"] >= 20:
                        st.session_state["content_is_large"] = True
                        # st.warning("Content is Large for system to process")
                else:
                    if content["metadata"]["duration"] >= 1200:  # 20 minutes
                        st.session_state["content_is_large"] = True
                        # st.warning("Content is Large for system to process")

                if content is None:
                    st.stop()

            elif youtube_url:
                if "youtube.com" not in youtube_url and "youtu.be" not in youtube_url:
                    st.error("Invalid YouTube URL. Please provide a valid link.")
                else:
                    with st.spinner("Extracting transcript from YouTube..."):
                        content = self.video_processor.process(youtube_url)
                        if content is not None:
                            st.success("YouTube content successfully extracted!")
                            if content["metadata"]["duration"] >= 1200:  # 20 minutes
                                st.session_state["content_is_large"] = True
                                # st.warning("Content is Large for system to process")
                        else:
                            st.stop()
            else:
                st.sidebar.info(
                    "Please upload your course materials here or Provide a YouTube link to your course"
                )
                st.stop()

            st.session_state["content"] = content
            # print(type(content))
            summarizer = ContentSummarizer(content)
            st.session_state["summarizer"] = summarizer
        st.session_state["instruction"] = False

        # else:
        #     if not (uploaded_file or st.session_state['youtube_url']):
        #         del st.session_state['content']
        if st.session_state["content_is_large"]:
            st.warning("Large Content Detected")

        if "design_evaluator" not in st.session_state:
            with st.spinner("Building Evaluator Models"):
                design = DesignEvaluator(
                    prompt=GENERAL_EVAL_PROMPT,
                    sprompt=GENERAL_SLIDING_EVAL_PROMPT,
                    content=content,
                )
                st.session_state["design_evaluator"] = design
        if "performance_evaluator" not in st.session_state:
            st.session_state["performance_evaluator"] = PerformanceEvaluator(
                prompt=GENERAL_EVAL_PROMPT,
                sprompt=GENERAL_SLIDING_EVAL_PROMPT,
                content=content,
            )

        if "transfer_evaluator" not in st.session_state:
            st.session_state["transfer_evaluator"] = TransferEvaluator(
                prompt=GENERAL_EVAL_PROMPT,
                sprompt=GENERAL_SLIDING_EVAL_PROMPT,
                content=content,
            )

        if "content_summary" in st.session_state:
            # print("GOT HERE")
            st.subheader("Extracted Content Summary")

        else:
            try:
                with st.spinner("Detecting Content Scope"):
                    summary = summarizer.summarize()
                    # print(summary)
                    st.session_state["content_summary"] = summary
                    st.subheader("Extracted Content Summary")
            except Exception as err:
                st.error("Error occured while extracting summary. Please contact admin")
                st.stop()

            ## Make the agent aware of the summary
            snapshot = graph.get_state(config)
            # print("GOT HERE")
            if "messages" not in snapshot.values:
                snapshot.values["messages"] = []
                snapshot.values["messages"].append(
                    (SystemMessage(content=SYSTEM_PROMPT))
                )
                snapshot.values["messages"].append(
                    (AIMessage(content=summary["summary"]))
                )
                # Update the graph state
                new_messages = snapshot.values
                graph.update_state(config, new_messages)

        snapshot = graph.get_state(config)
        # st.write(snapshot)

        messages = snapshot.values.get("messages", [])
        if messages:
            for message in snapshot.values["messages"]:
                if isinstance(message, HumanMessage):
                    st.chat_message("human").write(message.content)
                elif isinstance(message, AIMessage):
                    st.chat_message("ai").write(message.content, unsafe_allow_html=True)
        else:
            st.write("No messages found")

        if user_input := st.chat_input():
            st.chat_message("human").write(user_input)

            res = graph.invoke({"messages": [user_input]}, config)
            ai_message = res["messages"][-1].content
            if ai_message:
                st.chat_message("ai").write(ai_message, unsafe_allow_html=True)

            while True:
                snapshot = graph.get_state(config)
                pre_result = self.router(snapshot.values)
                if pre_result:
                    # print("GOT RESULT")
                    # st.write("GOT RESULT")
                    snapshot.values["messages"] += pre_result

                    updated_state = snapshot.values
                    graph.update_state(config, updated_state)
                    res = graph.invoke({"proceed": True}, config)
                    eval_summary = list(
                        filter(
                            lambda msg: msg.name == "synthesize_evalaution_summary",
                            snapshot.values["messages"],
                        )
                    )

                    if pre_result[-1].name == "synthesize_evalaution_summary":
                        # ai_message = pre_result[-1].content + "\n Would you like to recieve actionable suggestions or we proceed to wrap up?"
                        if len(res["messages"][-1].content) < 500:
                            print("USING TOOL MESSAGE")
                            snapshot = graph.get_state(config)
                            snapshot.values["messages"] += [
                                AIMessage(content=pre_result[-1].content)
                            ]
                            updated_state = snapshot.values
                            graph.update_state(config, updated_state)
                            st.chat_message("ai").markdown(
                                pre_result[-1].content, unsafe_allow_html=True
                            )
                        else:

                            st.chat_message("ai").markdown(
                                res["messages"][-1].content, unsafe_allow_html=True
                            )
                    else:
                        st.chat_message("ai").markdown(
                            res["messages"][-1].content, unsafe_allow_html=True
                        )

                else:
                    break
        if st.session_state["report_status"]:
            # snapshot = graph.get_state(config)
            try:
                content = list(
                    filter(
                        lambda msg: msg.name == "synthesize_evalaution_summary",
                        snapshot.values["messages"],
                    )
                )[0].content
                report_buffer = self.save_to_pdf(content)
                st.download_button(
                    "Download Evaluation PDF",
                    data=report_buffer,
                    file_name="Evaluation_Summary_Report.pdf",
                    mime="application/pdf",
                )
            except IndexError:
                st.download_button(
                    "Download Evaluation PDF",
                    data=st.session_state["report"],
                    file_name="Evaluation_Summary_Report.pdf",
                    mime="application/pdf",
                )
            st.session_state["report_status"] = False


if __name__ == "__main__":
    app = CourseEvaluatorApp()
    try:
        app.main()
    except openai.BadRequestError as err:
        if err.status_code == 400:
            st.error(
                "Oops! System Could not process the content\n The content you provided is too large or the model ran out memory space"
            )
    except Exception as err:
        st.write(f"Error occure {str(err)}. Please reload page")
    # except Exception as err:
    #     st.error(f"Error has occured {str(err)}")
