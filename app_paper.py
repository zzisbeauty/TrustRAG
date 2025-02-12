#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: app.py
@time: 2024/05/21
@contact: yanqiangmiffy@gamil.com
"""
import os
import shutil

import gradio as gr
import loguru
import pandas as pd

from trustrag.applications.rag_openai_bge import RagApplication, ApplicationConfig
from trustrag.modules.reranker.bge_reranker import BgeRerankerConfig
from trustrag.modules.retrieval.dense_retriever import DenseRetrieverConfig

# ========================== Config Start====================
app_config = ApplicationConfig()
app_config.docs_path = r"H:\Projects\TrustRAG\data\docs"
# app_config.key = "sk-04031f18c05a4dd5a561d33d984ca40f"
app_config.key = "sk-gDbFoQAYz9pwqBsH0aPA1H8DN9s0B9F3vWNjjPcijRBFjk7f"
retriever_config = DenseRetrieverConfig(
    model_name_or_path=r"H:\pretrained_models\mteb\bge-large-zh-v1.5",
    dim=1024,
    index_path=r'H:\Projects\TrustRAG\examples\retrievers\dense_cache'
)
rerank_config = BgeRerankerConfig(
    model_name_or_path=r"H:\pretrained_models\mteb\bge-reranker-large"
)
app_config.retriever_config = retriever_config
app_config.rerank_config = rerank_config
application = RagApplication(app_config)
application.init_vector_store()
# ========================== Config End====================


def upload_files(
        upload_files,
        chunk_size,
        chunk_overlap,
        enable_multimodal,
        enable_mandatory_ocr,
        upload_index,
):
    if not upload_files:
        return [
            gr.update(visible=False),
            gr.update(
                visible=True,
                value="No file selected. Please choose at least one file.",
            ),
        ]
    for state_info in upload_knowledge(
            upload_files=upload_files,
            oss_path=None,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            enable_multimodal=enable_multimodal,
            enable_mandatory_ocr=enable_mandatory_ocr,
            index_name=upload_index,
    ):
        yield state_info
def upload_knowledge(
        upload_files,
        oss_path,
        chunk_size,
        chunk_overlap,
        enable_multimodal,
        enable_mandatory_ocr,
        index_name,
        from_oss: bool = False,
):
    upload_result = "Upload success."
    error_msg = "Error"
    if error_msg:
        upload_result = f"Upload failed: {error_msg}"
    result = []
    yield [
        gr.update(visible=True, value=pd.DataFrame(result)),
        gr.update(
            visible=True,
            value=upload_result,
        ),
    ]


def clear_files():
    yield [
        gr.update(visible=False, value=pd.DataFrame()),
        gr.update(visible=False, value=""),
    ]


def clear_session():
    return '', None


def shorten_label(text, max_length=10):
    if len(text) > 2 * max_length:
        return text[:max_length] + "..." + text[-max_length:]
    return text


def predict(question,
            large_language_model,
            embedding_model,
            top_k,
            use_web,
            use_pattern,
            history=None):
    loguru.logger.info("User Question：" + question)
    if history is None:
        history = []
    # Handle web content
    web_content = ''
    if use_web == 'Use':
        loguru.logger.info("Use Web Search")
        results = application.web_searcher.retrieve(query=question, top_k=5)
        for search_result in results:
            web_content += search_result['title'] + " " + search_result['body'] + "\n"
    search_text = ''
    if use_pattern == 'Only LLM':
        # Handle model Q&A mode
        loguru.logger.info('Only LLM Mode:')

        # result = application.llm.chat(query=question, web_content=web_content)
        system_prompt = "You are a helpful assistant."
        user_input = [
            {"role": "user", "content": question}
        ]
        # 调用 chat 方法进行对话
        result, total_tokens = application.llm.chat(system=system_prompt, history=user_input)
        history.append((question, result))
        search_text += web_content

        # Return empty judge results for Q&A mode
        checkboxes = []
        for item in range(5):
            checkbox = gr.Checkbox(value=False, visible=False, interactive=False)
            checkboxes.append(checkbox)
        return '', history, history, search_text, '', checkboxes[0], checkboxes[1], checkboxes[2], checkboxes[3], \
            checkboxes[4]

    else:
        # Handle RAG mode
        loguru.logger.info('RAG Mode:')
        response, _, contents, rewrite_query = application.chat(
            question=question,
            top_k=top_k,
        )
        history.append((question, response))
        # Format search results
        for idx, source in enumerate(contents):
            sep = f'----------【搜索结果{idx + 1}：】---------------\n'
            search_text += f'{sep}\n{source["text"]}\n分数：{source["score"]:.2f}\n\n'
        # Add web content if available
        if web_content:
            search_text += "----------【网络检索内容】-----------\n"
            search_text += web_content
        checkboxes = []
        for idx,item in enumerate(contents[:5]):
            checked = bool(item.get('label', 0))
            label_text = item.get('text', '')
            shortened_label = str(idx+1)+"."+shorten_label(label_text)
            checkbox = gr.Checkbox(value=checked, visible=True, label=shortened_label, interactive=True)
            checkboxes.append(checkbox)
        return '', history, history, search_text, rewrite_query, checkboxes[0], checkboxes[1], checkboxes[2], \
            checkboxes[3], checkboxes[4]

with gr.Blocks(theme="soft") as demo:
    gr.Markdown("""<h1><center>TrustRAG Studio</center></h1><center><font size=3></center></font>""")
    with gr.Tab("\N{rocket} Corpus"):
        with gr.Row():
            with gr.Column(scale=2):
                upload_index = gr.Dropdown(
                    choices=["default_index"],
                    value="default_index",
                    label="\N{book} Knowledge Name",
                    elem_id="knowledge_name",
                )
                chunk_size = gr.Slider(
                    minimum=128,
                    maximum=1024,
                    value=128,
                    step=64,
                    label="\N{GEAR} Chunk Size",
                    info="Split Document With The Chunk Size",
                    interactive=True
                )

                chunk_overlap = gr.Slider(
                    minimum=0,
                    maximum=128,
                    value=0,
                    step=1,
                    label="\N{GEAR} Chunk Overlap",
                    info="Chunk Overlap Within Chunks",
                    interactive=True
                )

                enable_decontextualization = gr.Checkbox(
                    label="Yes",
                    value=True,
                    info="Process with Contextual Decontextualization",
                    elem_id="enable_Decontextualization",
                    visible=True,
                )

            with gr.Column(scale=8):
                with gr.Tab("Files"):
                    upload_file = gr.File(
                        label="Upload a knowledge file.", file_count="multiple"
                    )
                    upload_file_state_df = gr.DataFrame(
                        label="Upload Status Info", visible=False
                    )
                    upload_file_state = gr.Textbox(label="Upload Status", visible=False)
                with gr.Tab("Directory"):
                    upload_file_dir = gr.File(
                        label="Upload a knowledge directory.",
                        file_count="directory",
                    )
                    upload_dir_state_df = gr.DataFrame(
                        label="Upload Status Info", visible=False
                    )
                    upload_dir_state = gr.Textbox(label="Upload Status", visible=False)

                upload_file.upload(
                    fn=upload_files,
                    inputs=[
                        upload_file,
                        chunk_size,
                        chunk_overlap,
                        upload_index,
                    ],
                    outputs=[upload_file_state_df, upload_file_state],
                    api_name="upload_knowledge",
                )
                upload_file.clear(
                    fn=clear_files,
                    inputs=[],
                    outputs=[upload_file_state_df, upload_file_state],
                    api_name="clear_file",
                )
                dummy_component = gr.Textbox(visible=False, value="")
                upload_file_dir.upload(
                    fn=upload_knowledge,
                    inputs=[
                        upload_file_dir,
                        dummy_component,
                        chunk_size,
                        chunk_overlap,
                        upload_index,
                    ],
                    outputs=[upload_dir_state_df, upload_dir_state],
                    api_name="upload_knowledge_dir",
                )
                upload_file_dir.clear(
                    fn=clear_files,
                    inputs=[],
                    outputs=[upload_dir_state_df, upload_dir_state],
                    api_name="clear_file_dir",
                )
    with gr.Tab("\N{fire} Chat"):
        state = gr.State()
        with gr.Row():
            with gr.Column(scale=1):
                embedding_model = gr.Dropdown(
                    choices=[
                        "text2vec-base",
                        "bge-large-v1.5",
                        "bge-base-v1.5"
                    ],
                    label="Embedding model",
                    value="bge-large-v1.5"
                )

                large_language_model = gr.Dropdown(
                    choices=[
                        "DeepSeek-V3",
                    ],
                    label="Large Language model",
                    value="DeepSeek-V3"
                )

                top_k = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=5,
                    step=1,
                    label="Retrieve Top-k Documents",
                    interactive=True
                )

                use_web = gr.Radio(
                    choices=["Use", "Not used"],
                    label="Web Search",
                    info="Do you use network search? When using it, make sure the network is normal.",
                    value="Not used",
                    interactive=True
                )
                use_pattern = gr.Radio(
                    choices=[
                        'Only LLM',
                        'RAG',
                    ],
                    label="Chat Mode",
                    value='RAG',
                    interactive=True
                )
                knowledge_name = gr.Dropdown(
                    choices=[
                        "default_index",
                    ],
                    label="Knowledge Name",
                    value="default_index",
                    interactive=True
                )
            with gr.Column(scale=4):
                with gr.Row():
                    chatbot = gr.Chatbot(label='TrustRAG Application').style(height=650)
                with gr.Row():
                    message = gr.Textbox(label='Please enter a question')
                with gr.Row():
                    clear_history = gr.Button("🧹 Clear")
                    send = gr.Button("🚀 Send")
                with gr.Row():
                    gr.Markdown(
                        """Remind：[TrustRAG Application](https://github.com/TrustRAG-community/TrustRAG)If you have any questions, please provide feedback in [Github Issue区](https://github.com/TrustRAG-community/TrustRAG) . <br>""")
            with gr.Column(scale=2):
                with gr.Row():
                    rewrite = gr.Textbox(label='Query Reformulate')
                with gr.Row():
                    # todo:创建judge显示结果，使用复选框
                    with gr.Column() as checkbox_container:
                        # gr.Markdown("Document Judge")
                        checkbox_outputs = [gr.Checkbox(visible=False, interactive=True) for _ in range(5)]
                with gr.Row():
                    search = gr.Textbox(label='Claim Attribute')

            # submit
            send.click(predict,
                       inputs=[
                           message,
                           large_language_model,
                           embedding_model,
                           top_k,
                           use_web,
                           use_pattern,
                           state
                       ],
                       outputs=[message, chatbot, state, search, rewrite] + checkbox_outputs)

            # clear
            clear_history.click(fn=clear_session,
                                inputs=[],
                                outputs=[chatbot, state],
                                queue=False)
            # enter
            message.submit(predict,
                           inputs=[
                               message,
                               large_language_model,
                               embedding_model,
                               top_k,
                               use_web,
                               use_pattern,
                               state
                           ],
                           outputs=[message, chatbot, state, search, rewrite] + checkbox_outputs)

demo.queue(concurrency_count=2).launch(
    server_name='0.0.0.0',
    server_port=7860,
    share=True,
    show_error=True,
    debug=True,
    enable_queue=True,
    inbrowser=False,
)
