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

from trustrag.applications.rag_openai_bge import RagApplication, ApplicationConfig
from trustrag.modules.reranker.bge_reranker import BgeRerankerConfig
from trustrag.modules.retrieval.dense_retriever import DenseRetrieverConfig

# 修改成自己的配置！！！
app_config = ApplicationConfig()

# linux example
# app_config.docs_path = "/data/users/searchgpt/yq/trustrag/data/docs/"
# app_config.llm_model_path = "/data/users/searchgpt/pretrained_models/glm-4-9b-chat"
#
# retriever_config = DenseRetrieverConfig(
#     model_name_or_path="/data/users/searchgpt/pretrained_models/bge-large-zh-v1.5",
#     dim=1024,
#     index_path='/data/users/searchgpt/yq/TrustRAG/examples/retrievers/dense_cache'
# )
# rerank_config = BgeRerankerConfig(
#     model_name_or_path="/data/users/searchgpt/pretrained_models/bge-reranker-large"
# )

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

def get_file_list():
    if not os.path.exists(app_config.docs_path):
        return []
    return [f for f in os.listdir(app_config.docs_path)]


file_list = get_file_list()

def info_fn(filename):
    gr.Info(f"upload file:{filename} success!")

def upload_file(file):
    cache_base_dir = app_config.docs_path
    if not os.path.exists(cache_base_dir):
        os.mkdir(cache_base_dir)
    filename = os.path.basename(file.name)
    shutil.move(file.name, cache_base_dir + filename)
    # file_list首位插入新上传的文件
    file_list.insert(0, filename)
    application.add_document(app_config.docs_path + filename)
    info_fn(filename)
    return gr.Dropdown(choices=file_list, value=filename,interactive=True)

def set_knowledge(kg_name, history):
    try:
        application.load_vector_store()
        msg_status = f'{kg_name}知识库已成功加载'
    except Exception as e:
        print(e)
        msg_status = f'{kg_name}知识库未成功加载'
    return history + [[None, msg_status]]


def clear_session():
    return '', None


def create_checkboxes(contents):
    """
    Create checkbox choices and selected labels from contents

    Args:
        contents (list): List of dicts containing text, score, and label
    Returns:
        tuple: (choices list, selected labels list)
    """
    checkboxes = []
    selected_labels = []
    for content in contents:
        # Add score to the display text
        display_text = f"Score: {content['score']:.2f} | {content['text']}"
        checkboxes.append(display_text)
        if content.get('label') == 1:  # Check if label exists and equals 1
            selected_labels.append(display_text)
    return checkboxes, selected_labels

def shorten_label(text, max_length=10):
    if len(text) > 2 * max_length:
        return text[:max_length] + "..." + text[-max_length:]
    return text
def predict(input,
            large_language_model,
            embedding_model,
            top_k,
            use_web,
            use_pattern,
            history=None):
    if history is None:
        history = []

    # Handle web content
    web_content = ''
    if use_web == '使用':
        web_content = application.retriever.search_web(query=input)

    search_text = ''

    # Handle model Q&A mode
    if use_pattern == '模型问答':
        result = application.chat(query=input, web_content=web_content)
        history.append((input, result))
        search_text += web_content
        # Return empty judge results for Q&A mode
        return '', history, history, search_text, '', ([], [])

    # Handle RAG mode
    else:
        response, _, contents, rewrite_query = application.chat(
            question=input,
            top_k=top_k,
        )
        history.append((input, response))

        # Format search results
        for idx, source in enumerate(contents):
            sep = f'----------【搜索结果{idx + 1}：】---------------\n'
            search_text += f'{sep}\n{source["text"]}\n分数：{source["score"]:.2f}\n\n'

        # Add web content if available
        if web_content:
            search_text += "----------【网络检索内容】-----------\n"
            search_text += web_content

        # 创建复选框组件列表
        checkboxes = []
        for item in contents[:5]:
            # 根据label决定是否选中
            checked = bool(item.get('label', 0))
            # 创建复选框的HTML
            label_text = item.get('text', '')
            shortened_label = shorten_label(label_text)
            checkbox = gr.Checkbox(value=checked, visible=True, label=shortened_label, interactive=True)
            checkboxes.append(checkbox)
        return '', history, history, search_text, rewrite_query, checkboxes[0], checkboxes[1], checkboxes[2], checkboxes[3], checkboxes[4]

with gr.Blocks(theme="soft") as demo:
    gr.Markdown("""<h1><center>TrustRAG Application</center></h1>
        <center><font size=3>
        </center></font>
        """)
    state = gr.State()

    with gr.Row():
        with gr.Column(scale=1):
            embedding_model = gr.Dropdown([
                "text2vec-base",
                "bge-large-v1.5",
                "bge-base-v1.5",
            ],
                label="Embedding model",
                value="bge-large-v1.5")

            large_language_model = gr.Dropdown(
                [
                    "DeepSeek-V3",
                ],
                label="Large Language model",
                value="DeepSeek-V3")

            top_k = gr.Slider(1,
                              20,
                              value=5,
                              step=1,
                              label="retrieve top-k documents",
                              interactive=True)

            use_web = gr.Radio(["Use", "Not used"], label="web search",
                               info="Do you use network search? When using it, make sure the network is normal.",
                               value="Not used", interactive=False
                               )
            use_pattern = gr.Radio(
                [
                    'Only LLM',
                    'RAG',
                ],
                label="Mode",
                value='RAG',
                interactive=False)

            kg_name = gr.Radio(["Local Test Knowledge"],
                               label="Knowledge Base",
                               value=None,
                               info="To use the knowledge base, please load the knowledge base",
                               interactive=True)
            set_kg_btn = gr.Button("Load Knowledge Base")

            file = gr.File(label="Upload the file to the knowledge base, and try to match the content",
                           visible=True,
                           file_types=['.txt', '.md', '.docx', '.pdf']
                           )
            uploaded_files = gr.Dropdown(
                file_list,
                label="List of uploaded files",
                value=file_list[0] if len(file_list) > 0 else '',
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
                gr.Markdown("""Remind：<br>
                                        [TrustRAG Application](https://github.com/TrustRAG-community/TrustRAG) <br>
                                        If you have any questions, please provide feedback in [Github Issue区](https://github.com/TrustRAG-community/TrustRAG) . 
                                        <br>
                                        """)
        with gr.Column(scale=2):
            with gr.Row():
                rewrite = gr.Textbox(label='Rewrite Results')
            with gr.Row():
                search = gr.Textbox(label='Search Results')
            with gr.Row():
                # todo:创建judge显示结果，使用复选框
                # contents=[
                # 	{'text': '这是一段文本', 'score': 4.48828125, 'label': 1},
                # 	{'text': '这是一段文本', 'score': 1.48828125, 'label': 1
                # 	{'text': '这是一段文本', 'score': 0, 'label': 0},
                # ]
                with gr.Column() as checkbox_container:
                    # gr.Label("Judge Results")
                    checkbox_outputs = [gr.Checkbox(visible=False, interactive=True) for _ in range(5)]
        # ============= 触发动作=============
        file.upload(upload_file,
                    inputs=file,
                    outputs=None)
        set_kg_btn.click(
            set_knowledge,
            show_progress=True,
            inputs=[kg_name, chatbot],
            outputs=chatbot
        )
        # 发送按钮 提交
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
                   outputs=[message, chatbot, state, search,rewrite]+checkbox_outputs)

        # 清空历史对话按钮 提交
        clear_history.click(fn=clear_session,
                            inputs=[],
                            outputs=[chatbot, state],
                            queue=False)

        # 输入框 回车
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
                       outputs=[message, chatbot, state, search,rewrite]+checkbox_outputs)

demo.queue(concurrency_count=2).launch(
    server_name='0.0.0.0',
    server_port=7860,
    share=True,
    show_error=True,
    debug=True,
    enable_queue=True,
    inbrowser=False,
)