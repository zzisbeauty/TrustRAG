#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: RagApplication.py
@time: 2024/05/20
@contact: yanqiangmiffy@gamil.com
@description:use openai etc. llm service demo
"""
import os
from trustrag.modules.document.common_parser import CommonParser
from trustrag.modules.generator.chat import DeepSeekChat
from trustrag.modules.reranker.bge_reranker import BgeReranker
from trustrag.modules.retrieval.dense_retriever import DenseRetriever
from trustrag.modules.document.chunk import TextChunker

class ApplicationConfig():
    def __init__(self):
        self.retriever_config = None
        self.rerank_config = None


class RagApplication():
    def __init__(self, config):
        self.config = config
        self.parser = CommonParser()
        self.retriever = DenseRetriever(self.config.retriever_config)
        self.reranker = BgeReranker(self.config.rerank_config)
        self.llm = DeepSeekChat(key=self.config.your_key)
        self.tc=TextChunker()
        self.rag_prompt="""请结合参考的上下文内容回答用户问题，如果上下文不能支撑用户问题，那么回答不知道或者我无法根据参考信息回答。
        问题: {question}
        可参考的上下文：
        ···
        {context}
        ···
        如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，你不知道。
        有用的回答:"""
    def init_vector_store(self):
        """

        """
        print("init_vector_store ... ")
        all_paragraphs = []
        all_chunks = []
        for filename in os.listdir(self.config.docs_path):
            file_path = os.path.join(self.config.docs_path, filename)
            try:
                paragraphs=self.parser.parse(file_path)
                all_paragraphs.append(paragraphs)
            except:
                pass
        print("chunking for paragraphs")
        for paragraphs in all_paragraphs:
            chunks=self.tc.get_chunks(paragraphs, 256)
            all_chunks.extend(chunks)
        self.retriever.build_from_texts(all_chunks)
        print("init_vector_store done! ")
        self.retriever.save_index(self.config.retriever_config.index_path)

    def load_vector_store(self):
        self.retriever.load_index(self.config.retriever_config.index_path)

    def add_document(self, file_path):
        chunks = self.parser.parse(file_path)
        for chunk in chunks:
            self.retriever.add_text(chunk)
        print("add_document done!")

    def chat(self, question: str = '', top_k: int = 5):
        contents = self.retriever.retrieve(query=question, top_k=top_k)
        contents = self.reranker.rerank(query=question, documents=[content['text'] for content in contents])
        content = '\n'.join([content['text'] for content in contents])

        system_prompt = "你是一个人工智能助手."
        prompt = self.rag_prompt.format(system_prompt=system_prompt, question=question,context=content)
        history = [
            {"role": "user", "content": prompt}
        ]
        gen_conf = {
            "temperature": 0.7,
            "max_tokens": 100
        }

        # 调用 chat 方法进行对话
        result = self.llm.chat(system=system_prompt, history=history, gen_conf=gen_conf)
        return result, history, contents
