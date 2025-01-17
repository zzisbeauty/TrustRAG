#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: yanqiangmiffy
@contact:1185918903@qq.com
@license: Apache Licence
@time: 2024/6/5 23:07
"""
import gc
import os
from typing import List,Dict,Union
from openai import OpenAI
import faiss
import numpy as np
from FlagEmbedding import FlagModel
from tqdm import tqdm

from trustrag.modules.retrieval.base import BaseConfig, BaseRetriever

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class DenseRetrieverConfig(BaseConfig):
    """
    Configuration class for Dense Retriever.

    Attributes:
        model_name (str): Name of the transformer model to be used.
        dim (int): Dimension of the embeddings.
        index_path (str): Path to save or load the FAISS index.
        rebuild_index (bool): Flag to rebuild the index if True.
    """

    def __init__(
            self,
            model_name_or_path='sentence-transformers/all-mpnet-base-v2',
            dim=768,
            index_path=None,
            batch_size=32,
            api_key=None,
            base_url=None
    ):
        self.model_name = model_name_or_path
        self.dim = dim
        self.index_path = index_path
        self.batch_size = batch_size
        self.api_key = api_key
        self.base_url = base_url

    def validate(self):
        """Validate Dense configuration parameters."""
        if not isinstance(self.model_name, str) or not self.model_name:
            raise ValueError("Model name must be a non-empty string.")
        if not isinstance(self.dim, int) or self.dim <= 0:
            raise ValueError("Dimension must be a positive integer.")
        if self.index_path and not isinstance(self.index_path, str):
            raise ValueError("Index directory path must be a string.")
        print("Dense configuration is valid.")


class DenseRetriever(BaseRetriever):
    """
        Implements a dense retriever for efficiently searching documents.

        Methods:
            __init__(config): Initializes the retriever with given configuration.
            mean_pooling(model_output, attention_mask): Performs mean pooling on model outputs.
            get_embedding(sentences): Generates embeddings for provided sentences.
            load_index(index_path): Loads the FAISS index from a file.
            save_index(): Saves the current FAISS index to a file.
            add_doc(document_text): Adds a document to the index.
            build_from_texts(texts): Processes and indexes a list of texts.
            retrieve(query): Retrieves the top_k documents relevant to the query.
    """

    def __init__(self, config):
        self.config = config
        # self.model = FlagModel(config.model_name)
        self.client = OpenAI(
                            base_url=config.base_url,  # 替换为你的 API 地址
                            api_key=config.api_key  # 替换为你的 API 密钥
                        )
        self.index = faiss.IndexFlatIP(config.dim)
        self.dim = config.dim
        self.embeddings = []
        self.documents = []
        self.num_documents = 0
        self.index_path = config.index_path
        self.batch_size = config.batch_size

    def load_index(self, index_path: str = None):
        """
        Load the FAISS index from the specified path.

        Args:
            index_path (str, optional): The path to load the index from. Defaults to self.index_path.
        """
        if index_path is None:
            index_path = self.index_path
        # Load the document embeddings and texts from the saved file
        data = np.load(os.path.join(index_path, 'document.vecstore.npz'), allow_pickle=True)
        self.documents, self.embeddings = data['documents'].tolist(), data['embeddings'].tolist()
        # Load the FAISS index
        self.index = faiss.read_index(os.path.join(index_path, 'fassis.index'))
        print("Index loaded successfully from", index_path)
        del data  # Free up memory
        gc.collect()  # Perform garbage collection

    def save_index(self, index_path: str = None):
        """
        Save the FAISS index to the specified path.

        Args:
            index_path (str, optional): The path to save the index to. Defaults to self.index_path.
        """
        if self.index and self.embeddings and self.documents:
            if index_path is None:
                index_path = self.index_path
            # Create the directory if it doesn't exist
            if not os.path.exists(index_path):
                os.makedirs(index_path, exist_ok=True)
                print(f"Index saving to：{index_path}")
            # Save the document embeddings and texts
            np.savez(
                os.path.join(index_path, 'document.vecstore'),
                embeddings=self.embeddings,
                documents=self.documents
            )
            # Save the FAISS index
            faiss.write_index(self.index, os.path.join(index_path, 'fassis.index'))
            print("Index saved successfully to", index_path)

    def get_embedding(self, sentences: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of sentences.

        Args:
            sentences (List[str]): List of sentences to generate embeddings for.

        Returns:
            np.ndarray: A numpy array of embeddings.
        """
        # Using configured batch_size
        # return self.model.encode(sentences=sentences, batch_size=self.batch_size)
    
        #防止chunk为空字符串
        sentences = [sentence if sentence else "This is a none string." for sentence in sentences]

        response = self.client.embeddings.create(
            input=sentences,
            model=self.config.model_name
        )
        embedding = [np.array(item.embedding) for item in response.data]
        # 提取嵌入向量
        embedding = np.array(embedding)
        return embedding

    def add_texts(self, texts: List[str]):
        """
        Add multiple texts to the index.

        Args:
            texts (List[str]): List of texts to add to the index.
        """
        embeddings = self.get_embedding(texts)
        # Convert embeddings to float32 (required by FAISS)
        # faiss issue:https://github.com/facebookresearch/faiss/issues/1732
        self.index.add(embeddings.astype("float32"))
        self.documents.extend(texts)  # Add texts to the documents list
        self.embeddings.extend(embeddings)  # Add embeddings to the embeddings list
        self.num_documents += len(texts)  # Update the document count


    def add_text(self, text: str):
        """
        Add a single text to the index.

        Args:
            text (str): The text to add to the index.
        """
        self.add_texts([text])

    def build_from_texts(self, corpus: List[str]):
        """
        Process and index a list of texts in batches.

        Args:
            corpus (List[str]): List of texts to index.
        """
        if not corpus:
            return

        # Process texts in batches
        for i in tqdm(range(0, len(corpus), self.batch_size), desc="Building index"):
            batch = corpus[i:i + self.batch_size]
            self.add_texts(batch)


    def retrieve(self, query: str = None, top_k: int = 5) -> List[Dict[str, Union[str, float]]]:
        """
        Retrieve the top_k documents relevant to the query.

        Args:
            query (str, optional): The query string. Defaults to None.
            top_k (int, optional): The number of top documents to retrieve. Defaults to 5.

        Returns:
            List[Dict[str, Union[str, float]]]: A list of dictionaries containing the retrieved documents and their scores.
        """
        # generate query embedding
        query_embedding = self.get_embedding([query]).astype("float32")
        # search the index
        D, I = self.index.search(query_embedding, top_k)
        # free up memory
        del query_embedding
        # Return the retrieved documents with their scores
        return [{'text': self.documents[idx], 'score': score} for idx, score in zip(I[0], D[0])]