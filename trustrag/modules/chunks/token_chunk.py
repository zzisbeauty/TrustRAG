from typing import List, Callable, Optional
from trustrag.modules.chunks.base import BaseChunker
from trustrag.modules.document.rag_tokenizer import RagTokenizer
import jieba
from transformers import AutoTokenizer

class TokenChunker(BaseChunker):
    def __init__(
            self,
            chunk_size: int = 64,
            tokenizer_type: str = "rag",
            model_name_or_path: Optional[str] = None
    ):
        """
        Initialize the TokenChunker.

        :param chunk_size: The number of tokens per chunk, default is 64.
        :param tokenizer_type: The type of tokenizer, supports "rag", "jieba", and "hf", default is "rag".
        :param model_name_or_path: When tokenizer_type is "hf", specify the model name or path for Hugging Face.
        """
        super().__init__()
        self.chunk_size = chunk_size
        self.tokenizer_func = self.init_tokenizer(tokenizer_type, model_name_or_path)

    def init_tokenizer(
            self,
            tokenizer_type: str = "rag",
            model_name_or_path: Optional[str] = None
    ) -> Callable[[str], List[str]]:
        """
        Initialize the tokenizer.

        :param tokenizer_type: The type of tokenizer, supports "rag", "jieba", and "hf".
        :param model_name_or_path: When tokenizer_type is "hf", specify the model name or path for Hugging Face.
        :return: A tokenizer function that takes a string as input and returns a list of tokens.
        """
        if tokenizer_type == "rag":
            return RagTokenizer().tokenize
        elif tokenizer_type == "jieba":
            return lambda text: list(jieba.cut(text))
        elif tokenizer_type == "hf":
            if model_name_or_path is None:
                model_name_or_path = "Qwen/Qwen2-0.5B-Instruct"  # Default model
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            return lambda text: [tokenizer.decode([token]) for token in tokenizer.encode(text)]
        else:
            raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")

    def get_chunks(self, paragraphs: List[str]) -> List[str]:
        """
        Split paragraphs into chunks of the specified size.

        :param paragraphs: A list of input paragraphs.
        :return: A list of chunks after splitting.
        """
        chunks = []
        for paragraph in paragraphs:
            tokens = self.tokenizer_func(paragraph)
            tokens = [token for token in tokens]  # Ensure tokens is a list
            print(tokens)  # Print tokenized results (for debugging)
            for i in range(0, len(tokens), self.chunk_size):
                chunk = "".join(tokens[i:i + self.chunk_size])
                chunks.append(chunk)
        return chunks

if __name__ == '__main__':
    with open("../../../data/docs/news.txt","r",encoding="utf-8") as f:
        content=f.read()
    # print(content)
    tc=TokenChunker(chunk_size=62,tokenizer_type="jieba")
    chunks = tc.get_chunks([content])
    for chunk in chunks:
        print(f"Chunk Content：\n{chunk}")
