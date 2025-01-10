from trustrag.modules.chunks.base import BaseChunker
from typing import List
from trustrag.modules.document.rag_tokenizer import RagTokenizer
class TokenChunker(BaseChunker):
    def __init__(self,chunk_size:int=64,tokenizer_name="rag"):
        super().__init__()
        self.chunk_size = chunk_size
        self.tokenizer_func = self.init_tokenizer(tokenizer_name)

    def init_tokenizer(self,tokenizer_name="rag"):
        if tokenizer_name=="rag":
            self.tokenizer_func = RagTokenizer()
        return self.tokenizer_func

    def get_chunks(self, paragraphs:List[str]) -> List[str]:
        chunks= []
        for paragraph in paragraphs:
            tokens = self.tokenizer_func.tokenize(paragraph)
            for i in range(0, len(tokens), self.chunk_size):
                chunk = "".join(tokens[i:i + self.chunk_size])
                chunks.append(chunk)
        return chunks



if __name__ == '__main__':
    with open("../../../data/docs/news.txt","r",encoding="utf-8") as f:
        content=f.read()
    print(content)
    tc=TokenChunker()
    chunks = tc.get_chunks([content])
    print(chunks)