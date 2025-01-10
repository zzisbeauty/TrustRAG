from trustrag.modules.chunks.base import BaseChunker
from typing import List

class CharChunker(BaseChunker):
    def __init__(self,chunk_size:int=64):
        super().__init__()
        self.chunk_size = chunk_size
    def get_chunks(self, texts:List[str]) -> List[str]:
        chunks= []
        for paragraph in texts:
            for i in range(0, len(paragraph), self.chunk_size):
                chunk = paragraph[i:i + self.chunk_size]
                chunks.append(chunk)
        return chunks


if __name__ == "__main__":
    cc=CharChunker(chunk_size=64)
    print(cc.get_chunks(["我喜欢北京。"]))