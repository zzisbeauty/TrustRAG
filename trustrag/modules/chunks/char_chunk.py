from trustrag.modules.chunks.base import BaseChunker
from typing import List

class CharChunker(BaseChunker):
    """
    A character-based chunker that splits input texts into fixed-size chunks of characters.

    This class inherits from `BaseChunker` and implements the `get_chunks` method to divide
    input texts into smaller chunks, where each chunk contains a specified number of characters.
    This is useful for processing long texts in smaller, manageable pieces.

    Attributes:
        chunk_size (int): The number of characters per chunk. Defaults to 64.
    """

    def __init__(self, chunk_size: int = 64) -> None:
        """
        Initializes the CharChunker with a specified chunk size.

        Args:
            chunk_size (int): The number of characters per chunk. Defaults to 64.
        """
        super().__init__()
        self.chunk_size = chunk_size

    def get_chunks(self, paragraphs: List[str]) -> List[str]:
        """
        Splits the input paragraphs into chunks of characters based on the specified chunk size.

        Args:
            paragraphs (List[str]): A list of strings (paragraph) to be chunked.

        Returns:
            List[str]: A list of chunks, where each chunk is a string of characters.
        """
        chunks = []
        for paragraph in paragraphs:
            for i in range(0, len(paragraph), self.chunk_size):
                chunk = paragraph[i:i + self.chunk_size]
                chunks.append(chunk)
        return chunks


if __name__ == "__main__":
    cc = CharChunker(chunk_size=64)
    print(cc.get_chunks(["我喜欢北京。"]))