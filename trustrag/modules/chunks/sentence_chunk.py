import re
from trustrag.modules.document import rag_tokenizer
from trustrag.modules.chunks.base import BaseChunker

class SentenceChunker(BaseChunker):
    """
    A class for splitting text into chunks based on sentences, ensuring each chunk does not exceed a specified token size.

    This class is designed to handle both Chinese and English text, splitting it into sentences using punctuation marks.
    It then groups these sentences into chunks, ensuring that the total number of tokens in each chunk does not exceed
    the specified `chunk_size`. The class also provides methods to preprocess the text chunks by normalizing excessive
    newlines and spaces.

    Attributes:
        tokenizer (callable): A tokenizer function used to count tokens in sentences.
        chunk_size (int): The maximum number of tokens allowed per chunk.

    Methods:
        split_sentences(text: str) -> list[str]:
            Splits the input text into sentences based on Chinese and English punctuation marks.

        process_text_chunks(chunks: list[str]) -> list[str]:
            Preprocesses text chunks by normalizing excessive newlines and spaces.

        get_chunks(paragraphs: list[str]) -> list[str]:
            Splits a list of paragraphs into chunks based on a specified token size.
    """

    def __init__(self, chunk_size=512):
        """
        Initializes the SentenceChunker with a tokenizer and a specified chunk size.

        Args:
            chunk_size (int, optional): The maximum number of tokens allowed per chunk. Defaults to 512.
        """
        super().__init__()
        self.tokenizer = rag_tokenizer
        self.chunk_size = chunk_size

    def split_sentences(self, text: str) -> list[str]:
        """
        Splits the input text into sentences based on Chinese and English punctuation marks.

        Args:
            text (str): The input text to be split into sentences.

        Returns:
            list[str]: A list of sentences extracted from the input text.
        """
        # Use regex to split text by sentence-ending punctuation marks
        sentence_endings = re.compile(r'([。！？.!?])')
        sentences = sentence_endings.split(text)

        # Merge punctuation marks with their preceding sentences
        result = []
        for i in range(0, len(sentences) - 1, 2):
            if sentences[i]:
                result.append(sentences[i] + sentences[i + 1])

        # Handle the last sentence if it lacks punctuation
        if sentences[-1]:
            result.append(sentences[-1])

        # Remove whitespace and filter out empty sentences
        result = [sentence.strip() for sentence in result if sentence.strip()]

        return result

    def process_text_chunks(self, chunks: list[str]) -> list[str]:
        """
        Preprocesses text chunks by normalizing excessive newlines and spaces.

        Args:
            chunks (list[str]): A list of text chunks to be processed.

        Returns:
            list[str]: A list of processed text chunks with normalized formatting.
        """
        processed_chunks = []
        for chunk in chunks:
            # Normalize four or more consecutive newlines
            while '\n\n\n\n' in chunk:
                chunk = chunk.replace('\n\n\n\n', '\n\n')

            # Normalize four or more consecutive spaces
            while '    ' in chunk:
                chunk = chunk.replace('    ', '  ')

            processed_chunks.append(chunk)

        return processed_chunks

    def get_chunks(self, paragraphs: list[str]) -> list[str]:
        """
        Splits a list of paragraphs into chunks based on a specified token size.

        Args:
            paragraphs (list[str]|str): A list of paragraphs to be chunked.

        Returns:
            list[str]: A list of text chunks, each containing sentences that fit within the token limit.
        """
        # Combine paragraphs into a single text
        text = ''.join(paragraphs)

        # Split the text into sentences
        sentences = self.split_sentences(text)

        # If no sentences are found, treat paragraphs as sentences
        if len(sentences) == 0:
            sentences = paragraphs

        chunks = []
        current_chunk = []
        current_chunk_tokens = 0

        # Iterate through sentences and build chunks based on token count
        for sentence in sentences:
            tokens = self.tokenizer.tokenize(sentence)
            if current_chunk_tokens + len(tokens) <= self.chunk_size:
                # Add sentence to the current chunk if it fits
                current_chunk.append(sentence)
                current_chunk_tokens += len(tokens)
            else:
                # Finalize the current chunk and start a new one
                chunks.append(''.join(current_chunk))
                current_chunk = [sentence]
                current_chunk_tokens = len(tokens)

        # Add the last chunk if it contains any sentences
        if current_chunk:
            chunks.append(''.join(current_chunk))

        # Preprocess the chunks to normalize formatting
        chunks = self.process_text_chunks(chunks)
        return chunks

if __name__ == '__main__':
    with open("../../../data/docs/测试新闻2.txt","r",encoding="utf-8") as f:
        content=f.read()
    tc=SentenceChunker(chunk_size=128)
    chunks = tc.get_chunks([content])
    for chunk in chunks:
        print(f"Chunk Content：\n{chunk}")
