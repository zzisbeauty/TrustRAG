from trustrag.modules.document.common_parser import CommonParser
from trustrag.modules.document.chunk import TextChunker
if __name__ == '__main__':
    cp=CommonParser()
    tc=TextChunker()
    # contents=cp.parse("../../data/docs/基础知识.md")
    paragraphs=cp.parse("../../data/docs/5G垂直行业基础知识介绍--口袋小册子.pdf")
    chunks=tc.chunk_sentences(paragraphs,chunk_size=256)
    print(chunks)
    print(len(chunks))

    for chunk in chunks:
        print(chunk)
        print("+++"*100)