from trustrag.modules.chunks.char_chunk import CharChunker

if __name__ == '__main__':
    with open("../../data/docs/news.txt","r",encoding="utf-8") as f:
        content=f.read()
    print(content)

    cc=CharChunker(chunk_size=64)

    chunks=cc.get_chunks([content])
    print(chunks)