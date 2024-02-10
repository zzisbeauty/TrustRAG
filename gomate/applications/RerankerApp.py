from gomate.modules import bge_large_reranker


class RerankerApp():
    """重排模块，评估文档的相关性并重新排序。把最有可能提供准确、相关回答的文档排在前面。

    实现包括
    1. bge-reranker-large。智源开源的Rerank模型。
    2. ...

    """

    def __init__(self, component_name=None):
        """Init required reranker according to component name."""
        self.reranker_list = ['bge_large']
        assert component_name in self.reranker_list
        if component_name == 'bge_large':
            self.reranker = bge_large_reranker()

    def run(self, query, contexts):
        """Run the required reranker"""
        if query is None:
            raise ValueError('missing query')
        if contexts is None:
            raise ValueError('missing contexts')
        return self.reranker.run(query, contexts)
