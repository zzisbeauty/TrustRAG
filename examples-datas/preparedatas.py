"""
todo 读取作文数据脚本，获取一批原始语料数据
"""

# def prepare_corpus():
#     import os
#     basePath = '/Users/chuizhishilizhuanjia/Documents/作文数据集'
#     txtLists = os.listdir(basePath)[:100]
#     with open('./作文数据集0-100.txt','a',encoding='utf-8') as fw:
#         for eachFile in txtLists:
#             eachFilePath = os.path.join(basePath, eachFile)
#             with open(eachFilePath,'r',encoding='utf-8') as fr:
#                 lines = fr.readlines()
#                 for line in lines:
#                     fw.write(line)
#                 fw.write('\n====================================\n')


# prepare_corpus()