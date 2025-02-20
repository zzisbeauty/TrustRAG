#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: textparser_exmaple.py
@time: 2024/06/06
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: coding..
"""
from trustrag.modules.document.txt_parser import TextParser




if __name__ == '__main__':
    text_parser=TextParser()
    chunks = text_parser.parse(fnm="../../data/docs/sample.txt")
    print(len(chunks))

    for chunk in chunks:
        print("=="*100)
        print(chunk)