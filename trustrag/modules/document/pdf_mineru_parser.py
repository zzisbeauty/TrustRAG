import requests

class PdfParserWithMinerU:
    def __init__(self,url='http://localhost:8888/pdf_parse'):

        # 服务器URL
        self.url = url
    def parse(self,pdf_file_path,output_dir:str="output"):

        # PDF文件路径
        # pdf_file_path = 'path/to/your/file.pdf'
        print("正在基于MinerU解析pdf文件，请耐心等待，耗时时间较长。")
        # 请求参数
        params = {
            'parse_method': 'auto',
            'is_json_md_dump': 'true',
            'output_dir': output_dir
        }

        # 准备文件
        files = {
            'pdf_file': (pdf_file_path.split('/')[-1], open(pdf_file_path, 'rb'), 'application/pdf')
        }

        # 发送POST请求
        response = requests.post(self.url, params=params, files=files,timeout=2000)

        # 检查响应
        if response.status_code == 200:
            print("PDF解析成功")
            print(response.json())
            print(response.json().keys())
        else:
            print(f"错误: {response.status_code}")
            print(response.text)


if __name__ == '__main__':
    pdf_parser=PdfParserWithMinerU(url='https://aicloud.oneainexus.cn:30013/inference/aicloud-yanqiang/mineru/pdf_parse')
    pdf_file_path= '../../../data/docs/汽车操作手册.pdf'
    pdf_parser.parse(pdf_file_path)