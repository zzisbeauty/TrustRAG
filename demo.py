import gradio as gr

# 示例数据
contents = [
    {'text': '这是一段文本', 'score': 4.48828125, 'label': 1},
    {'text': '这是一段文本', 'score': 1.48828125, 'label': 1},
    {'text': '这是一段文本', 'score': 0, 'label': 0},
]


def predict():
    # 创建复选框组件列表
    checkboxes = []
    for item in contents:
        # 根据label决定是否选中
        checked = bool(item.get('label', 0))
        # 创建复选框的HTML
        checkbox = gr.Checkbox(value=checked, visible=True,label=item.get('text', ''),interactive=True)
        checkboxes.append(checkbox)
    return checkboxes


with gr.Blocks() as demo:
    # 创建一个容器来存放复选框
    with gr.Column() as checkbox_container:
        checkbox_outputs = [gr.Checkbox(visible=False,interactive=True) for _ in range(len(contents))]

    # 添加提交按钮
    submit_btn = gr.Button("提交")

    # 设置按钮点击事件
    submit_btn.click(
        fn=predict,
        inputs=None,
        outputs=checkbox_outputs
    )

# 启动应用
if __name__ == "__main__":
    demo.launch()


