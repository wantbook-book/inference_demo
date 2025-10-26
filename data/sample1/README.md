# 多模态示例数据 (sample1)

本目录包含可直接用于 `/nfs9/fwk/project/multi_modal_inference.py` Gradio 界面的示例输入数据：

- `image.ppm`：8x8 彩色条纹图像（P3 PPM 格式，PIL 可读取）。
- `graph.json`：图拓扑结构，包含 `nodes` 与 `edges`。
- `timeseries.csv`：时序数据，列示例为 `timestamp,value`，解析器会读取数值列。
- `text.txt`：自由文本示例。

## 在界面中使用
1. 启动应用：
   - `pip install gradio pillow`
   - `python /nfs9/fwk/project/multi_modal_inference.py`
2. 在浏览器中打开页面后：
   - 图像输入：选择 `image.ppm`。
   - 图拓扑结构：上传 `graph.json` 或在文本框粘贴边列表。
   - 时序数据：上传 `timeseries.csv` 或在文本框粘贴数字列表。
   - 文本输入：将 `text.txt` 内容粘贴到文本框。
3. 点击“开始推理”，输出将基于 Mock 逻辑返回输入摘要说明。