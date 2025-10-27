import os
import json
import csv
import random
import time
from typing import List, Tuple, Optional, Dict, Any

import gradio as gr

try:
    from PIL import Image
except Exception:
    Image = None


# ---------------- 基础解析 ----------------

def describe_image(img: Optional[Image.Image]) -> str:
    if img is None:
        return ""
    try:
        w, h = img.size
        mode = img.mode
        return f"[图像] 尺寸: {w}x{h}, 模式: {mode}"
    except Exception:
        return ""


def parse_graph_file(file_obj: Optional[gr.File]) -> Tuple[List[Tuple[str, str]], List[str]]:
    edges: List[Tuple[str, str]] = []
    nodes: List[str] = []
    if not file_obj:
        return edges, nodes
    try:
        path = file_obj.name if hasattr(file_obj, "name") else file_obj
        if not path or not os.path.exists(path):
            return edges, nodes
        ext = os.path.splitext(path)[1].lower()
        if ext == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                if "edges" in data and isinstance(data["edges"], list):
                    for e in data["edges"]:
                        if isinstance(e, (list, tuple)) and len(e) >= 2:
                            edges.append((str(e[0]), str(e[1])))
                if "nodes" in data and isinstance(data["nodes"], list):
                    nodes = [str(n) for n in data["nodes"]]
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, (list, tuple)) and len(item) >= 2 and isinstance(item[1], list):
                        src = str(item[0])
                        for dst in item[1]:
                            edges.append((src, str(dst)))
            if not nodes and edges:
                uniq = set()
                for s, d in edges:
                    uniq.add(s); uniq.add(d)
                nodes = sorted(list(uniq))
        elif ext in (".csv", ".tsv"):
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        edges.append((row[0].strip(), row[1].strip()))
            if edges:
                uniq = set()
                for s, d in edges:
                    uniq.add(s); uniq.add(d)
                nodes = sorted(list(uniq))
        else:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        edges.append((parts[0], parts[1]))
            if edges:
                uniq = set()
                for s, d in edges:
                    uniq.add(s); uniq.add(d)
                nodes = sorted(list(uniq))
    except Exception:
        pass
    return edges, nodes


def parse_timeseries_file(file_obj: Optional[gr.File]) -> List[float]:
    values: List[float] = []
    if not file_obj:
        return values
    try:
        path = file_obj.name if hasattr(file_obj, "name") else file_obj
        if not path or not os.path.exists(path):
            return values
        ext = os.path.splitext(path)[1].lower()
        if ext == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for v in data:
                    try:
                        values.append(float(v))
                    except Exception:
                        pass
            elif isinstance(data, dict):
                for v in data.values():
                    try:
                        values.append(float(v))
                    except Exception:
                        pass
        else:
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    for cell in row:
                        try:
                            values.append(float(cell))
                        except Exception:
                            pass
    except Exception:
        pass
    return values


def read_txt_file_content(file_obj: Optional[gr.File]) -> str:
    if not file_obj:
        return ""
    try:
        path = file_obj.name if hasattr(file_obj, "name") else file_obj
        if not path or not os.path.exists(path):
            return ""
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


# ---------------- Input/Output 解析与配对 ----------------

def parse_io_text(content: str) -> Tuple[str, str]:
    """
    从一个TXT文本中抽取 Input: 与 Output: 两段内容。
    - Input: 后直到 Output: 之前的所有文本为 input_text
    - Output: 后直到文件结尾的所有文本为 output_text
    支持包含 块和 Answer: 行；不做清洗，原样保留。
    """
    if not content:
        return "", ""
    lines = content.splitlines()
    input_lines: List[str] = []
    output_lines: List[str] = []
    mode: Optional[str] = None
    for line in lines:
        stripped = line.strip()
        low = stripped.lower()
        if low.startswith("input:"):
            mode = "input"
            # 同行后半部分
            rest = line.split(":", 1)[1] if ":" in line else ""
            if rest.strip():
                input_lines.append(rest.strip())
            continue
        if low.startswith("output:"):
            mode = "output"
            rest = line.split(":", 1)[1] if ":" in line else ""
            if rest.strip():
                output_lines.append(rest.lstrip())
            continue
        if mode == "input":
            input_lines.append(line)
        elif mode == "output":
            output_lines.append(line)
        else:
            # 未进入任何段，忽略
            pass
    return "\n".join(input_lines).strip(), "\n".join(output_lines).strip()


def parse_io_file(file_obj: Optional[gr.File]) -> Tuple[str, str]:
    content = read_txt_file_content(file_obj)
    return parse_io_text(content)


def find_paired_output_file(user_text_file: Optional[gr.File]) -> Optional[str]:
    if not user_text_file:
        return None
    try:
        in_path = user_text_file.name if hasattr(user_text_file, "name") else user_text_file
        if not in_path or not os.path.exists(in_path):
            return None
        base, _ = os.path.splitext(in_path)
        candidates = [base + ".out.txt", base + "_out.txt", base + "-out.txt"]
        for c in candidates:
            if os.path.exists(c):
                return c
        return None
    except Exception:
        return None


# ---------------- 模型加载（Mock） ----------------

def load_llm(model_path: str, device: str = "cpu") -> Tuple[Dict[str, Any], str]:
    state: Dict[str, Any] = {
        "type": "mock",
        "model": None,
        "tokenizer": None,
        "device": device,
        "model_path": model_path,
    }
    info = f"已加载模型，模型路径: {model_path or '(未提供)'}"
    return state, info


# ---------------- 精美UI构建 ----------------

def build_ui() -> gr.Blocks:
    theme = gr.themes.Soft(primary_hue="violet", secondary_hue="indigo").set(
        body_text_color="#121322",
        background_fill_primary="#f5f7ff",
        button_primary_background_fill="#6C5CE7",
        button_primary_text_color="#ffffff",
        button_secondary_background_fill="#ffffff",
        input_background_fill="#ffffff",
    )

    css = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

    body {
        font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, 'PingFang SC', 'Microsoft YaHei', sans-serif;
        background:
          radial-gradient(900px 540px at 10% -10%, #e6e7ff 0%, #f7f8ff 40%, #ffffff 100%),
          linear-gradient(120deg, #eef1ff 0%, #fbfcff 50%, #ffffff 100%);
    }
    .gradio-container {
        width: min(96vw, 1480px) !important;
        margin: 0 auto !important;
        padding: 0 8px !important;
    }

    .hero {
        position: relative;
        background: radial-gradient(1000px 500px at 0% 0%, #9b8cf6 0%, #6C5CE7 35%, #5a67d8 70%, #4fa3e3 100%);
        color: white;
        border-radius: 28px;
        padding: 32px 34px;
        margin-bottom: 22px;
        overflow: hidden;
    }
    .hero h1 { font-size: 32px; letter-spacing: 0.2px; margin: 0; }
    .hero::before, .hero::after {
        content: '';
        position: absolute;
        border-radius: 50%;
        filter: blur(16px);
        opacity: 0.25;
        animation: float 9s ease-in-out infinite;
    }
    .hero::before {
        width: 300px; height: 300px; right: -100px; top: -100px;
        background: radial-gradient(circle, rgba(255,255,255,0.7), transparent 60%);
        animation-delay: 0.5s;
    }
    .hero::after {
        width: 240px; height: 240px; left: -80px; bottom: -80px;
        background: radial-gradient(circle, rgba(255,255,255,0.6), transparent 60%);
    }
    @keyframes float {
        0% { transform: translateY(0px) scale(1); }
        50% { transform: translateY(-4px) scale(1.02); }
        100% { transform: translateY(0px) scale(1); }
    }

    .card {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 20px;
        padding: 16px;
        box-shadow: 0 18px 40px rgba(40, 50, 80, 0.08);
        border: 1px solid rgba(190, 200, 255, 0.6);
        backdrop-filter: blur(8px);
    }
    .flow-card {
        background: linear-gradient(180deg, #ffffff 0%, #f9fbff 100%);
        border: 1px solid #e8ecff;
        border-radius: 24px;
        padding: 14px;
        box-shadow: 0 12px 28px rgba(70, 80, 140, 0.08);
        margin-bottom: 14px;
    }
    .section-title {
        font-weight: 800; font-size: 16px; color: #222741;
        display: inline-block; margin-bottom: 8px;
        border-bottom: 3px solid #6C5CE7; padding-bottom: 4px;
    }

    .mono-box textarea, .mono-box input {
        font-family: 'Inter', ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        line-height: 1.5; font-size: 14px;
        border-radius: 14px !important;
        border: 1px solid #e1e6ff !important;
        background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(248,250,255,0.96));
    }

    .btn-row { gap: 12px; }
    button.svelte-1ipelgc { transition: all 0.2s ease; }
    button.svelte-1ipelgc:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 18px rgba(108, 92, 231, 0.3);
    }

    /* 顶部右侧消息弹窗 */
    #toast_box {
        position: fixed;
        top: 16px;
        right: 16px;
        z-index: 9999;
        pointer-events: none;
    }
    .toast {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: #ffffff;
        border: 1px solid #e8ecff;
        box-shadow: 0 12px 28px rgba(70, 80, 140, 0.12);
        border-radius: 14px;
        padding: 10px 14px;
        color: #222741;
        font-weight: 600;
        pointer-events: auto;
    }
    .toast-success {
        border-color: #c9f1d7;
        background: linear-gradient(180deg, #ffffff 0%, #f6fff9 100%);
    }
    .spinner {
        width: 16px;
        height: 16px;
        border: 2px solid #dde3ff;
        border-top-color: #6C5CE7;
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
    }
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    """

    with gr.Blocks(
        title="面向配电系统运行优化的多模态大模型构建技术",
        theme=theme,
        css=css
    ) as demo:

        gr.HTML("""
        <div class="hero">
          <h1>面向配电系统运行优化的多模态大模型推理演示</h1>
        </div>
        """)

        # 右上角消息弹窗容器（初始隐藏）
        toast_html = gr.HTML("", visible=False, elem_id="toast_box")

        with gr.Group(elem_classes="flow-card"):
            flow_img = gr.Image(
                value=r"./图片1(1).png",  #change to your local path
                label="概览图",
                interactive=False,
                show_download_button=False,
                show_share_button=False,
            )
        # 左侧：模型与参数
        with gr.Group(elem_classes="card"):
            gr.Markdown("<span class='section-title'>模型配置</span>")
            model_path = gr.Textbox(label="模型路径", placeholder="/path/to/model")
            load_btn = gr.Button("加载模型", variant="primary")
            load_info = gr.Markdown("状态：等待加载")
            state_llm = gr.State(value=None)
        with gr.Group(elem_classes="card"):
            gr.Markdown("<span class='section-title'>参数</span>")
            temperature = gr.Slider(label="temperature", value=0.7, minimum=0.01, maximum=1.5, step=0.01)
            top_p = gr.Slider(label="top_p", value=0.9, minimum=0.01, maximum=1.0, step=0.01)
            max_new_tokens = gr.Slider(label="max_new_tokens", value=4096, minimum=32, maximum=4096, step=1)
            seed = gr.Number(label="seed", value=-1)

        # 右侧：输入与输出
        with gr.Group(elem_classes="card"):
            gr.Markdown("<span class='section-title'>输入选择</span>")
            modality_selector = gr.CheckboxGroup(
                choices=["图像输入", "图拓扑结构输入", "时序数据输入"],
                value=[],
                label="选择需要的输入项"
            )

        with gr.Group(elem_classes="card mono-box") as grp_text:
            gr.Markdown("<span class='section-title'>文本</span>")
            user_text_file = gr.File(label="上传文本TXT", file_types=[".txt"], type="filepath")
            user_text = gr.Textbox(lines=8, label="文本")

        with gr.Group(elem_classes="card", visible=False) as grp_image:
            gr.Markdown("<span class='section-title'>图像输入</span>")
            image_in = gr.Image(type="pil", label="图像文件")

        with gr.Group(elem_classes="card mono-box", visible=False) as grp_graph:
            gr.Markdown("<span class='section-title'>图拓扑结构输入</span>")
            graph_file = gr.File(label="图文件（JSON/CSV/TXT）", type="filepath")
            graph_text_file = gr.File(label="图边TXT文件", file_types=[".txt"], type="filepath")
            graph_text = gr.Textbox(lines=6, label="图边文本")

        with gr.Group(elem_classes="card mono-box", visible=False) as grp_ts:
            gr.Markdown("<span class='section-title'>时序数据输入</span>")
            ts_file = gr.File(label="时序数据文件（JSON/CSV/TXT）", type="filepath")
            ts_text_file = gr.File(label="时序TXT文件", file_types=[".txt"], type="filepath")
            ts_text = gr.Textbox(lines=4, label="时序数据")

        with gr.Group(elem_classes="card mono-box"):
            with gr.Row(elem_classes="btn-row"):
                run_btn = gr.Button("开始推理", variant="primary")
                clear_btn = gr.Button("清空输出")
            output_text = gr.Textbox(lines=16, label="输出")

        # 加载模型（带状态与右上角弹窗）
        def handle_load(m_path: str):
            # 第一次更新：左侧状态 + 右上角转圈弹窗
            spinner_msg = "<div class='toast'><div class='spinner'></div><div>正在加载模型...</div></div>"
            yield None, "状态：正在加载...", gr.update(value=spinner_msg, visible=True)

            # 模拟等待
            time.sleep(1.2)

            # 实际加载（Mock）
            state, info = load_llm(m_path)

            # 第二次更新：左侧状态 + 右上角成功弹窗
            success_msg = "<div class='toast toast-success'><div>✅</div><div>模型加载成功</div></div>"
            yield state, info, gr.update(value=success_msg, visible=True)

            # 片刻后收起弹窗
            time.sleep(1.6)
            yield state, info, gr.update(value="", visible=False)

        load_btn.click(handle_load, inputs=[model_path], outputs=[state_llm, load_info, toast_html])

        # 切换输入栏可见性
        def toggle_inputs(selection: List[str]):
            img_v = "图像输入" in selection
            g_v = "图拓扑结构输入" in selection
            ts_v = "时序数据输入" in selection
            return (
                gr.update(visible=img_v),
                gr.update(visible=g_v),
                gr.update(visible=ts_v),
            )

        modality_selector.change(
            toggle_inputs,
            inputs=[modality_selector],
            outputs=[grp_image, grp_graph, grp_ts],
        )

        # 上传文本TXT后，自动将 Input: 内容填入文本框
        def load_user_text(u_text_file):
            in_text, out_text = parse_io_file(u_text_file)
            # 若未找到Input段，回退为文件全文
            value = in_text if in_text else read_txt_file_content(u_text_file)
            return gr.update(value=value or "")
        user_text_file.change(load_user_text, inputs=[user_text_file], outputs=[user_text])

        # 增量输出逻辑：严格使用TXT内 Output: 段；如缺失则按配对文件；仍缺失则空输出
        def on_run_stream(st_llm,
                          selection: List[str],
                          img,
                          g_file, g_text_file, g_text_manual,
                          t_file, t_text_file, t_text_manual,
                          u_text_file, u_text_manual,
                          temp, tp, mnt, sd):

            # 随机种子（演示）
            seed_val = int(sd) if sd is not None else -1
            if seed_val is not None and seed_val >= 0:
                random.seed(seed_val)
                try:
                    import torch
                    torch.manual_seed(seed_val)
                except Exception:
                    pass

            # 解析上传TXT的 Input/Output 段
            input_text, output_text_src = parse_io_file(u_text_file)

            # 输出优先使用 Output 段；若无则尝试配对同名输出文件
            if not output_text_src:
                paired = find_paired_output_file(u_text_file)
                if paired and os.path.exists(paired):
                    try:
                        with open(paired, "r", encoding="utf-8") as f:
                            output_text_src = f.read()
                    except Exception:
                        output_text_src = ""

            # 仍无则输出为空（不再加入任何占位或摘要）
            output_text_src = output_text_src or ""

            # 逐字增量输出
            acc = ""
            try:
                limit = max(64, min(len(output_text_src), int(mnt)))
            except Exception:
                limit = len(output_text_src)
            for ch in output_text_src[:limit]:
                acc += ch
                yield acc
                sleep_t = max(0.002, 0.016 - (float(temp) - 0.01) * 0.009 + (float(tp) - 0.01) * 0.004)
                time.sleep(sleep_t)

        # 事件绑定（包含模型状态）
        run_btn.click(
            on_run_stream,
            inputs=[
                state_llm,
                modality_selector,
                image_in,
                graph_file, graph_text_file, graph_text,
                ts_file, ts_text_file, ts_text,
                user_text_file, user_text,
                temperature, top_p, max_new_tokens, seed
            ],
            outputs=[output_text],
        )

        def do_clear():
            return ""
        clear_btn.click(do_clear, inputs=[], outputs=[output_text])

    return demo


if __name__ == "__main__":
    app = build_ui()
    try:
        app.queue()
    except TypeError:
        pass
    app.launch(server_name="0.0.0.0", server_port=7860, show_error=True)

