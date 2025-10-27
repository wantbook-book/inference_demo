
import os
import json
import csv
import random
import time
import traceback
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


# ---------------- 关联图像查找 ----------------

def find_associated_image(file_obj_or_path: Optional[str]) -> Optional[str]:
    if not file_obj_or_path:
        return None
    try:
        print(file_obj_or_path)
        filename = os.path.basename(file_obj_or_path)
        base, _ = os.path.splitext(filename)

        folder = os.path.dirname(file_obj_or_path)
        candidates = [os.path.join(folder, base + ext) for ext in [".jpg", ".png", ".jpeg"]]

        cwd_candidates = [os.path.join(os.getcwd(), base + ext) for ext in [".jpg", ".png", ".jpeg"]]
        candidates += cwd_candidates
        print(candidates)
        for c in candidates:
            if os.path.exists(c):
                return c
    except Exception:
        pass
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


# ---------------- 精美UI构建（双列布局 + 字体增强） ----------------

def build_ui() -> gr.Blocks:
    theme = gr.themes.Soft(primary_hue="violet", secondary_hue="indigo").set(
        body_text_color="#101323",
        background_fill_primary="#f5f7ff",
        button_primary_background_fill="#6C5CE7",
        button_primary_text_color="#ffffff",
        button_secondary_background_fill="#ffffff",
        input_background_fill="#ffffff",
    )

    css = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;600;700;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    :root {
        --font-size-base: 16px;
        --font-size-large: 18px;
        --font-size-xlarge: 22px;
        --font-size-hero: 42px;
    }

    body {
        font-family: 'Noto Sans SC', 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, 'PingFang SC', 'Microsoft YaHei', sans-serif;
        font-size: var(--font-size-base);
        background:
          radial-gradient(900px 540px at 10% -10%, #e6e7ff 0%, #f7f8ff 40%, #ffffff 100%),
          linear-gradient(120deg, #eef1ff 0%, #fbfcff 50%, #ffffff 100%);
    }
    .gradio-container {
        width: min(96vw, 1480px) !important;
        margin: 0 auto !important;
        padding: 0 8px !important;
        font-size: var(--font-size-base);
    }

    .hero {
        position: relative;
        background: radial-gradient(1000px 500px at 0% 0%, #9b8cf6 0%, #6C5CE7 35%, #5a67d8 70%, #4fa3e3 100%);
        color: white;
        border-radius: 28px;
        padding: 36px 38px;
        margin-bottom: 22px;
        overflow: hidden;
    }
    .hero h1 {
        font-size: var(--font-size-hero);
        font-weight: 900;
        letter-spacing: 0.4px;
        margin: 0;
    }
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
        padding: 18px;
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
        font-weight: 900;
        font-size: var(--font-size-xlarge);
        color: #1f2547;
        display: inline-block;
        margin-bottom: 10px;
        border-bottom: 4px solid #6C5CE7;
        padding-bottom: 6px;
        letter-spacing: 0.3px;
    }

    /* 标签字体整体放大 */
    .gradio-container .label, .gradio-container label {
        font-size: var(--font-size-large) !important;
        font-weight: 700 !important;
        color: #1d2341;
        letter-spacing: 0.2px;
    }
    .gradio-container .checklist .item label,
    .gradio-container .checkbox label,
    .gradio-container .radio label,
    .gradio-container .wrap .item label {
        font-size: var(--font-size-large) !important;
        font-weight: 700 !important;
    }

    /* 进一步确保特定组件标签（如“文本内容”“输出结果”）更大 */
    .big-label .label, .big-label label {
        font-size: calc(var(--font-size-large) * 1.05) !important;
        font-weight: 800 !important;
    }

    /* 文本框内容英文字体更美观，略微增大 */
    .mono-box textarea, .mono-box input {
        font-family: 'Inter', ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        line-height: 1.6;
        font-size: 15.8px;
        border-radius: 14px !important;
        border: 1px solid #e1e6ff !important;
        background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(248,250,255,0.96));
    }
    .pretty-text textarea, .pretty-output textarea {
        font-family: 'JetBrains Mono', 'Inter', ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace !important;
        font-size: 16.6px !important;
        line-height: 1.72 !important;
        letter-spacing: 0.2px !important;
    }

    .btn-row { gap: 12px; }

    /* ============ 新增：三类按钮的统一美化（加载模型、开始推理、清空输出） ============ */

    .btn-load, .btn-run, .btn-clear {
    border-radius: 12px !important;
    font-size: var(--font-size-large) !important;
    font-weight: 800 !important;
    color: #ffffff !important;
    border: none !important;
    box-shadow: 0 12px 24px rgba(70, 80, 140, 0.12) !important;
    transition: transform 0.15s ease, box-shadow 0.2s ease, filter 0.2s ease !important;
    }
    .btn-load > button, .btn-run > button, .btn-clear > button { /* 兼容“外层容器 + 内部 button”结构 */
    border-radius: 12px !important;
    font-size: var(--font-size-large) !important;
    font-weight: 800 !important;
    color: #ffffff !important;
    border: none !important;
    box-shadow: 0 12px 24px rgba(70, 80, 140, 0.12) !important;
    transition: transform 0.15s ease, box-shadow 0.2s ease, filter 0.2s ease !important;
    }


    .btn-load, .btn-load > button {
    background: linear-gradient(90deg, #4FA3E3 0%, #6C5CE7 100%) !important;
    }
    .btn-run, .btn-run > button {
    background: linear-gradient(90deg, #6C5CE7 0%, #5A67D8 100%) !important;
    }
    .btn-clear, .btn-clear > button {
    background: linear-gradient(90deg, #5EA9F8 0%, #8A79FF 100%) !important;
    }


    .btn-load:hover, .btn-run:hover, .btn-clear:hover,
    .btn-load > button:hover, .btn-run > button:hover, .btn-clear > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 16px 30px rgba(108, 92, 231, 0.30) !important;
    filter: brightness(1.03) !important;
    }
    .btn-load:active, .btn-run:active, .btn-clear:active,
    .btn-load > button:active, .btn-run > button:active, .btn-clear > button:active {
    transform: translateY(0) !important;
    box-shadow: 0 8px 18px rgba(108, 92, 231, 0.25) !important;
    filter: brightness(0.98) !important;
    }
    .btn-load:focus-visible, .btn-run:focus-visible, .btn-clear:focus-visible,
    .btn-load > button:focus-visible, .btn-run > button:focus-visible, .btn-clear > button:focus-visible {
    outline: 2px solid rgba(108, 92, 231, 0.66) !important;
    outline-offset: 2px !important;
    }
    

    /* 顶部右侧消息弹窗（保留样式） */
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
        font-weight: 800;
        pointer-events: auto;
        font-size: var(--font-size-large);
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

        with gr.Group(elem_classes="flow-card"):
            flow_img = gr.Image(
                value=r"./图片1(1).png",
                label="系统概览图",
                interactive=False,
                show_download_button=False,
                show_share_button=False,
            )

        
        with gr.Group(elem_classes="card"):
            gr.Markdown("<span class='section-title'>模型配置</span>")
            model_path = gr.Textbox(label="模型路径", placeholder="/path/to/model")
            # 修改：加载模型按钮添加类名，使用自定义样式
            load_btn = gr.Button("加载模型",  elem_classes=["btn-load"])
            load_info = gr.Markdown("状态：等待加载")
            state_llm = gr.State(value=None)

        with gr.Group(elem_classes="card"):
            gr.Markdown("<span class='section-title'>参数</span>")
            temperature = gr.Slider(label="temperature", value=0.7, minimum=0.01, maximum=1.5, step=0.01)
            top_p = gr.Slider(label="top_p", value=0.9, minimum=0.01, maximum=1.0, step=0.01)
            max_new_tokens = gr.Slider(label="max_new_tokens", value=4096, minimum=32, maximum=4096, step=1)
            seed = gr.Number(label="seed", value=-1)

        with gr.Group(elem_classes="card"):
            gr.Markdown("<span class='section-title'>输入选择</span>")
            modality_selector = gr.CheckboxGroup(
                choices=["图像输入", "图拓扑结构输入", "时序数据输入"],
                value=[],
                label="选择需要的输入项"
            )

        with gr.Group(elem_classes="card mono-box") as grp_text:
            gr.Markdown("<span class='section-title'>文本</span>")
            user_text_file = gr.File(label="上传文本（TXT）", file_types=[".txt"], type="filepath")
            user_text = gr.Textbox(lines=8, label="文本内容", elem_classes=["big-label", "pretty-text"])

        with gr.Group(elem_classes="card", visible=False) as grp_image:
            gr.Markdown("<span class='section-title'>图像输入</span>")
            image_in = gr.Image(type="pil", label="图像文件")

        # 图拓扑结构：仅文件选择 + 关联图像预览
        with gr.Group(elem_classes="card mono-box", visible=False) as grp_graph:
            gr.Markdown("<span class='section-title'>图拓扑结构输入</span>")
            graph_file = gr.File(label="图文件（JSON/CSV）", file_types=[".json", ".csv"], type="filepath")
            graph_image_preview = gr.Image(
                label="关联图像预览",
                interactive=False,
                visible=False,
                show_download_button=False,
                show_share_button=False
            )

        # 时序数据：仅文件选择 + 关联图像预览
        with gr.Group(elem_classes="card mono-box", visible=False) as grp_ts:
            gr.Markdown("<span class='section-title'>时序数据输入</span>")
            ts_file = gr.File(label="时序数据文件（JSON/CSV）", file_types=[".json", ".csv"], type="filepath")
            ts_image_preview = gr.Image(
                label="关联图像预览",
                interactive=False,
                visible=False,
                show_download_button=False,
                show_share_button=False
            )

        with gr.Group(elem_classes="card mono-box"):
            with gr.Row(elem_classes="btn-row"):
                # 修改：开始推理、清空输出按钮添加类名，使用自定义样式
                run_btn = gr.Button("开始推理", elem_classes=["btn-run"])
                clear_btn = gr.Button("清空输出", elem_classes=["btn-clear"])
            output_text = gr.Textbox(lines=16, label="输出结果", elem_classes=["big-label", "pretty-output"])

        # 加载模型：使用 gr.Info + 进度条，完成后再显示“成功”与详细信息
        def handle_load(m_path: str, current_state, progress=gr.Progress(track_tqdm=True)):
            try:
                # 顶部右侧提示
                gr.Info("🔄 正在加载模型，请稍候...")
                # 立刻将状态区域置为“正在加载...”，不显示成功信息
                yield current_state, "状态：正在加载..."

                # 释放之前的模型（如有）
                try:
                    if current_state is not None:
                        del current_state
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                except Exception:
                    pass

                # 模拟可视化加载进度（顶部全局进度条）
                steps = 8
                for i in range(steps):
                    progress((i + 1) / steps, desc="Loading...")
                    time.sleep(0.18)

                # 实际加载
                state, info = load_llm(m_path or "")

                # 进度“跑完后”才输出成功信息
                yield state, f"✅ 模型加载成功\n{info}"
            except Exception as e:
                yield current_state, f"❌ 模型加载失败: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"

        load_btn.click(handle_load, inputs=[model_path, state_llm], outputs=[state_llm, load_info])

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
            value = in_text if in_text else read_txt_file_content(u_text_file)
            return gr.update(value=value or "")
        user_text_file.change(load_user_text, inputs=[user_text_file], outputs=[user_text])

        # 图拓扑文件改变时，自动加载同名 jpg/png 关联图像
        def on_graph_file_change(g_file_path: Optional[str]):
            img_path = find_associated_image(g_file_path)
            if img_path:
                print('find', img_path)
                return gr.update(value=img_path, visible=True)
            else:
                print('not find', img_path)
                return gr.update(value=None, visible=False)
        graph_file.change(on_graph_file_change, inputs=[graph_file], outputs=[graph_image_preview])

        # 时序文件改变时，自动加载同名 jpg/png 关联图像
        def on_ts_file_change(t_file_path: Optional[str]):
            img_path = find_associated_image(t_file_path)
            if img_path:
                print('find', img_path)
                return gr.update(value=img_path, visible=True)
            else:
                print('not find', img_path)
                return gr.update(value=None, visible=False)
        ts_file.change(on_ts_file_change, inputs=[ts_file], outputs=[ts_image_preview])

        # 增量输出逻辑：严格使用TXT内 Output: 段；如缺失则按配对文件；仍缺失则空输出
        def on_run_stream(st_llm,
                          selection: List[str],
                          img,
                          g_file,
                          t_file,
                          u_text_file, u_text_manual,
                          temp, tp, mnt, sd):

            seed_val = int(sd) if sd is not None else -1
            if seed_val is not None and seed_val >= 0:
                random.seed(seed_val)
                try:
                    import torch
                    torch.manual_seed(seed_val)
                except Exception:
                    pass

            input_text, output_text_src = parse_io_file(u_text_file)

            if not output_text_src:
                paired = find_paired_output_file(u_text_file)
                if paired and os.path.exists(paired):
                    try:
                        with open(paired, "r", encoding="utf-8") as f:
                            output_text_src = f.read()
                    except Exception:
                        output_text_src = ""

            output_text_src = output_text_src or ""

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

        run_btn.click(
            on_run_stream,
            inputs=[
                state_llm,
                modality_selector,
                image_in,
                graph_file,
                ts_file,
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
