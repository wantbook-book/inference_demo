import os
import io
import json
import csv
import random
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

try:
    from PIL import Image
except Exception:
    Image = None

# 可选使用transformers，如果不可用则回退到Mock LLM
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False


# --------------- 数据解析与摘要 ---------------

def describe_image(img: Optional[Image.Image]) -> str:
    if img is None:
        return "[无图像输入]"
    try:
        w, h = img.size
        mode = img.mode
        return f"[图像] 尺寸: {w}x{h}, 模式: {mode}"
    except Exception:
        return "[图像] 无法获取尺寸信息"


def parse_graph_file(file_obj: Optional[gr.File]) -> Tuple[List[Tuple[str, str]], List[str]]:
    """
    支持格式：
    - JSON: {"edges": [["a","b"], ...]} 或 {"nodes": [...], "edges": [...]} 或 adjacency list
    - CSV: 两列，src,dst
    - TXT: 每行一个边：nodeA nodeB
    返回: edges, nodes
    """
    edges: List[Tuple[str, str]] = []
    nodes: List[str] = []
    if not file_obj:
        return edges, nodes
    try:
        # gr.File 为临时路径
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
                # adjacency list: [[node, [neighbors...]], ...]
                for item in data:
                    if isinstance(item, (list, tuple)) and len(item) >= 2 and isinstance(item[1], list):
                        src = str(item[0])
                        for dst in item[1]:
                            edges.append((src, str(dst)))
            # 填充节点
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
        else:  # txt or other
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


def parse_graph_text(text: str) -> Tuple[List[Tuple[str, str]], List[str]]:
    edges: List[Tuple[str, str]] = []
    nodes: List[str] = []
    if not text:
        return edges, nodes
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            edges.append((parts[0], parts[1]))
    if edges:
        uniq = set()
        for s, d in edges:
            uniq.add(s); uniq.add(d)
        nodes = sorted(list(uniq))
    return edges, nodes


def summarize_graph(edges: List[Tuple[str, str]], nodes: List[str]) -> str:
    if not edges and not nodes:
        return "[图拓扑] 无输入"
    edge_cnt = len(edges)
    node_cnt = len(nodes)
    sample_edges = ", ".join([f"{s}->{d}" for s, d in edges[:5]]) if edges else "无"
    return f"[图拓扑] 节点数: {node_cnt}, 边数: {edge_cnt}, 示例边: {sample_edges}"


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
        else:  # csv/tsv/txt
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


def parse_timeseries_text(text: str) -> List[float]:
    values: List[float] = []
    if not text:
        return values
    for token in text.replace("\n", ",").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(float(token))
        except Exception:
            pass
    return values


def summarize_timeseries(values: List[float]) -> str:
    if not values:
        return "[时序] 无输入"
    n = len(values)
    min_v = min(values)
    max_v = max(values)
    mean_v = sum(values) / n
    direction = "上升" if values[-1] - values[0] > 0 else "下降" if values[-1] - values[0] < 0 else "持平"
    sample = ", ".join([f"{v:.2f}" for v in values[:8]])
    return f"[时序] 点数: {n}, 范围: [{min_v:.2f}, {max_v:.2f}], 均值: {mean_v:.2f}, 趋势: {direction}, 示例: {sample}"


# --------------- LLM 加载与推理 ---------------

def load_llm(model_path: str, device: str = "cuda") -> Tuple[Dict[str, Any], str]:
    """加载LLM为Mock模式，忽略实际模型加载。"""
    info_lines: List[str] = []
    state: Dict[str, Any] = {
        "type": "mock",
        "model": None,
        "tokenizer": None,
        "device": device,
        "model_path": model_path,
    }
    info_lines.append("LLM推理已设置为Mock模式，忽略实际模型加载。")
    if model_path:
        info_lines.append(f"接收的模型路径/名称: {model_path}")
    return state, "\n".join(info_lines)


def build_prompt(img_desc: str, graph_desc: str, ts_desc: str, user_text: str) -> str:
    sections = [
        "你是一个多模态智能助手。",
        img_desc,
        graph_desc,
        ts_desc,
        f"[文本] {user_text.strip() if user_text else '[无文本输入]'}",
    ]
    return "\n".join(sections) + "\n请根据以上信息生成有用的文字回答。"


def generate_with_llm(state: Dict[str, Any], prompt: str, temperature: float, top_p: float, max_new_tokens: int, seed: Optional[int]) -> str:
    """强制使用Mock推理，返回基于输入摘要的文本。"""
    if seed is not None and seed >= 0:
        random.seed(seed)
        # 可选设置torch随机种子，但此处为Mock不必强制依赖
        try:
            import torch  # 局部导入以避免环境无torch时报错
            torch.manual_seed(seed)
        except Exception:
            pass
    # 直接返回Mock输出，忽略真实模型
    header = "[Mock输出] 根据多模态摘要生成的回复:\n"
    body = prompt[:max(128, min(1000, int(max_new_tokens)))]
    return header + body


# --------------- Gradio UI ---------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="面向配电系统运行优化的多模态大模型构建技术") as demo:
        gr.Markdown("""
        # 面向配电系统运行优化的多模态大模型推理界面
        - 选择或输入LLM模型路径并加载（可选Transformers），或使用Mock。
        - 设置推理参数：`temperature`, `top_p`, `max_new_tokens`, `seed`。
        - 输入数据包含：图像、图拓扑结构、时序数据、文本。
        - 输出：文本结果。
        """)

        state_llm = gr.State(value=None)

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("## LLM配置")
                    model_path = gr.Textbox(label="模型路径", placeholder="/path/to/model")
                    load_btn = gr.Button("加载模型", variant="primary")
                    load_info = gr.Markdown("加载状态将在此显示")

                with gr.Group():
                    gr.Markdown("## 推理参数")
                    temperature = gr.Slider(label="temperature", value=0.7, minimum=0.01, maximum=1.5, step=0.01)
                    top_p = gr.Slider(label="top_p", value=0.9, minimum=0.01, maximum=1.0, step=0.01)
                    max_new_tokens = gr.Slider(label="max_new_tokens", value=128, minimum=4, maximum=2048, step=1)
                    seed = gr.Number(label="seed(>=0设置固定随机种子)", value=-1)

            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("## 输入数据")
                    image_in = gr.Image(type="pil", label="图像输入")

                    gr.Markdown("### 图拓扑结构输入")
                    graph_file = gr.File(label="图文件(JSON/CSV/TXT)")
                    graph_text = gr.Textbox(lines=6, label="图边文本(每行: src dst)")

                    gr.Markdown("### 时序数据输入")
                    ts_file = gr.File(label="时序数据文件(JSON/CSV/TXT)")
                    ts_text = gr.Textbox(lines=4, label="时序数据文本(逗号或换行分隔数字)")

                    user_text = gr.Textbox(lines=6, label="文本输入")

                run_btn = gr.Button("开始推理", variant="primary")
                output_text = gr.Textbox(lines=12, label="文本输出")

        load_btn.click(load_llm, inputs=[model_path], outputs=[state_llm, load_info])

        def on_run(st_llm, img, g_file, g_text, t_file, t_text, u_text, temp, tp, mnt, sd):
            # 解析与摘要
            img_desc = describe_image(img)
            e1, n1 = parse_graph_file(g_file)
            e2, n2 = parse_graph_text(g_text or "")
            edges = e1 + e2
            nodes = sorted(list(set((n1 or []) + (n2 or []))))
            graph_desc = summarize_graph(edges, nodes)

            v1 = parse_timeseries_file(t_file)
            v2 = parse_timeseries_text(t_text or "")
            values = v1 + v2
            ts_desc = summarize_timeseries(values)

            prompt = build_prompt(img_desc, graph_desc, ts_desc, u_text or "")
            seed_val = int(sd) if sd is not None else -1
            seed_val = seed_val if seed_val >= 0 else None
            out = generate_with_llm(st_llm or {"type": "mock"}, prompt, float(temp), float(tp), int(mnt), seed_val)
            return out

        run_btn.click(
            on_run,
            inputs=[state_llm, image_in, graph_file, graph_text, ts_file, ts_text, user_text, temperature, top_p, max_new_tokens, seed],
            outputs=[output_text],
        )

    return demo


if __name__ == "__main__":
    app = build_ui()
    # 监听所有地址，便于外部访问
    app.launch(server_name="0.0.0.0", server_port=7860, show_error=True)