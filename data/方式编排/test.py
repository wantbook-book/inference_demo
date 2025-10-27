import json
import networkx as nx
import matplotlib.pyplot as plt

def draw_topology(data, layout="spring", output_path="topology.png", figsize=(6, 5)):
    """
    根据输入数据绘制并保存拓扑结构图
    
    参数：
    ----------
    data : dict 或 str
        - 若为 dict：必须包含 "nodes" 和 "edges"；
        - 若为 str：表示 JSON 文件路径。
    layout : str
        布局方式，可选：
          - "spring"：弹簧布局（默认）
          - "circular"：环形布局
          - "kamada_kawai"：力导布局
          - "random"：随机布局
    output_path : str
        保存图片的文件路径，如 "topology.png"
    figsize : tuple
        图像尺寸 (宽, 高)
    """
    # === 1. 读取数据 ===
    if isinstance(data, str):
        with open(data, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif not isinstance(data, dict):
        raise TypeError("参数 data 必须是 dict 或 JSON 文件路径字符串。")

    nodes = data.get("nodes", [])
    edges = [tuple(e) for e in data.get("edges", [])]

    if not nodes or not edges:
        raise ValueError("输入数据必须包含 'nodes' 与 'edges'。")

    # === 2. 构建图 ===
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # === 3. 选择布局 ===
    if layout == "spring":
        pos = nx.spring_layout(G, seed=42)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.random_layout(G)

    # === 4. 绘制 ===
    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(G, pos, node_color="#AAB6C4", node_size=800, edgecolors="black")
    nx.draw_networkx_edges(G, pos, width=2, edge_color="#1f77b4")
    nx.draw_networkx_labels(G, pos, font_size=10, font_color="black")
    plt.title("拓扑结构图", fontsize=12)
    plt.axis("off")
    plt.tight_layout()

    # === 5. 保存 ===
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ 拓扑结构图已保存为：{output_path}")

    # 同时显示图
    plt.show()


if __name__ == "__main__":
    data = {
        "nodes": ["A", "B", "C", "D"],
        "edges": [["A", "B"], ["A", "C"], ["B", "D"], ["C", "D"], ["D", "A"]]
    }
    draw_topology(data, layout="spring", output_path="topology.png")
