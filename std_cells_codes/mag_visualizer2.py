import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import os

_parsed_cell_cache = {}

def parse_mag_data(file_path, current_dir=None):
    abs_file_path = os.path.abspath(file_path)
    if abs_file_path in _parsed_cell_cache:
        return _parsed_cell_cache[abs_file_path]

    if current_dir is None:
        current_dir = os.path.dirname(abs_file_path)
    
    full_file_path = os.path.join(current_dir, os.path.basename(file_path))
    current_dir_for_subcells = os.path.dirname(full_file_path)

    try:
        with open(full_file_path, 'r') as file:
            mag_content = file.read()
    except FileNotFoundError:
        return None

    parsed_data = {"header": {}, "layers": {}, "instances": []}
    current_layer = None
    lines = mag_content.strip().split('\n')
    current_instance = None 

    for line in lines:
        line = line.strip()
        if not line: continue
        parts = line.split()
        command = parts[0] if parts else ""

        if line.startswith("<<") and line.endswith(">>"):
            layer_name = line.strip("<<>> ").strip()
            if layer_name != "end":
                current_layer = layer_name
                if current_layer not in parsed_data["layers"]:
                    parsed_data["layers"][current_layer] = {"rects": [], "labels": []}
        elif command == "rect" and current_layer and len(parts) == 5:
            parsed_data["layers"][current_layer]["rects"].append({"x1": int(parts[1]), "y1": int(parts[2]), "x2": int(parts[3]), "y2": int(parts[4])})
        elif command == "use" and len(parts) >= 3:
            if current_instance: parsed_data["instances"].append(current_instance)
            current_instance = {
                "cell_type": parts[1],
                "instance_name": parts[2],
                "parsed_content": parse_mag_data(os.path.join(current_dir_for_subcells, f"{parts[1]}.mag"), current_dir_for_subcells),
                "transform": [1, 0, 0, 0, 1, 0], "box": [0, 0, 0, 0]
            }
        elif command == "transform" and current_instance:
            current_instance["transform"] = [int(v) for v in parts[1:]]
        elif command == "box" and current_instance:
            current_instance["box"] = [int(v) for v in parts[1:]]
        elif line == "<< end >>":
            if current_instance: parsed_data["instances"].append(current_instance)
            break

    if current_instance and current_instance not in parsed_data["instances"]:
        parsed_data["instances"].append(current_instance)

    _parsed_cell_cache[abs_file_path] = parsed_data
    return parsed_data

def visualize_layout(file_path: str, output_image_path: str = None):
    """
    Reads a .mag file, creates a visualization, and saves it to a file.
    """
    if not os.path.exists(file_path):
        print(f"Visualization failed: File not found at '{file_path}'")
        return

    parsed_data = parse_mag_data(os.path.abspath(file_path))
    if not parsed_data:
        print(f"Visualization failed: Could not parse '{file_path}'")
        return

    fig, ax = plt.subplots(figsize=(12, 10))
    min_x, max_x, min_y, max_y = float('inf'), float('-inf'), float('inf'), float('-inf')
    layer_colors = {}
    
    def get_random_color():
        return (random.random(), random.random(), random.random())

    def _apply_transform(x, y, T):
        return (T[0] * x + T[1] * y + T[2], T[3] * x + T[4] * y + T[5])

    def _draw_elements(data_to_draw, transform=[1, 0, 0, 0, 1, 0]):
        nonlocal min_x, max_x, min_y, max_y
        if not data_to_draw: return

        for layer_name, layer_data in data_to_draw.get("layers", {}).items():
            if layer_name not in layer_colors: layer_colors[layer_name] = get_random_color()
            color = layer_colors[layer_name]
            for rect in layer_data.get("rects", []):
                tx1, ty1 = _apply_transform(rect["x1"], rect["y1"], transform)
                tx2, ty2 = _apply_transform(rect["x2"], rect["y2"], transform)
                width, height = abs(tx2-tx1), abs(ty2-ty1)
                x, y = min(tx1, tx2), min(ty1, ty2)
                min_x, max_x, min_y, max_y = min(min_x, x), max(max_x, x+width), min(min_y, y), max(max_y, y+height)
                ax.add_patch(patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='black', facecolor=color, alpha=0.7))

        for instance in data_to_draw.get("instances", []):
            _draw_elements(instance.get("parsed_content"), instance["transform"])

    _draw_elements(parsed_data)

    if min_x != float('inf'):
        padding = (max_x - min_x) * 0.1
        ax.set_xlim(min_x - padding, max_x + padding)
        ax.set_ylim(min_y - padding, max_y + padding)

    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"Layout: {os.path.basename(file_path)}")
    ax.grid(True, linestyle='--', alpha=0.6)
    
    if output_image_path:
        plt.savefig(output_image_path, bbox_inches='tight')
        print(f"Visualization saved to {output_image_path}")
    else:
        plt.show()
    plt.close(fig)