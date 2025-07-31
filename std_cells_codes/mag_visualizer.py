# # import matplotlib.pyplot as plt
# # import matplotlib.patches as patches
# # import random
# # import os

# # def parse_mag_data(mag_content):
# #     """
# #     Parses .mag file content into a detailed, organized data structure,
# #     perfectly matching the logic from the original magReader.py.
# #     """
# #     parsed_data = {
# #         "header": {},
# #         "layers": {}
# #     }
# #     current_layer = None
# #     lines = mag_content.strip().split('\n')

# #     for line in lines:
# #         line = line.strip()
# #         if not line:
# #             continue

# #         # Header parsing
# #         if line.startswith("tech"):
# #             parsed_data["header"]["tech"] = line.split(" ", 1)[1]
# #         elif line.startswith("timestamp"):
# #             try:
# #                 parsed_data["header"]["timestamp"] = int(line.split(" ", 1)[1])
# #             except (ValueError, IndexError):
# #                 pass
# #         # Layer section start
# #         elif line.startswith("<<") and line.endswith(">>"):
# #             current_layer = line.strip("<<>> ").strip()
# #             if current_layer not in parsed_data["layers"]:
# #                 parsed_data["layers"][current_layer] = {"rects": [], "labels": []}
# #         # Rectangle parsing
# #         elif line.startswith("rect") and current_layer:
# #             parts = line.split()
# #             if len(parts) == 5:
# #                 try:
# #                     parsed_data["layers"][current_layer]["rects"].append(
# #                         {"x1": int(parts[1]), "y1": int(parts[2]), "x2": int(parts[3]), "y2": int(parts[4])}
# #                     )
# #                 except (ValueError, IndexError):
# #                     pass
# #         # Detailed label parsing (Correctly implemented as per magReader.py)
# #         elif line.startswith("rlabel") and current_layer:
# #             parts = line.split(" ", 7)
# #             if len(parts) >= 8:
# #                 try:
# #                     parsed_data["layers"][current_layer]["labels"].append({
# #                         "layer": parts[1],
# #                         "x1": int(parts[2]), "y1": int(parts[3]), "x2": int(parts[4]), "y2": int(parts[5]),
# #                         "rotation": int(parts[6]),
# #                         "text": parts[7]
# #                     })
# #                 except (ValueError, IndexError):
# #                     pass
# #         elif line == "<< end >>":
# #             break

# #     return parsed_data

# # def visualize_from_file(file_path: str):
# #     """
# #     Reads a .mag file and creates a detailed visualization that is
# #     identical in functionality to the original magReader.py script.
# #     """
# #     print(f"📊 Visualizing layout from: {file_path}")
# #     if not os.path.exists(file_path):
# #         print(f"🔴 Error: File not found at '{file_path}'")
# #         return

# #     with open(file_path, 'r') as f:
# #         content = f.read()

# #     parsed_data = parse_mag_data(content)
# #     file_name = os.path.basename(file_path)

# #     # --- Visualization logic ---
# #     fig, ax = plt.subplots(figsize=(12, 10))
# #     min_x, max_x, min_y, max_y = float('inf'), float('-inf'), float('inf'), float('-inf')
# #     layer_colors = {}

# #     for layer_name, layer_data in parsed_data["layers"].items():
# #         if not layer_data["rects"] and not layer_data["labels"]:
# #             continue

# #         if layer_name not in layer_colors:
# #             layer_colors[layer_name] = (random.random(), random.random(), random.random())
# #         color = layer_colors[layer_name]

# #         # Plot rectangles
# #         for rect in layer_data["rects"]:
# #             x, y = min(rect["x1"], rect["x2"]), min(rect["y1"], rect["y2"])
# #             width, height = abs(rect["x2"] - rect["x1"]), abs(rect["y2"] - rect["y1"])
# #             min_x, max_x = min(min_x, x), max(max_x, x + width)
# #             min_y, max_y = min(min_y, y), max(max_y, y + height)

# #             rect_patch = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='black',
# #                                            facecolor=color, alpha=0.7,
# #                                            label=layer_name if layer_name not in ax.get_legend_handles_labels()[1] else "")
# #             ax.add_patch(rect_patch)

# #         # Debugging: Print the labels being parsed
# #         # if layer_data["labels"]:
# #         #     print(f"Labels for layer {layer_name}:")
# #         #     for label in layer_data["labels"]:
# #         #         print(f"  {label['text']} at ({label['x1']}, {label['y1']}) to ({label['x2']}, {label['y2']})")

# #         # Plot labels for ports (Correctly implemented)
# #         # In the visualize_from_file function, when plotting labels
# #         # Ensure labels appear on top of other elements (by adjusting z-order)
# #         for label in layer_data["labels"]:
# #             center_x = (label["x1"] + label["x2"]) / 2
# #             center_y = (label["y1"] + label["y2"]) / 2
# #             ax.text(center_x, center_y, label["text"],
# #                     color='blue', fontsize=12, ha='center', va='center',
# #                     rotation=label["rotation"],
# #                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'),
# #                     zorder=10)  # Ensure labels are above other elements


# #     if not all(v != float('inf') and v != float('-inf') for v in [min_x, max_x, min_y, max_y]):
# #         print("🟡 Warning: No rectangles found to determine plot bounds.")
# #         plt.close(fig)
# #         return

# #     # Set plot limits and appearance
# #     padding = 20
# #     ax.set_xlim(min_x - padding, max_x + padding)
# #     ax.set_ylim(min_y - padding, max_y + padding)
# #     ax.set_aspect('equal', adjustable='box')
# #     ax.set_title(f"Layout Visualization: {file_name}", fontsize=14)
    
# #     # Add axis labels
# #     ax.set_xlabel("X Coordinate")
# #     ax.set_ylabel("Y Coordinate")
    
# #     ax.grid(True, linestyle='--', alpha=0.6)
# #     ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

# #     # Display header info on plot
# #     header_text = f"Tech: {parsed_data['header'].get('tech', 'N/A')}\n" \
# #                   f"Timestamp: {parsed_data['header'].get('timestamp', 'N/A')}"
# #     ax.text(1.02, 0.98, header_text, transform=ax.transAxes, fontsize=9,
# #             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.7))

# #     plt.tight_layout(rect=[0, 0, 0.85, 1])
# #     plt.show()

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import os

# Cache to store parsed cell data to avoid re-parsing
_parsed_cell_cache = {}

def parse_mag_data(file_path, current_dir=None):
    """
    Parses the content of a .mag file and returns an organized data structure.
    Supports recursive parsing of instanced cells and labels.
    """
    # Use absolute path for caching to prevent duplicates
    abs_file_path = os.path.abspath(file_path)
    if abs_file_path in _parsed_cell_cache:
        return _parsed_cell_cache[abs_file_path]

    # Determine the current directory if not provided
    if current_dir is None:
        current_dir = os.path.dirname(abs_file_path)
    
    # Construct the full path to the file
    full_file_path = os.path.join(current_dir, os.path.basename(file_path))

    # Directory for resolving sub-cell paths
    current_dir_for_subcells = os.path.dirname(full_file_path)

    try:
        with open(full_file_path, 'r') as file:
            mag_content = file.read()
    except FileNotFoundError:
        print(f"🔴 Warning: Referenced file '{full_file_path}' not found. Skipping instance.")
        return None
    except Exception as e:
        print(f"Error reading file '{full_file_path}': {e}. Skipping instance.")
        return None

    parsed_data = {
        "header": {},
        "layers": {},
        "instances": []
    }
    current_layer = None
    lines = mag_content.strip().split('\n')
    current_instance = None 

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        command = parts[0] if parts else ""

        if command == "tech":
            if len(parts) > 1:
                parsed_data["header"]["tech"] = parts[1]
        elif command == "timestamp":
            try:
                parsed_data["header"]["timestamp"] = int(parts[1])
            except (ValueError, IndexError): pass
        elif line.startswith("<<") and line.endswith(">>"):
            layer_name = line.strip("<<>> ").strip()
            if layer_name != "end":
                current_layer = layer_name
                if current_layer not in parsed_data["layers"]:
                    parsed_data["layers"][current_layer] = {"rects": [], "labels": []}
        
        elif command == "rect":
            try:
                if len(parts) == 5 and current_layer:
                    parsed_data["layers"][current_layer]["rects"].append(
                        {"x1": int(parts[1]), "y1": int(parts[2]), "x2": int(parts[3]), "y2": int(parts[4])}
                    )
            except (ValueError, IndexError):
                 print(f"🟡 Warning: Skipping malformed rect line: '{line}'")
        
        elif command == "rlabel":
            try:
                if len(parts) >= 8:
                    label_layer = parts[1]
                    if label_layer not in parsed_data["layers"]:
                         parsed_data["layers"][label_layer] = {"rects": [], "labels": []}
                    parsed_data["layers"][label_layer]["labels"].append({
                        "text": " ".join(parts[7:]), 
                        "x": int(parts[2]), "y": int(parts[3]),
                        "rotation": int(parts[6])
                    })
            except (ValueError, IndexError):
                 print(f"🟡 Warning: Skipping malformed rlabel line: '{line}'")

        elif command == "use":
            if current_instance:
                parsed_data["instances"].append(current_instance)
            
            if len(parts) >= 3:
                cell_type = parts[1]
                instance_name = parts[2]
                sub_file_path = os.path.join(current_dir_for_subcells, f"{cell_type}.mag")
                
                current_instance = {
                    "cell_type": cell_type,
                    "instance_name": instance_name,
                    "parsed_content": parse_mag_data(sub_file_path, current_dir_for_subcells),
                    "transform": [1, 0, 0, 0, 1, 0], "box": [0, 0, 0, 0]
                }
                if not current_instance["parsed_content"]:
                    current_instance = None
            
        elif command == "transform" and current_instance:
            try:
                current_instance["transform"] = [int(v) for v in parts[1:]]
            except (ValueError, IndexError): pass
        elif command == "box" and current_instance:
            try:
                current_instance["box"] = [int(v) for v in parts[1:]]
            except (ValueError, IndexError): pass
        
        elif line == "<< end >>":
            if current_instance:
                parsed_data["instances"].append(current_instance)
                current_instance = None

    if current_instance:
        parsed_data["instances"].append(current_instance)

    _parsed_cell_cache[abs_file_path] = parsed_data
    return parsed_data

def visualize_layout(file_path: str):
    """
    Reads a .mag file and creates a detailed, hierarchical visualization.
    This function handles nested cell instances, transformations, and labels.
    """
    print(f"📊 Visualizing layout from: {file_path}")
    if not os.path.exists(file_path):
        print(f"🔴 Error: File not found at '{file_path}'")
        return

    # 1. Parse the top-level .mag file
    parsed_data = parse_mag_data(os.path.abspath(file_path))
    if not parsed_data:
        print(f"🔴 Error: Failed to parse the file '{file_path}'.")
        return

    # 2. Setup visualization
    fig, ax = plt.subplots(figsize=(14, 11))
    min_x, max_x, min_y, max_y = float('inf'), float('-inf'), float('inf'), float('-inf')
    layer_colors = {}
    
    def get_random_color():
        return (random.random(), random.random(), random.random())

    def _apply_transform(x, y, T):
        # Applies a 2x3 affine transformation matrix T = [a, b, c, d, e, f]
        # NewX = a*x + b*y + c
        # NewY = d*x + e*y + f
        return (T[0] * x + T[1] * y + T[2], T[3] * x + T[4] * y + T[5])

    def _draw_elements(data_to_draw, current_transform=[1, 0, 0, 0, 1, 0]):
        nonlocal min_x, max_x, min_y, max_y

        # Draw rectangles for each layer
        for layer_name, layer_data in data_to_draw["layers"].items():
            if layer_name not in layer_colors:
                layer_colors[layer_name] = get_random_color()
            color = layer_colors[layer_name]

            for rect in layer_data.get("rects", []):
                # Transform corners of the rectangle
                x1, y1, x2, y2 = rect["x1"], rect["y1"], rect["x2"], rect["y2"]
                tx1, ty1 = _apply_transform(x1, y1, current_transform)
                tx2, ty2 = _apply_transform(x2, y2, current_transform)

                width, height = abs(tx2 - tx1), abs(ty2 - ty1)
                x_start, y_start = min(tx1, tx2), min(ty1, ty2)
                
                min_x, max_x = min(min_x, x_start), max(max_x, x_start + width)
                min_y, max_y = min(min_y, y_start), max(max_y, y_start + height)

                ax.add_patch(patches.Rectangle((x_start, y_start), width, height,
                                               linewidth=1, edgecolor='black', facecolor=color, alpha=0.7))

            # Draw labels (ports)
            for label in layer_data.get("labels", []):
                tx, ty = _apply_transform(label["x"], label["y"], current_transform)
                ax.text(tx, ty, label["text"], color='blue', fontsize=10, ha='center', va='center',
                        rotation=label.get("rotation", 0),
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='blue', boxstyle='round,pad=0.2'), zorder=10)

        # Recursively draw all child instances
        for instance in data_to_draw.get("instances", []):
            if instance.get("parsed_content"):
                # Combine parent transform with child's local transform
                # This part is crucial for correct hierarchical placement (though not fully implemented in MAG format)
                _draw_elements(instance["parsed_content"], instance["transform"])
                
                # Draw instance name at the center of its bounding box
                box = instance.get("box", [0, 0, 0, 0])
                center_x, center_y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                inst_center_x, inst_center_y = _apply_transform(center_x, center_y, instance["transform"])
                
                ax.text(inst_center_x, inst_center_y, instance["instance_name"],
                        color='red', fontsize=11, ha='center', va='center', fontweight='bold',
                        bbox=dict(facecolor='yellow', alpha=0.6, edgecolor='red', boxstyle='round,pad=0.3'), zorder=11)

    # 3. Start drawing from the top-level cell
    _draw_elements(parsed_data)

    # 4. Finalize plot
    if not all(v != float('inf') and v != float('-inf') for v in [min_x, max_x, min_y, max_y]):
        print("🟡 Warning: No geometric elements found to determine plot bounds.")
        plt.close(fig)
        return

    padding = (max_x - min_x) * 0.1
    ax.set_xlim(min_x - padding, max_x + padding)
    ax.set_ylim(min_y - padding, max_y + padding)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"Layout Visualization: {os.path.basename(file_path)}", fontsize=16)
    ax.set_xlabel("X Coordinate", fontsize=12)
    ax.set_ylabel("Y Coordinate", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    legend_patches = [patches.Patch(color=color, label=name, alpha=0.7) for name, color in layer_colors.items()]
    ax.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1.02, 0.5), title="Layers")

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()
