import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import os

## magReader.py
## Corrected version to fix parsing issues and display port labels.
## Now correctly parses all instances, handles multiple 'rect' formats, and shows 'rlabel' text.

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
        print(f"Warning: Referenced file '{full_file_path}' not found. Skipping instance.")
        return None
    except Exception as e:
        print(f"Error reading file '{full_file_path}': {e}. Skipping instance.")
        return None

    parsed_data = {
        "header": {},
        "layers": {},
        "instances": []  # List to store cell instances
    }
    current_layer = None
    lines = mag_content.strip().split('\n')

    # Temporary variable to hold the instance being parsed
    current_instance = None 

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        command = parts[0] if parts else ""

        if command == "magic":
            if len(parts) > 1:
                parsed_data["header"]["magic_version"] = parts[1]
        elif command == "tech":
            if len(parts) > 1:
                parsed_data["header"]["tech"] = parts[1]
        elif command == "timestamp":
            try:
                ts = int(parts[1])
                if current_instance:
                    current_instance["timestamp"] = ts
                else:
                    parsed_data["header"]["timestamp"] = ts
            except (ValueError, IndexError):
                pass
        elif line.startswith("<<") and line.endswith(">>"):
            layer_name = line.strip("<<>> ").strip()
            # Avoid treating "<< end >>" as a layer
            if layer_name != "end":
                current_layer = layer_name
                if current_layer not in parsed_data["layers"]:
                    parsed_data["layers"][current_layer] = {"rects": [], "labels": []}
        
        elif command == "rect":
            try:
                layer_for_rect = None
                rect_coords = []
                if len(parts) == 5 and current_layer: # Format: rect llx lly urx ury
                    layer_for_rect = current_layer
                    rect_coords = list(map(int, parts[1:]))
                elif len(parts) == 6: # Format: rect layer llx lly urx ury
                    layer_for_rect = parts[1]
                    rect_coords = list(map(int, parts[2:]))

                if layer_for_rect and len(rect_coords) == 4:
                    if layer_for_rect not in parsed_data["layers"]:
                        parsed_data["layers"][layer_for_rect] = {"rects": [], "labels": []}
                    parsed_data["layers"][layer_for_rect]["rects"].append(
                        {"x1": rect_coords[0], "y1": rect_coords[1], "x2": rect_coords[2], "y2": rect_coords[3]}
                    )
            except (ValueError, IndexError):
                print(f"Warning: Skipping malformed rect line: '{line}'")
        
        # --- NEW: Parse 'rlabel' for port names ---
        elif command == "rlabel":
            try:
                # Format: rlabel <layer> <x> <y> <x> <y> <rotation> <text>
                if len(parts) >= 8:
                    label_layer = parts[1]
                    x, y = int(parts[2]), int(parts[3])
                    text = " ".join(parts[7:]) # Join the rest as text
                    if label_layer not in parsed_data["layers"]:
                         parsed_data["layers"][label_layer] = {"rects": [], "labels": []}
                    parsed_data["layers"][label_layer]["labels"].append({
                        "text": text, "x": x, "y": y
                    })
            except (ValueError, IndexError):
                 print(f"Warning: Skipping malformed rlabel line: '{line}'")

        elif command == "use":
            if current_instance:
                parsed_data["instances"].append(current_instance)
            
            if len(parts) >= 3:
                cell_type = parts[1]
                instance_name = parts[2]
                sub_file_path = os.path.join(current_dir_for_subcells, f"{cell_type}.mag")
                
                parsed_sub_cell_content = parse_mag_data(sub_file_path, current_dir_for_subcells)
                
                current_instance = {
                    "cell_type": cell_type,
                    "instance_name": instance_name,
                    "file_path": sub_file_path,
                    "parsed_content": parsed_sub_cell_content,
                    "transform": [1, 0, 0, 0, 1, 0],
                    "box": [0, 0, 0, 0],
                    "timestamp": None
                }
                if not parsed_sub_cell_content:
                    current_instance = None
            
        elif command == "transform" and current_instance:
            try:
                transform_values = [float(x) for x in parts[1:]]
                if len(transform_values) == 6:
                    current_instance["transform"] = transform_values
            except (ValueError, IndexError): pass
        elif command == "box" and current_instance:
            try:
                box_values = [int(x) for x in parts[1:]]
                if len(box_values) == 4:
                    current_instance["box"] = box_values
            except (ValueError, IndexError): pass
        
        elif line == "<< end >>":
            if current_instance:
                parsed_data["instances"].append(current_instance)
                current_instance = None

    if current_instance:
        parsed_data["instances"].append(current_instance)

    _parsed_cell_cache[abs_file_path] = parsed_data
    return parsed_data

def visualize_layout(parsed_data, file_name, title_prefix="Layout Visualization", layer_colors=None):
    """
    Visualizes the parsed layout data using matplotlib, including header info, instances, and labels.
    """
    fig, ax = plt.subplots(1, figsize=(12, 10))
    min_x, max_x, min_y, max_y = float('inf'), float('-inf'), float('inf'), float('-inf')

    if layer_colors is None:
        layer_colors = {}
    
    def get_random_color():
        return (random.random(), random.random(), random.random())

    def _apply_transform(x, y, T):
        return (T[0] * x + T[1] * y + T[2], T[3] * x + T[4] * y + T[5])

    def _draw_elements(data_to_draw, current_transform=[1, 0, 0, 0, 1, 0]):
        nonlocal min_x, max_x, min_y, max_y

        for layer_name, layer_data in data_to_draw["layers"].items():
            if layer_name not in layer_colors:
                layer_colors[layer_name] = get_random_color()
            color = layer_colors[layer_name]

            for rect in layer_data.get("rects", []):
                corners = [(rect["x1"], rect["y1"]), (rect["x2"], rect["y1"]), (rect["x1"], rect["y2"]), (rect["x2"], rect["y2"])]
                transformed_corners = [_apply_transform(cx, cy, current_transform) for cx, cy in corners]
                
                tx_coords = [c[0] for c in transformed_corners]
                ty_coords = [c[1] for c in transformed_corners]
                
                tx1, ty1, tx2, ty2 = min(tx_coords), min(ty_coords), max(tx_coords), max(ty_coords)
                width, height = tx2 - tx1, ty2 - ty1

                min_x, max_x = min(min_x, tx1), max(max_x, tx2)
                min_y, max_y = min(min_y, ty1), max(max_y, ty2)

                ax.add_patch(patches.Rectangle((tx1, ty1), width, height,
                                               linewidth=1, edgecolor='black', facecolor=color, alpha=0.7))

            # --- NEW: Draw the labels (port names) ---
            for label in layer_data.get("labels", []):
                tx, ty = _apply_transform(label["x"], label["y"], current_transform)
                ax.text(tx, ty, label["text"],
                        color='darkgreen', fontsize=9, ha='center', va='center',
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='green', boxstyle='round,pad=0.2'))


    # Draw the main cell's geometry and labels
    _draw_elements(parsed_data)

    # Draw all encapsulated instances
    for instance in parsed_data.get("instances", []):
        if instance.get("parsed_content"):
            _draw_elements(instance["parsed_content"], instance["transform"])
            
            box = instance.get("box", [0, 0, 0, 0])
            center_x, center_y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            inst_center_x, inst_center_y = _apply_transform(center_x, center_y, instance["transform"])
            
            ax.text(inst_center_x, inst_center_y, instance["instance_name"],
                    color='red', fontsize=10, ha='center', va='center', fontweight='bold',
                    bbox=dict(facecolor='yellow', alpha=0.6, edgecolor='black', boxstyle='round,pad=0.3'))

    if max_x > min_x and max_y > min_y:
        padding_x = (max_x - min_x) * 0.1
        padding_y = (max_y - min_y) * 0.1
        ax.set_xlim(min_x - padding_x, max_x + padding_x)
        ax.set_ylim(min_y - padding_y, max_y + padding_y)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_title(f"{title_prefix}: {os.path.basename(file_name)}")
    ax.grid(True, linestyle='--', alpha=0.6)
    
    legend_patches = [patches.Patch(color=color, label=name) for name, color in layer_colors.items()]
    ax.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1.02, 0.5))

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()

# --- Main execution block ---
if __name__ == "__main__":
    file_path_input = input("Enter your top-level .mag file name (e.g., and_common_3.mag): ")

    if not os.path.exists(file_path_input):
        print(f"Error: File '{file_path_input}' not found. Please provide the full path or ensure it's in the correct directory.")
    else:
        try:
            parsed_mag = parse_mag_data(os.path.abspath(file_path_input))

            if parsed_mag:
                print("Parsing complete. Visualizing layout...")
                visualize_layout(parsed_mag, file_name=file_path_input, title_prefix="Layout Design")
            else:
                print(f"Failed to parse the main file: '{file_path_input}'.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
