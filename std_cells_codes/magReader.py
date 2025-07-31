import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

## magReader.py
## run in terminal
## 

# python d:/std_cells_datasets/std_cells_datasets/magReader.py

def parse_mag_data(mag_content):
    """
    Mengurai konten data .mag dan mengembalikan struktur data yang terorganisir.
    """
    parsed_data = {
        "header": {},
        "layers": {}
    }
    current_layer = None
    lines = mag_content.strip().split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("magic"):
            parts = line.split(" ", 2)
            if len(parts) > 1:
                parsed_data["header"]["file_type"] = parts[0]
                parsed_data["header"]["magic_version"] = parts[1]
                if len(parts) > 2:
                    parsed_data["header"]["extra"] = parts[2]
        elif line.startswith("tech"):
            parsed_data["header"]["tech"] = line.split(" ", 1)[1]
        elif line.startswith("timestamp"):
            try:
                parsed_data["header"]["timestamp"] = int(line.split(" ", 1)[1])
            except ValueError:
                pass
        elif line.startswith("<<") and line.endswith(">>"):
            current_layer = line.strip("<<>> ").strip()
            if current_layer not in parsed_data["layers"]:
                parsed_data["layers"][current_layer] = {
                    "rects": [],
                    "labels": []
                }
        elif line.startswith("rect"):
            if current_layer:
                parts = line.split()
                if len(parts) == 5:
                    try:
                        x1, y1, x2, y2 = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
                        parsed_data["layers"][current_layer]["rects"].append({
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2
                        })
                    except ValueError:
                        pass
        elif line.startswith("rlabel"):
            if current_layer:
                parts = line.split(" ", 7)
                if len(parts) >= 8:
                    try:
                        label_layer = parts[1]
                        x1, y1, x2, y2 = int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
                        rotation = int(parts[6])
                        label_text = parts[7]
                        parsed_data["layers"][current_layer]["labels"].append({
                            "layer": label_layer,
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                            "rotation": rotation,
                            "text": label_text
                        })
                    except (ValueError, IndexError):
                        pass
        elif line == "<< end >>":
            break

    return parsed_data

def visualize_layout(parsed_data, file_name, title_prefix="Layout Visualization"):
    """
    Memvisualisasikan data layout yang diurai menggunakan matplotlib.
    Juga menampilkan informasi header di plot.
    """
    fig, ax = plt.subplots(1, figsize=(12, 10)) # Ukuran figur diperbesar sedikit

    min_x, max_x, min_y, max_y = float('inf'), float('-inf'), float('inf'), float('-inf')

    layer_colors = {}
    
    # Generate random colors for layers
    def get_random_color():
        r = random.random()
        g = random.random()
        b = random.random()
        return (r, g, b)

    for layer_name, layer_data in parsed_data["layers"].items():
        if layer_name not in layer_colors:
            layer_colors[layer_name] = get_random_color()
        
        color = layer_colors[layer_name]

        for rect in layer_data["rects"]:
            # Koordinat bawah-kiri dan dimensi persegi panjang
            x = min(rect["x1"], rect["x2"])
            y = min(rect["y1"], rect["y2"])
            width = abs(rect["x2"] - rect["x1"])
            height = abs(rect["y2"] - rect["y1"])

            # Perbarui batas plot
            min_x = min(min_x, x)
            max_x = max(max_x, x + width)
            min_y = min(min_y, y)
            max_y = max(max_y, y + height)

            # Buat patch persegi panjang dan tambahkan ke axes
            # Facecolor untuk mengisi, edgecolor untuk garis tepi, alpha untuk transparansi
            rect_patch = patches.Rectangle((x, y), width, height,
                                           linewidth=1, edgecolor='black', facecolor=color, alpha=0.7,
                                           label=layer_name if layer_name not in ax.get_legend_handles_labels()[1] else "")
            ax.add_patch(rect_patch)

        for label in layer_data["labels"]:
            # Posisi tengah label untuk penempatan teks
            center_x = (label["x1"] + label["x2"]) / 2
            center_y = (label["y1"] + label["y2"]) / 2
            ax.text(center_x, center_y, label["text"],
                    color='blue', fontsize=8, ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))

    # Mengatur batas plot sedikit lebih luas dari desain
    padding = 20
    ax.set_xlim(min_x - padding, max_x + padding)
    ax.set_ylim(min_y - padding, max_y + padding)

    ax.set_aspect('equal', adjustable='box') # Memastikan skala x dan y sama
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    
    # --- Set title to include the exact file name entered by the user ---
    ax.set_title(f"{title_prefix}: {file_name}") # Judul langsung dengan nama file yang diinput

    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5)) # Legenda di luar plot

    # --- Tambahkan informasi header ke plot ---
    header_text = f"Tech: {parsed_data['header'].get('tech', 'N/A')}\n" \
                  f"Timestamp: {parsed_data['header'].get('timestamp', 'N/A')}"
    
    # --- Tempatkan teks di sudut kanan atas plot ---
    ax.text(1.02, 0.98, header_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.7))

    plt.tight_layout(rect=[0, 0, 0.9, 1]) # Sesuaikan layout untuk memberi ruang pada legenda dan teks
    plt.show() # Menampilkan plot

# --- Fungsi untuk mencetak detail ke konsol (seperti .txt) ---
def print_parsed_details(parsed_data):
    print("Header Information:")
    for key, value in parsed_data["header"].items():
        print(f"  {key}: {value}")

    print("\nLayer Information:")
    for layer_name, layer_data in parsed_data["layers"].items():
        # Jangan mencetak layer 'end' karena itu hanya penanda file
        if layer_name == "end":
            continue

        print(f"  Layer: {layer_name}")
        if layer_data["rects"]:
            print("    Rectangles:")
            for rect in layer_data["rects"]:
                width = abs(rect["x2"] - rect["x1"])
                height = abs(rect["y2"] - rect["y1"])
                print(f"      x1: {rect['x1']}, y1: {rect['y1']}, x2: {rect['x2']}, y2: {rect['y2']}, Width: {width}, Height: {height}")
        if layer_data["labels"]:
            print("    Labels:")
            for label in layer_data["labels"]:
                print(f"      Text: '{label['text']}' at ({label['x1']}, {label['y1']}) to ({label['x2']}, {label['y2']}), Layer: {label['layer']}, Rotation: {label['rotation']}")


# --- Input file data ---
file_path = input("Enter your file name (e.g., xor.mag): ")

try:
    with open(file_path, 'r') as file:
        mag_data_from_file = file.read()

    parsed_mag = parse_mag_data(mag_data_from_file)

    # Cetak detail ke konsol terlebih dahulu
    print_parsed_details(parsed_mag)

    # Kemudian tampilkan visualisasi, meneruskan nama file
    visualize_layout(parsed_mag, file_name=file_path, title_prefix="Layout Design")

except FileNotFoundError:
    print(f"Error: File '{file_path}' not found. Please make sure the file path is correct and in the same directory, or provide the full path.")
except Exception as e:
    print(f"An error occurred: {e}")
