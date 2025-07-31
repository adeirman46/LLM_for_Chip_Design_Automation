import streamlit as st
import os
import shutil
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import re
import asyncio
import nest_asyncio
import glob

# Fix for the asyncio event loop error in Streamlit's thread
nest_asyncio.apply()

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# --- Configuration ---
FAISS_INDEX_PATH = "faiss_mag_index_st"
GENERATED_MAG_DIR = "generated_mag_st"
MAG_FILES_PATH = '../std_cells_datasets/' # Adjust if your dataset is elsewhere

# --- Caching ---
_parsed_cell_cache = {}

# ==============================================================================
# VISUALIZER (To work with Streamlit)
# ==============================================================================

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
        st.warning(f"Referenced file '{os.path.basename(full_file_path)}' not found. Instance will be skipped.")
        return None
    except Exception as e:
        st.error(f"Error reading file '{full_file_path}': {e}")
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

        if command == "tech" and len(parts) > 1:
            parsed_data["header"]["tech"] = parts[1]
        elif command == "timestamp":
            pass # Simplified for app
        elif line.startswith("<<") and line.endswith(">>"):
            layer_name = line.strip("<<>> ").strip()
            if layer_name != "end":
                current_layer = layer_name
                if current_layer not in parsed_data["layers"]:
                    parsed_data["layers"][current_layer] = {"rects": [], "labels": []}
        elif command == "rect" and len(parts) == 5 and current_layer:
            try:
                parsed_data["layers"][current_layer]["rects"].append(
                    {"x1": int(parts[1]), "y1": int(parts[2]), "x2": int(parts[3]), "y2": int(parts[4])}
                )
            except ValueError:
                pass # Ignore malformed rect lines
        elif command == "rlabel" and len(parts) >= 8:
            label_layer = parts[1]
            if label_layer not in parsed_data["layers"]:
                 parsed_data["layers"][label_layer] = {"rects": [], "labels": []}
            parsed_data["layers"][label_layer]["labels"].append({
                "text": " ".join(parts[7:]), "x": int(parts[2]), "y": int(parts[3]), "rotation": int(parts[6])
            })
        elif command == "use":
            if current_instance: parsed_data["instances"].append(current_instance)
            if len(parts) >= 3:
                sub_file_path = os.path.join(current_dir_for_subcells, f"{parts[1]}.mag")
                current_instance = {
                    "cell_type": parts[1], "instance_name": parts[2],
                    "parsed_content": parse_mag_data(sub_file_path, current_dir_for_subcells),
                    "transform": [1, 0, 0, 0, 1, 0], "box": [0, 0, 0, 0]
                }
                if not current_instance["parsed_content"]: current_instance = None
        elif command == "transform" and current_instance:
            current_instance["transform"] = [int(v) for v in parts[1:]]
        elif command == "box" and current_instance:
            current_instance["box"] = [int(v) for v in parts[1:]]
        elif line == "<< end >>" and current_instance:
            parsed_data["instances"].append(current_instance)
            current_instance = None

    if current_instance: parsed_data["instances"].append(current_instance)
    _parsed_cell_cache[abs_file_path] = parsed_data
    return parsed_data

def visualize_layout_for_streamlit(file_path: str):
    if not os.path.exists(file_path):
        st.error(f"Cannot visualize. File not found at '{file_path}'")
        return None

    _parsed_cell_cache.clear()
    parsed_data = parse_mag_data(os.path.abspath(file_path))
    if not parsed_data:
        return None

    fig, ax = plt.subplots(figsize=(12, 9))
    min_x, max_x, min_y, max_y = float('inf'), float('-inf'), float('inf'), float('-inf')
    layer_colors = {}

    def get_random_color(): return (random.random(), random.random(), random.random())
    def _apply_transform(x, y, T): return (T[0] * x + T[1] * y + T[2], T[3] * x + T[4] * y + T[5])

    def _draw_elements(data_to_draw, current_transform=[1, 0, 0, 0, 1, 0]):
        nonlocal min_x, max_x, min_y, max_y
        for layer_name, layer_data in data_to_draw["layers"].items():
            if layer_name not in layer_colors: layer_colors[layer_name] = get_random_color()
            color = layer_colors[layer_name]
            for rect in layer_data.get("rects", []):
                x1, y1, x2, y2 = rect["x1"], rect["y1"], rect["x2"], rect["y2"]
                tx1, ty1 = _apply_transform(x1, y1, current_transform)
                tx2, ty2 = _apply_transform(x2, y2, current_transform)
                width, height = abs(tx2 - tx1), abs(ty2 - ty1)
                x_start, y_start = min(tx1, tx2), min(ty1, ty2)
                min_x, max_x = min(min_x, x_start), max(max_x, x_start + width)
                min_y, max_y = min(min_y, y_start), max(max_y, y_start + height)
                ax.add_patch(patches.Rectangle((x_start, y_start), width, height, linewidth=1, edgecolor='black', facecolor=color, alpha=0.7))
            for label in layer_data.get("labels", []):
                tx, ty = _apply_transform(label["x"], label["y"], current_transform)
                ax.text(tx, ty, label["text"], color='blue', fontsize=9, ha='center', va='center', rotation=label.get("rotation", 0),
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='blue', boxstyle='round,pad=0.2'), zorder=10)
        for instance in data_to_draw.get("instances", []):
            if instance.get("parsed_content"):
                _draw_elements(instance["parsed_content"], instance["transform"])
                box = instance.get("box", [0, 0, 0, 0])
                center_x, center_y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                inst_center_x, inst_center_y = _apply_transform(center_x, center_y, instance["transform"])
                ax.text(inst_center_x, inst_center_y, instance["instance_name"], color='darkred', fontsize=10, ha='center', va='center', fontweight='bold',
                        bbox=dict(facecolor='yellow', alpha=0.5, edgecolor='red', boxstyle='round,pad=0.3'), zorder=11)

    _draw_elements(parsed_data)

    if not all(v != float('inf') and v != float('-inf') for v in [min_x, max_x, min_y, max_y]):
        plt.close(fig)
        return None

    padding = (max_x - min_x) * 0.1 if (max_x > min_x) else 10
    ax.set_xlim(min_x - padding, max_x + padding)
    ax.set_ylim(min_y - padding, max_y + padding)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"Layout: {os.path.basename(file_path)}", fontsize=16)
    ax.set_xlabel("X Coordinate", fontsize=12)
    ax.set_ylabel("Y Coordinate", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    legend_patches = [patches.Patch(color=color, label=name, alpha=0.7) for name, color in layer_colors.items()]
    ax.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1.02, 0.5), title="Layers")
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    return fig

# ==============================================================================
# LAYOUT GENERATOR
# ==============================================================================

class MagicLayoutGenerator:
    def __init__(self, mag_files_directory: str):
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key: raise ValueError("GOOGLE_API_KEY not found in .env file.")

        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=api_key, temperature=0.1)
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        self._load_or_create_vector_store(mag_files_directory)

        synthesis_prompt_template = """
You are a Magic VLSI layout expert. Your task is to generate a .mag file based on the user's QUESTION.
**CRITICAL INSTRUCTIONS:**
1.  **Analyze CONTEXTS**: You will be given up to 3 contexts. Identify the SINGLE most relevant context.
2.  **Prioritize the Best Context**: For simple gates, replicate the best context's geometry and rename labels. For complex cells, use component contexts to build the layout with 'use' commands and routing.
3.  **DO NOT SIMPLIFY**: You are forbidden from simplifying the provided contexts. Replicate them.
4.  **OUTPUT FORMAT**: Your response MUST BE ONLY the raw .mag file content, starting with 'magic' and ending with '<< end >>'.

CONTEXTS:
{context}

QUESTION:
{question}

ANSWER (A complete .mag file, based on the most relevant context):
"""
        synthesis_prompt = PromptTemplate.from_template(synthesis_prompt_template)
        self.synthesis_chain = synthesis_prompt | self.llm | StrOutputParser()

    def _preprocess_mag_file(self, file_path: str):
        with open(file_path, 'r') as f: content = f.read()
        cell_name = os.path.splitext(os.path.basename(file_path))[0]
        keyword_part = f"cell name is {cell_name}. function is {cell_name}. " * 5
        description = f"Magic layout for standard cell. {keyword_part}"
        match = re.search(r"#\s*this is\s+(.*)", content, re.IGNORECASE)
        if match: description += f"It is described as a {match.group(1).strip()}. "
        return Document(page_content=description + content, metadata={"source": file_path})

    @st.cache_resource
    def _load_or_create_vector_store(_self, path: str):
        if os.path.exists(FAISS_INDEX_PATH):
            _self.vector_store = FAISS.load_local(FAISS_INDEX_PATH, _self.embeddings, allow_dangerous_deserialization=True)
        else:
            mag_files = glob.glob(os.path.join(path, '**', '*.mag'), recursive=True)
            if not mag_files: raise FileNotFoundError(f"No .mag files found in '{path}'.")
            with st.spinner(f"Creating new vector store from {len(mag_files)} files..."):
                documents = [_self._preprocess_mag_file(file_path) for file_path in mag_files]
                _self.vector_store = FAISS.from_documents(documents, _self.embeddings)
                _self.vector_store.save_local(FAISS_INDEX_PATH)
        _self.retriever = _self.vector_store.as_retriever(search_kwargs={"k": 3})

    def _parse_dependencies(self, mag_content: str) -> set:
        return set(re.findall(r"^\s*use\s+([\w\d_]+)", mag_content, re.MULTILINE))

    def generate_and_save_yield(self, initial_query: str, top_level_filename: str):
        if os.path.exists(GENERATED_MAG_DIR):
            shutil.rmtree(GENERATED_MAG_DIR)
        os.makedirs(GENERATED_MAG_DIR, exist_ok=True)

        top_level_cell_name = os.path.splitext(top_level_filename)[0]
        generation_queue = [(initial_query, top_level_cell_name)]
        completed_cells = set()

        while generation_queue:
            current_query, current_cell_name = generation_queue.pop(0)
            if current_cell_name in completed_cells: continue

            with st.status(f"Synthesizing '{current_cell_name}'...", expanded=False) as status:
                retrieved_docs = self.retriever.get_relevant_documents(current_query)
                if not retrieved_docs:
                    status.update(label=f"Synthesis failed for '{current_cell_name}'", state="error"); continue
                
                context_str = "".join([f"--- CONTEXT {i+1}: {os.path.basename(doc.metadata.get('source', 'Unknown'))} ---\n{doc.page_content}\n\n" for i, doc in enumerate(retrieved_docs)])
                status.write(f"Found {len(retrieved_docs)} relevant contexts for '{current_cell_name}'.")
                mag_content = self.synthesis_chain.invoke({"context": context_str, "question": current_query})

                if not mag_content or not mag_content.strip().startswith("magic"):
                    status.update(label=f"LLM failed to generate valid .mag for '{current_cell_name}'", state="error"); continue
                
                file_path = os.path.join(GENERATED_MAG_DIR, f"{current_cell_name}.mag")
                with open(file_path, "w") as f: f.write(mag_content)
                completed_cells.add(current_cell_name)
                
                yield {"path": file_path, "name": current_cell_name, "code": mag_content}
                status.update(label=f"Completed synthesis for '{current_cell_name}'", state="complete")
                
                dependencies = self._parse_dependencies(mag_content)
                for dep_cell_name in dependencies:
                    if dep_cell_name not in completed_cells:
                        generation_queue.append((f"a {dep_cell_name.replace('_', ' ')} layout", dep_cell_name))

# ==============================================================================
# STREAMLIT APPLICATION
# ==============================================================================

st.set_page_config(layout="wide", page_title="Generative Chip Layout Designer")

# --- Sidebar for controls ---
with st.sidebar:
    st.header("⚙️ Design Controls")
    with st.form(key='design_form'):
        query = st.text_input("Enter your design description:", "a 4-input AND gate named MY_AND4")
        filename = st.text_input("Top-level filename:", "my_and4_gate.mag")
        submit_button = st.form_submit_button(label='Generate Layout')

    st.info("This app uses AI to generate chip layouts from text. Start by describing a logic gate.")


# --- Main page layout ---
st.title("🤖 Generative Chip Layout Designer")
st.write("This app uses a Large Language Model (LLM) with Retrieval-Augmented Generation (RAG) to hierarchically design and visualize `.mag` chip layouts.")
st.divider()

# --- Initialize the generator ---
try:
    generator = MagicLayoutGenerator(mag_files_directory=MAG_FILES_PATH)
except Exception as e:
    st.error(f"Failed to initialize the generator: {e}")
    st.stop()

if 'top_level_path' not in st.session_state:
    st.session_state.top_level_path = ""
if 'generation_complete' not in st.session_state:
    st.session_state.generation_complete = False

# --- Centered content column ---
_, center_col, _ = st.columns([1, 2, 1])
with center_col:
    if submit_button:
        st.session_state.generation_complete = False
        st.session_state.top_level_path = ""

        if not query or not filename:
            st.sidebar.error("Please provide both a description and a filename.")
        else:
            st.header("✨ Live Generation Process")
            try:
                generation_process = generator.generate_and_save_yield(query, filename)
                for result in generation_process:
                    with st.container(border=True):
                        cell_name, file_path, mag_code = result["name"], result["path"], result["code"]
                        st.subheader(f"Component: `{cell_name}`")
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            st.text("📄 Generated .mag Code")
                            st.code(mag_code, language='text', line_numbers=True)
                        with col2:
                            st.text("🖼️ Layout Visualization")
                            with st.spinner(f"Visualizing {cell_name}..."):
                                 fig = visualize_layout_for_streamlit(file_path)
                            if fig:
                                st.pyplot(fig)
                                plt.close(fig)
                            else:
                                st.warning(f"No visualization.")
                    st.write("") # Add vertical space
                    
                    if os.path.basename(file_path) == filename:
                         st.session_state.top_level_path = file_path
                
                st.session_state.generation_complete = True
                st.balloons()
                st.success("🎉 Top-level design generation complete!")

            except Exception as e:
                st.error(f"An error occurred during generation: {e}")
                st.session_state.generation_complete = False

    # --- Optimization / Correction Section ---
    if st.session_state.generation_complete and st.session_state.top_level_path:
        st.write("")
        with st.container(border=True):
            st.header("✏️ Optimize or Correct Design")
            st.write("The initial design is complete. You can now provide a new prompt to refine it, then click 'Generate Layout' in the sidebar.")
            
            st.subheader("Current Top-Level Design")
            fig = visualize_layout_for_streamlit(st.session_state.top_level_path)
            if fig:
                st.pyplot(fig)
                plt.close(fig)
            st.info("Example: 'Regenerate the layout but make the routing more compact.' To apply, enter the new prompt in the sidebar and click Generate.")