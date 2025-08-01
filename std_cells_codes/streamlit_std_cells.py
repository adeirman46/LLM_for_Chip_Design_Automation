# # # # import streamlit as st
# # # # import os
# # # # import shutil
# # # # import time
# # # # import matplotlib.pyplot as plt
# # # # import matplotlib.patches as patches
# # # # import random
# # # # import re
# # # # import asyncio
# # # # import nest_asyncio
# # # # import glob

# # # # # Fix for the asyncio event loop error in Streamlit's thread
# # # # nest_asyncio.apply()

# # # # from dotenv import load_dotenv
# # # # from langchain.prompts import PromptTemplate
# # # # from langchain_core.output_parsers import StrOutputParser
# # # # from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# # # # from langchain_community.vectorstores import FAISS
# # # # from langchain.docstore.document import Document

# # # # # --- Configuration ---
# # # # FAISS_INDEX_PATH = "faiss_mag_index_st"
# # # # GENERATED_MAG_DIR = "generated_mag_st"
# # # # MAG_FILES_PATH = 'std_cells_datasets/' # Adjust if your dataset is elsewhere

# # # # # --- Caching ---
# # # # _parsed_cell_cache = {}

# # # # # ==============================================================================
# # # # # VISUALIZER (To work with Streamlit)
# # # # # ==============================================================================
# # # # # This section contains the functions to parse and visualize the .mag files.
# # # # # It remains unchanged.

# # # # def parse_mag_data(file_path, current_dir=None):
# # # #     abs_file_path = os.path.abspath(file_path)
# # # #     if abs_file_path in _parsed_cell_cache:
# # # #         return _parsed_cell_cache[abs_file_path]
# # # #     if current_dir is None:
# # # #         current_dir = os.path.dirname(abs_file_path)
# # # #     full_file_path = os.path.join(current_dir, os.path.basename(file_path))
# # # #     current_dir_for_subcells = os.path.dirname(full_file_path)
# # # #     try:
# # # #         with open(full_file_path, 'r') as file:
# # # #             mag_content = file.read()
# # # #     except FileNotFoundError:
# # # #         st.warning(f"Referenced file '{os.path.basename(full_file_path)}' not found. Instance will be skipped.")
# # # #         return None
# # # #     except Exception as e:
# # # #         st.error(f"Error reading file '{full_file_path}': {e}")
# # # #         return None
# # # #     parsed_data = {"header": {}, "layers": {}, "instances": []}
# # # #     current_layer = None
# # # #     lines = mag_content.strip().split('\n')
# # # #     current_instance = None
# # # #     for line in lines:
# # # #         line = line.strip()
# # # #         if not line: continue
# # # #         parts = line.split()
# # # #         command = parts[0] if parts else ""
# # # #         if command == "tech" and len(parts) > 1:
# # # #             parsed_data["header"]["tech"] = parts[1]
# # # #         elif line.startswith("<<") and line.endswith(">>"):
# # # #             layer_name = line.strip("<<>> ").strip()
# # # #             if layer_name != "end":
# # # #                 current_layer = layer_name
# # # #                 if current_layer not in parsed_data["layers"]:
# # # #                     parsed_data["layers"][current_layer] = {"rects": [], "labels": []}
# # # #         elif command == "rect" and len(parts) == 5 and current_layer:
# # # #             try:
# # # #                 parsed_data["layers"][current_layer]["rects"].append({"x1": int(parts[1]), "y1": int(parts[2]), "x2": int(parts[3]), "y2": int(parts[4])})
# # # #             except ValueError: pass
# # # #         elif command == "rlabel" and len(parts) >= 8:
# # # #             label_layer = parts[1]
# # # #             if label_layer not in parsed_data["layers"]:
# # # #                  parsed_data["layers"][label_layer] = {"rects": [], "labels": []}
# # # #             parsed_data["layers"][label_layer]["labels"].append({"text": " ".join(parts[7:]), "x": int(parts[2]), "y": int(parts[3]), "rotation": int(parts[6])})
# # # #         elif command == "use":
# # # #             if current_instance: parsed_data["instances"].append(current_instance)
# # # #             if len(parts) >= 3:
# # # #                 sub_file_path = os.path.join(current_dir_for_subcells, f"{parts[1]}.mag")
# # # #                 current_instance = {"cell_type": parts[1], "instance_name": parts[2], "parsed_content": parse_mag_data(sub_file_path, current_dir_for_subcells), "transform": [1, 0, 0, 0, 1, 0], "box": [0, 0, 0, 0]}
# # # #                 if not current_instance["parsed_content"]: current_instance = None
# # # #         elif command == "transform" and current_instance:
# # # #             current_instance["transform"] = [int(v) for v in parts[1:]]
# # # #         elif command == "box" and current_instance:
# # # #             current_instance["box"] = [int(v) for v in parts[1:]]
# # # #         elif line == "<< end >>" and current_instance:
# # # #             parsed_data["instances"].append(current_instance)
# # # #             current_instance = None
# # # #     if current_instance: parsed_data["instances"].append(current_instance)
# # # #     _parsed_cell_cache[abs_file_path] = parsed_data
# # # #     return parsed_data

# # # # def visualize_layout_for_streamlit(file_path: str):
# # # #     if not os.path.exists(file_path):
# # # #         st.error(f"Cannot visualize. File not found at '{file_path}'")
# # # #         return None
# # # #     _parsed_cell_cache.clear()
# # # #     parsed_data = parse_mag_data(os.path.abspath(file_path))
# # # #     if not parsed_data: return None
# # # #     fig, ax = plt.subplots(figsize=(12, 9))
# # # #     min_x, max_x, min_y, max_y = float('inf'), float('-inf'), float('inf'), float('-inf')
# # # #     layer_colors = {}
# # # #     def get_random_color(): return (random.random(), random.random(), random.random())
# # # #     def _apply_transform(x, y, T): return (T[0] * x + T[1] * y + T[2], T[3] * x + T[4] * y + T[5])
# # # #     def _draw_elements(data_to_draw, current_transform=[1, 0, 0, 0, 1, 0]):
# # # #         nonlocal min_x, max_x, min_y, max_y
# # # #         for layer_name, layer_data in data_to_draw["layers"].items():
# # # #             if layer_name not in layer_colors: layer_colors[layer_name] = get_random_color()
# # # #             color = layer_colors[layer_name]
# # # #             for rect in layer_data.get("rects", []):
# # # #                 x1, y1, x2, y2 = rect["x1"], rect["y1"], rect["x2"], rect["y2"]
# # # #                 tx1, ty1 = _apply_transform(x1, y1, current_transform)
# # # #                 tx2, ty2 = _apply_transform(x2, y2, current_transform)
# # # #                 width, height = abs(tx2 - tx1), abs(ty2 - ty1)
# # # #                 x_start, y_start = min(tx1, tx2), min(ty1, ty2)
# # # #                 min_x, max_x = min(min_x, x_start), max(max_x, x_start + width)
# # # #                 min_y, max_y = min(min_y, y_start), max(max_y, y_start + height)
# # # #                 ax.add_patch(patches.Rectangle((x_start, y_start), width, height, linewidth=1, edgecolor='black', facecolor=color, alpha=0.7))
# # # #             for label in layer_data.get("labels", []):
# # # #                 tx, ty = _apply_transform(label["x"], label["y"], current_transform)
# # # #                 ax.text(tx, ty, label["text"], color='blue', fontsize=9, ha='center', va='center', rotation=label.get("rotation", 0),
# # # #                         bbox=dict(facecolor='white', alpha=0.6, edgecolor='blue', boxstyle='round,pad=0.2'), zorder=10)
# # # #         for instance in data_to_draw.get("instances", []):
# # # #             if instance.get("parsed_content"):
# # # #                 _draw_elements(instance["parsed_content"], instance["transform"])
# # # #                 box = instance.get("box", [0, 0, 0, 0])
# # # #                 center_x, center_y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
# # # #                 inst_center_x, inst_center_y = _apply_transform(center_x, center_y, instance["transform"])
# # # #                 ax.text(inst_center_x, inst_center_y, instance["instance_name"], color='darkred', fontsize=10, ha='center', va='center', fontweight='bold',
# # # #                         bbox=dict(facecolor='yellow', alpha=0.5, edgecolor='red', boxstyle='round,pad=0.3'), zorder=11)
# # # #     _draw_elements(parsed_data)
# # # #     if not all(v != float('inf') and v != float('-inf') for v in [min_x, max_x, min_y, max_y]):
# # # #         plt.close(fig)
# # # #         return None
# # # #     padding = (max_x - min_x) * 0.1 if (max_x > min_x) else 10
# # # #     ax.set_xlim(min_x - padding, max_x + padding)
# # # #     ax.set_ylim(min_y - padding, max_y + padding)
# # # #     ax.set_aspect('equal', adjustable='box')
# # # #     ax.set_title(f"Layout: {os.path.basename(file_path)}", fontsize=16)
# # # #     ax.set_xlabel("X Coordinate", fontsize=12)
# # # #     ax.set_ylabel("Y Coordinate", fontsize=12)
# # # #     ax.grid(True, linestyle='--', alpha=0.6)
# # # #     legend_patches = [patches.Patch(color=color, label=name, alpha=0.7) for name, color in layer_colors.items()]
# # # #     ax.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1.02, 0.5), title="Layers")
# # # #     plt.tight_layout(rect=[0, 0, 0.85, 1])
# # # #     return fig


# # # # # ==============================================================================
# # # # # DATA LOADING & PREPARATION (Cached)
# # # # # ==============================================================================

# # # # def _preprocess_mag_file(file_path: str):
# # # #     """Helper function to preprocess a single .mag file for vectorization."""
# # # #     with open(file_path, 'r') as f: content = f.read()
# # # #     cell_name = os.path.splitext(os.path.basename(file_path))[0]
# # # #     keyword_part = f"cell name is {cell_name}. function is {cell_name}. " * 5
# # # #     description = f"Magic layout for standard cell. {keyword_part}"
# # # #     match = re.search(r"#\s*this is\s+(.*)", content, re.IGNORECASE)
# # # #     if match: description += f"It is described as a {match.group(1).strip()}. "
# # # #     return Document(page_content=description + content, metadata={"source": file_path})

# # # # @st.cache_resource(show_spinner="Initializing Vector Store...")
# # # # def get_retriever(mag_files_directory: str):
# # # #     """
# # # #     Creates and caches the FAISS vector store and retriever.
# # # #     This resource-intensive function runs only once.
# # # #     """
# # # #     load_dotenv()
# # # #     api_key = os.getenv("GOOGLE_API_KEY")
# # # #     if not api_key: st.error("GOOGLE_API_KEY not found in .env file."); st.stop()
    
# # # #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

# # # #     if os.path.exists(FAISS_INDEX_PATH):
# # # #         vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
# # # #     else:
# # # #         if not os.path.exists(mag_files_directory):
# # # #             st.error(f"Dataset directory not found at '{mag_files_directory}'. Please check the `MAG_FILES_PATH` variable.")
# # # #             st.stop()
# # # #         mag_files = glob.glob(os.path.join(mag_files_directory, '**', '*.mag'), recursive=True)
# # # #         if not mag_files:
# # # #             st.error(f"No .mag files found in '{mag_files_directory}'."); st.stop()
        
# # # #         documents = [_preprocess_mag_file(file_path) for file_path in mag_files]
# # # #         vector_store = FAISS.from_documents(documents, embeddings)
# # # #         vector_store.save_local(FAISS_INDEX_PATH)
        
# # # #     return vector_store.as_retriever(search_kwargs={"k": 3})

# # # # # ==============================================================================
# # # # # LAYOUT GENERATOR CLASS
# # # # # ==============================================================================

# # # # class MagicLayoutGenerator:
# # # #     def __init__(self, retriever):
# # # #         """Initializes the generator with a pre-loaded retriever."""
# # # #         load_dotenv()
# # # #         api_key = os.getenv("GOOGLE_API_KEY")
# # # #         if not api_key: raise ValueError("GOOGLE_API_KEY not found.")
        
# # # #         self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=api_key, temperature=0.1)
# # # #         # The retriever is now passed in directly.
# # # #         self.retriever = retriever

# # # #         synthesis_prompt_template = """
# # # # You are a Magic VLSI layout expert. Your task is to generate a .mag file based on the user's QUESTION.
# # # # **CRITICAL INSTRUCTIONS:**
# # # # 1.  **Analyze CONTEXTS**: You will be given up to 3 contexts. Identify the SINGLE most relevant context.
# # # # 2.  **Prioritize the Best Context**: For simple gates, replicate the best context's geometry and rename labels. For complex cells, use component contexts to build the layout with 'use' commands and routing.
# # # # 3.  **DO NOT SIMPLIFY**: You are forbidden from simplifying the provided contexts. Replicate them.
# # # # 4.  **OUTPUT FORMAT**: Your response MUST BE ONLY the raw .mag file content, starting with 'magic' and ending with '<< end >>'.

# # # # CONTEXTS:
# # # # {context}

# # # # QUESTION:
# # # # {question}

# # # # ANSWER (A complete .mag file, based on the most relevant context):
# # # # """
# # # #         synthesis_prompt = PromptTemplate.from_template(synthesis_prompt_template)
# # # #         self.synthesis_chain = synthesis_prompt | self.llm | StrOutputParser()

# # # #     def _parse_dependencies(self, mag_content: str) -> set:
# # # #         return set(re.findall(r"^\s*use\s+([\w\d_]+)", mag_content, re.MULTILINE))

# # # #     def generate_and_save_yield(self, initial_query: str, top_level_filename: str):
# # # #         if os.path.exists(GENERATED_MAG_DIR):
# # # #             shutil.rmtree(GENERATED_MAG_DIR)
# # # #         os.makedirs(GENERATED_MAG_DIR, exist_ok=True)

# # # #         top_level_cell_name = os.path.splitext(top_level_filename)[0]
# # # #         generation_queue = [(initial_query, top_level_cell_name)]
# # # #         completed_cells = set()

# # # #         while generation_queue:
# # # #             current_query, current_cell_name = generation_queue.pop(0)
# # # #             if current_cell_name in completed_cells: continue

# # # #             with st.status(f"Synthesizing '{current_cell_name}'...", expanded=False) as status:
# # # #                 retrieved_docs = self.retriever.get_relevant_documents(current_query)
# # # #                 if not retrieved_docs:
# # # #                     status.update(label=f"Synthesis failed for '{current_cell_name}'", state="error"); continue
                
# # # #                 context_str = "".join([f"--- CONTEXT {i+1}: {os.path.basename(doc.metadata.get('source', 'Unknown'))} ---\n{doc.page_content}\n\n" for i, doc in enumerate(retrieved_docs)])
# # # #                 status.write(f"Found {len(retrieved_docs)} relevant contexts for '{current_cell_name}'.")
# # # #                 mag_content = self.synthesis_chain.invoke({"context": context_str, "question": current_query})

# # # #                 if not mag_content or not mag_content.strip().startswith("magic"):
# # # #                     status.update(label=f"LLM failed to generate valid .mag for '{current_cell_name}'", state="error"); continue
                
# # # #                 file_path = os.path.join(GENERATED_MAG_DIR, f"{current_cell_name}.mag")
# # # #                 with open(file_path, "w") as f: f.write(mag_content)
# # # #                 completed_cells.add(current_cell_name)
                
# # # #                 yield {"path": file_path, "name": current_cell_name, "code": mag_content}
# # # #                 status.update(label=f"Completed synthesis for '{current_cell_name}'", state="complete")
                
# # # #                 dependencies = self._parse_dependencies(mag_content)
# # # #                 for dep_cell_name in dependencies:
# # # #                     if dep_cell_name not in completed_cells:
# # # #                         generation_queue.append((f"a {dep_cell_name.replace('_', ' ')} layout", dep_cell_name))

# # # # # ==============================================================================
# # # # # STREAMLIT APPLICATION UI
# # # # # ==============================================================================

# # # # st.set_page_config(layout="wide", page_title="Generative Chip Layout Designer")

# # # # st.title("🤖 Generative Chip Layout Designer")
# # # # st.write("An AI-powered tool to generate and visualize VLSI layouts from text descriptions.")

# # # # with st.container(border=True):
# # # #     st.subheader("⚙️ Control Panel")
# # # #     with st.form(key='design_form'):
# # # #         query = st.text_input("**Design Prompt**", "a 4-to-1 multiplexer named MY_MUX4", help="Describe the logic gate or circuit you want to create.")
# # # #         filename = st.text_input("**Top-level Filename**", "my_mux4.mag", help="The filename for the main component.")
# # # #         submit_button = st.form_submit_button(label="✨ Generate Layout", use_container_width=True)

# # # # try:
# # # #     retriever = get_retriever(MAG_FILES_PATH)
# # # #     generator = MagicLayoutGenerator(retriever)
# # # # except Exception as e:
# # # #     st.error(f"💥 **Initialization Error:** {e}")
# # # #     st.stop()

# # # # if 'top_level_path' not in st.session_state:
# # # #     st.session_state.top_level_path = ""
# # # # if 'generation_complete' not in st.session_state:
# # # #     st.session_state.generation_complete = False

# # # # st.divider()

# # # # if submit_button:
# # # #     st.session_state.generation_complete = False
# # # #     st.session_state.top_level_path = ""

# # # #     if not query or not filename:
# # # #         st.error("Please provide both a design prompt and a filename in the control panel.")
# # # #     else:
# # # #         st.header("⚡ Live Generation Process")
# # # #         _, center_col, _ = st.columns([1, 2, 1])
# # # #         with center_col:
# # # #             try:
# # # #                 generation_process = generator.generate_and_save_yield(query, filename)
# # # #                 for result in generation_process:
# # # #                     with st.container(border=True):
# # # #                         cell_name, file_path, mag_code = result["name"], result["path"], result["code"]
# # # #                         st.subheader(f"Component: `{cell_name}`")
# # # #                         col1, col2 = st.columns(2)
# # # #                         with col1:
# # # #                             st.text("📄 Generated .mag Code")
# # # #                             st.code(mag_code, language='text', line_numbers=True)
# # # #                         with col2:
# # # #                             st.text("🖼️ Layout Visualization")
# # # #                             with st.spinner(f"Visualizing {cell_name}..."):
# # # #                                  fig = visualize_layout_for_streamlit(file_path)
# # # #                             if fig:
# # # #                                 st.pyplot(fig)
# # # #                                 plt.close(fig)
# # # #                             else:
# # # #                                 st.warning("No visualization available.")
# # # #                     st.write("") 

# # # #                     if os.path.basename(file_path) == filename:
# # # #                          st.session_state.top_level_path = file_path
                
# # # #                 st.session_state.generation_complete = True
# # # #                 st.balloons()

# # # #             except Exception as e:
# # # #                 st.error(f"An error occurred during generation: {e}")
# # # #                 st.session_state.generation_complete = False

# # # # if st.session_state.generation_complete:
# # # #     st.header("✅ Final Design & Next Steps")
# # # #     _, center_col, _ = st.columns([1, 2, 1])
# # # #     with center_col:
# # # #         with st.container(border=True):
# # # #             st.subheader(f"Top-Level Layout: `{os.path.basename(st.session_state.top_level_path)}`")
# # # #             fig = visualize_layout_for_streamlit(st.session_state.top_level_path)
# # # #             if fig:
# # # #                 st.pyplot(fig)
# # # #                 plt.close(fig)

# # # #             st.info(
# # # #                 "**To optimize or correct the design:**\n\n"
# # # #                 "1. Modify the **Design Prompt** in the Control Panel above.\n"
# # # #                 "2. Click **Generate Layout** again."
# # # #             )

# # # import streamlit as st
# # # import os
# # # import shutil
# # # import time
# # # import matplotlib.pyplot as plt
# # # import matplotlib.patches as patches
# # # import random
# # # import re
# # # import asyncio
# # # import nest_asyncio
# # # import glob

# # # # Fix for the asyncio event loop error in Streamlit's thread
# # # nest_asyncio.apply()

# # # from dotenv import load_dotenv
# # # from langchain.prompts import PromptTemplate
# # # from langchain_core.output_parsers import StrOutputParser
# # # from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# # # from langchain_community.vectorstores import FAISS
# # # from langchain.docstore.document import Document

# # # # --- Configuration ---
# # # FAISS_INDEX_PATH = "faiss_mag_index_st"
# # # GENERATED_MAG_DIR = "generated_mag_st"
# # # MAG_FILES_PATH = 'std_cells_datasets/'

# # # # ==============================================================================
# # # # VISUALIZATION LOGIC
# # # # ==============================================================================
# # # _parsed_cell_cache = {}

# # # def parse_mag_data_hierarchical(file_path, current_dir=None):
# # #     abs_file_path = os.path.abspath(file_path)
# # #     if abs_file_path in _parsed_cell_cache:
# # #         return _parsed_cell_cache[abs_file_path]
# # #     if current_dir is None: current_dir = os.path.dirname(abs_file_path)
# # #     full_file_path = os.path.join(current_dir, os.path.basename(file_path))
# # #     try:
# # #         with open(full_file_path, 'r') as file: mag_content = file.read()
# # #     except FileNotFoundError:
# # #         st.warning(f"Sub-cell file not found: '{os.path.basename(full_file_path)}'. Instance will be skipped.")
# # #         return None
# # #     except Exception as e:
# # #         st.error(f"Error reading file '{full_file_path}': {e}"); return None

# # #     parsed_data = {"header": {}, "layers": {}, "instances": []}
# # #     current_layer, current_instance = None, None
# # #     for line in mag_content.strip().split('\n'):
# # #         line = line.strip()
# # #         if not line: continue
# # #         parts = line.split()
# # #         command = parts[0] if parts else ""
# # #         if line.startswith("<<") and line.endswith(">>"):
# # #             layer_name = line.strip("<<>> ").strip()
# # #             if layer_name != "end":
# # #                 current_layer = layer_name
# # #                 if current_layer not in parsed_data["layers"]:
# # #                     parsed_data["layers"][current_layer] = {"rects": [], "labels": []}
# # #         elif command == "rect" and len(parts) == 5 and current_layer:
# # #             try: parsed_data["layers"][current_layer]["rects"].append({"x1": int(parts[1]), "y1": int(parts[2]), "x2": int(parts[3]), "y2": int(parts[4])})
# # #             except (ValueError, IndexError): pass
# # #         elif command == "use":
# # #             if current_instance: parsed_data["instances"].append(current_instance)
# # #             if len(parts) >= 3:
# # #                 sub_file_path = os.path.join(os.path.dirname(full_file_path), f"{parts[1]}.mag")
# # #                 current_instance = {"cell_type": parts[1], "instance_name": parts[2], "parsed_content": parse_mag_data_hierarchical(sub_file_path, os.path.dirname(full_file_path)),"transform": [1, 0, 0, 0, 1, 0], "box": [0, 0, 0, 0]}
# # #                 if not current_instance["parsed_content"]: current_instance = None
# # #         elif command == "transform" and current_instance:
# # #             try: current_instance["transform"] = [int(v) for v in parts[1:]]
# # #             except (ValueError, IndexError): pass
# # #         elif command == "box" and current_instance:
# # #             try: current_instance["box"] = [int(v) for v in parts[1:]]
# # #             except (ValueError, IndexError): pass
# # #         elif line == "<< end >>" and current_instance:
# # #             parsed_data["instances"].append(current_instance)
# # #             current_instance = None
# # #     if current_instance: parsed_data["instances"].append(current_instance)
# # #     _parsed_cell_cache[abs_file_path] = parsed_data
# # #     return parsed_data

# # # def visualize_hierarchical_layout(file_path: str):
# # #     _parsed_cell_cache.clear()
# # #     parsed_data = parse_mag_data_hierarchical(file_path)
# # #     if not parsed_data: return None
# # #     fig, ax = plt.subplots(figsize=(15, 12))
# # #     min_x, max_x, min_y, max_y = float('inf'), float('-inf'), float('inf'), float('-inf')
# # #     layer_colors = {}
# # #     def get_random_color(): return (random.random(), random.random(), random.random())
# # #     def _apply_transform(x, y, T): return (T[0] * x + T[1] * y + T[2], T[3] * x + T[4] * y + T[5])
# # #     def _draw_elements(data_to_draw, current_transform=[1, 0, 0, 0, 1, 0]):
# # #         nonlocal min_x, max_x, min_y, max_y
# # #         for layer_name, layer_data in data_to_draw["layers"].items():
# # #             if layer_name not in layer_colors: layer_colors[layer_name] = get_random_color()
# # #             color = layer_colors[layer_name]
# # #             for rect in layer_data.get("rects", []):
# # #                 tx1, ty1 = _apply_transform(rect["x1"], rect["y1"], current_transform)
# # #                 tx2, ty2 = _apply_transform(rect["x2"], rect["y2"], current_transform)
# # #                 width, height = abs(tx2 - tx1), abs(ty2 - ty1)
# # #                 x_start, y_start = min(tx1, tx2), min(ty1, ty2)
# # #                 min_x, max_x = min(min_x, x_start), max(max_x, x_start + width)
# # #                 min_y, max_y = min(min_y, y_start), max(max_y, y_start + height)
# # #                 ax.add_patch(patches.Rectangle((x_start, y_start), width, height, linewidth=1, edgecolor='black', facecolor=color, alpha=0.7))
# # #         for instance in data_to_draw.get("instances", []):
# # #             if instance.get("parsed_content"): _draw_elements(instance["parsed_content"], instance["transform"])
# # #     _draw_elements(parsed_data)
# # #     if not all(v != float('inf') and v != float('-inf') for v in [min_x, max_x, min_y, max_y]):
# # #         plt.close(fig); return None
# # #     padding = (max_x - min_x) * 0.1 if (max_x > min_x) else 10
# # #     ax.set_xlim(min_x - padding, max_x + padding)
# # #     ax.set_ylim(min_y - padding, max_y + padding)
# # #     ax.set_aspect('equal', adjustable='box'); ax.set_title(f"Hierarchical Layout: {os.path.basename(file_path)}", fontsize=16); ax.grid(True, linestyle='--', alpha=0.6)
# # #     return fig

# # # def get_layout_bounds(mag_content):
# # #     min_x, max_x, min_y, max_y = float('inf'), float('-inf'), float('inf'), float('-inf')
# # #     for line in mag_content.split('\n'):
# # #         if line.strip().startswith('rect'):
# # #             try:
# # #                 parts = line.split(); x1, y1, x2, y2 = map(int, parts[1:])
# # #                 min_x, max_x = min(min_x, x1, x2), max(max_x, x1, x2)
# # #                 min_y, max_y = min(min_y, y1, y2), max(max_y, y1, y2)
# # #             except (ValueError, IndexError): continue
# # #     return min_x, max_x, min_y, max_y

# # # def visualize_layout_streamed(mag_content: str):
# # #     if not mag_content: return
# # #     min_x, max_x, min_y, max_y = get_layout_bounds(mag_content)
# # #     if not all(v != float('inf') and v != float('-inf') for v in [min_x, max_x, min_y, max_y]): min_x, max_x, min_y, max_y = -10, 10, -10, 10
# # #     padding = (max_x - min_x) * 0.15 if (max_x > min_x) else 20
# # #     fig, ax = plt.subplots(figsize=(15, 12))
# # #     ax.set_aspect('equal', adjustable='box')
# # #     ax.set_xlim(min_x - padding, max_x + padding)
# # #     ax.set_ylim(min_y - padding, max_y + padding)
# # #     ax.set_title("Layout Generation Progress", fontsize=18)
# # #     ax.grid(True, linestyle='--', alpha=0.6)
# # #     layer_colors = {}
# # #     def get_random_color(): return (random.random(), random.random(), random.random())
# # #     current_layer = None
# # #     yield fig 
# # #     for line in mag_content.split('\n'):
# # #         line = line.strip()
# # #         if not line: continue
# # #         parts = line.split()
# # #         if line.startswith("<<") and line.endswith(">>"):
# # #             layer_name = line.strip("<<>> ").strip()
# # #             if layer_name != "end":
# # #                 current_layer = layer_name
# # #                 if current_layer not in layer_colors:
# # #                     layer_colors[current_layer] = get_random_color()
# # #                     legend_patches = [patches.Patch(color=color, label=name, alpha=0.7) for name, color in layer_colors.items()]
# # #                     ax.legend(handles=legend_patches, loc='upper right', title="Layers")
# # #         elif parts[0] == "rect" and len(parts) == 5 and current_layer:
# # #             try: x1, y1, x2, y2 = map(int, parts[1:5])
# # #             except (ValueError, IndexError): continue
# # #             width, height = abs(x2 - x1), abs(y2 - y1)
# # #             x_start, y_start = min(x1, x2), min(y1, y2)
# # #             rect_patch = patches.Rectangle((x_start, y_start), width, height, linewidth=1.5, edgecolor='black', facecolor=layer_colors[current_layer], alpha=0.75)
# # #             ax.add_patch(rect_patch)
# # #             yield fig
# # #     plt.close(fig)

# # # # ==============================================================================
# # # # DATA LOADING & AI GENERATION
# # # # ==============================================================================
# # # @st.cache_resource(show_spinner="Initializing Vector Store...")
# # # def get_retriever(mag_files_directory: str):
# # #     load_dotenv(); api_key = os.getenv("GOOGLE_API_KEY")
# # #     if not api_key: st.error("GOOGLE_API_KEY not found."); st.stop()
# # #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
# # #     def _preprocess_mag_file(file_path: str):
# # #         with open(file_path, 'r') as f: content = f.read()
# # #         cell_name = os.path.splitext(os.path.basename(file_path))[0]
# # #         return Document(page_content=f"Magic layout for standard cell. cell name is {cell_name}. function is {cell_name}. " + content, metadata={"source": file_path})
# # #     if os.path.exists(FAISS_INDEX_PATH):
# # #         vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
# # #     else:
# # #         if not os.path.exists(mag_files_directory): st.error(f"Dataset directory not found: '{mag_files_directory}'."); st.stop()
# # #         mag_files = glob.glob(os.path.join(mag_files_directory, '**', '*.mag'), recursive=True)
# # #         if not mag_files: st.error(f"No .mag files found in '{mag_files_directory}'."); st.stop()
# # #         documents = [_preprocess_mag_file(file_path) for file_path in mag_files]
# # #         vector_store = FAISS.from_documents(documents, embeddings)
# # #         vector_store.save_local(FAISS_INDEX_PATH)
# # #     return vector_store.as_retriever(search_kwargs={"k": 3})

# # # class MagicLayoutGenerator:
# # #     def __init__(self, retriever):
# # #         load_dotenv(); api_key = os.getenv("GOOGLE_API_KEY")
# # #         if not api_key: raise ValueError("GOOGLE_API_KEY not found.")
# # #         self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=api_key, temperature=0.1)
# # #         self.retriever = retriever
# # #         self.synthesis_chain = PromptTemplate.from_template("CONTEXTS:\n{context}\n\nQUESTION:\n{question}\n\nBased on the contexts, generate a .mag file for the question. The response must be ONLY the raw .mag file content, starting with 'magic' and ending with '<< end >>'.") | self.llm | StrOutputParser()
# # #         self.improvement_chain = PromptTemplate.from_template("ORIGINAL .MAG FILE:\n{original_mag}\n\nUSER'S IMPROVEMENT REQUEST:\n{improvement_request}\n\nRegenerate the .mag file to incorporate the request. The output MUST be a complete, valid .mag file.") | self.llm | StrOutputParser()
# # #     def generate_single_cell(self, query: str):
# # #         retrieved_docs = self.retriever.invoke(query)
# # #         if not retrieved_docs: return {"content": None, "context": "No relevant contexts found.", "dependencies": set()}
# # #         context_str = "".join([f"--- CONTEXT {i+1}: From file '{os.path.basename(doc.metadata.get('source', 'Unknown'))}' ---\n{doc.page_content}\n\n" for i, doc in enumerate(retrieved_docs)])
# # #         mag_content = self.synthesis_chain.invoke({"context": context_str, "question": query})
# # #         if not mag_content or not mag_content.strip().startswith("magic"): return {"content": None, "context": context_str, "dependencies": set()}
# # #         dependencies = set(re.findall(r"^\s*use\s+([\w\d_]+)", mag_content, re.MULTILINE))
# # #         return {"content": mag_content, "context": context_str, "dependencies": dependencies}
# # #     def improve_single_cell(self, original_mag_content: str, improvement_request: str):
# # #         new_mag_content = self.improvement_chain.invoke({"original_mag": original_mag_content, "improvement_request": improvement_request})
# # #         dependencies = set(re.findall(r"^\s*use\s+([\w\d_]+)", new_mag_content, re.MULTILINE))
# # #         return {"content": new_mag_content, "dependencies": dependencies}

# # # # ==============================================================================
# # # # STREAMLIT APPLICATION UI
# # # # ==============================================================================
# # # st.set_page_config(layout="wide", page_title="Interactive Chip Layout Designer")
# # # try:
# # #     retriever = get_retriever(MAG_FILES_PATH)
# # #     generator = MagicLayoutGenerator(retriever)
# # # except Exception as e:
# # #     st.error(f"💥 **Initialization Error:** {e}"); st.stop()

# # # if "generation_queue" not in st.session_state: st.session_state.generation_queue = []
# # # if "completed_cells" not in st.session_state: st.session_state.completed_cells = {}
# # # if "current_cell_data" not in st.session_state: st.session_state.current_cell_data = None
# # # if "mode" not in st.session_state: st.session_state.mode = "Automatic"

# # # st.title("🤖 Interactive Chip Layout Designer")
# # # st.write("An advanced AI tool for generating, visualizing, and iteratively refining VLSI layouts.")

# # # with st.container(border=True):
# # #     st.subheader("⚙️ Control Panel")
# # #     st.session_state.mode = st.radio("**Select Mode**", ["Automatic", "Strict Review"], horizontal=True)
# # #     with st.form(key='design_form'):
# # #         query = st.text_input("**Design Prompt**", "a 4-to-1 multiplexer named MY_MUX4")
# # #         filename = st.text_input("**Top-level Filename**", "my_mux4.mag")
# # #         if st.form_submit_button(label="🚀 Start New Generation", use_container_width=True):
# # #             if query and filename:
# # #                 st.session_state.generation_queue = [(query, os.path.splitext(filename)[0])]
# # #                 st.session_state.completed_cells = {}
# # #                 st.session_state.current_cell_data = None
# # #                 # FIX: Ensure directory exists but do NOT delete it.
# # #                 os.makedirs(GENERATED_MAG_DIR, exist_ok=True)
# # #             else: st.error("Please provide both a prompt and a filename.")
# # # st.divider()

# # # if st.session_state.generation_queue:
# # #     current_query, current_cell_name = st.session_state.generation_queue[0]
# # #     if st.session_state.current_cell_data is None:
# # #         with st.spinner(f"AI is generating '{current_cell_name}'..."):
# # #             result = generator.generate_single_cell(current_query)
# # #             st.session_state.current_cell_data = {"name": current_cell_name, "query": current_query, **result}
    
# # #     data = st.session_state.current_cell_data
# # #     cell_name, mag_content = data['name'], data['content']
# # #     st.header(f"Processing: `{cell_name}`")
    
# # #     if not mag_content:
# # #         st.error(f"Failed to generate content for `{cell_name}`. Skipping.")
# # #         st.session_state.generation_queue.pop(0); st.session_state.current_cell_data = None
# # #         if st.button("Continue"): st.experimental_rerun()
# # #     else:
# # #         file_path = os.path.join(GENERATED_MAG_DIR, f"{cell_name}.mag")
# # #         with open(file_path, "w") as f: f.write(mag_content)
            
# # #         plot_placeholder = st.empty()
# # #         with st.container():
# # #             col1, col2 = st.columns(2)
# # #             with col1: st.subheader("📄 Generated Code"); st.code(mag_content, language='text', line_numbers=True)
# # #             with col2: st.subheader("🧠 AI Context"); st.expander("View AI Context").text(data.get('context', 'N/A'))
        
# # #         if "use " in mag_content:
# # #             with plot_placeholder.container():
# # #                 st.info("Hierarchical design detected. Rendering static layout.", icon="🏗️")
# # #                 fig = visualize_hierarchical_layout(file_path)
# # #                 if fig: st.pyplot(fig); plt.close(fig)
# # #                 else: st.warning("Could not render hierarchical layout.")
# # #         else:
# # #             with plot_placeholder.container():
# # #                 st.info("Flat design detected. Rendering animated layout.", icon="🎬")
# # #                 plot_area = st.empty()
# # #                 for fig in visualize_layout_streamed(mag_content):
# # #                     plot_area.pyplot(fig)
# # #                     time.sleep(0.02)
        
# # #         if st.session_state.mode == "Strict Review":
# # #             st.subheader("🔬 Review Component")
# # #             with st.container(border=True):
# # #                 with st.form("review_form"):
# # #                     improvement_prompt = st.text_area("Improvement Request (optional)", placeholder="e.g., Make the routing more compact.")
# # #                     approve_button = st.form_submit_button("✅ Looks Good, Continue", use_container_width=True)
# # #                     improve_button = st.form_submit_button("🔄 Improve This Component", use_container_width=True)
# # #                 if approve_button:
# # #                     st.session_state.completed_cells[cell_name] = mag_content
# # #                     st.session_state.generation_queue.pop(0)
# # #                     for dep in data.get('dependencies', set()):
# # #                         if dep not in st.session_state.completed_cells and all(dep != item[1] for item in st.session_state.generation_queue):
# # #                             st.session_state.generation_queue.append((f"a {dep} layout", dep))
# # #                     st.session_state.current_cell_data = None
# # #                     st.experimental_rerun()
# # #                 if improve_button and improvement_prompt:
# # #                     with st.spinner(f"AI is improving '{cell_name}'..."):
# # #                         improved_result = generator.improve_single_cell(mag_content, improvement_prompt)
# # #                         st.session_state.current_cell_data['content'] = improved_result['content']
# # #                         st.session_state.current_cell_data['dependencies'] = improved_result['dependencies']
# # #                     st.experimental_rerun()
# # #         else:
# # #             st.session_state.completed_cells[cell_name] = mag_content
# # #             st.session_state.generation_queue.pop(0)
# # #             for dep in data.get('dependencies', set()):
# # #                 if dep not in st.session_state.completed_cells and all(dep != item[1] for item in st.session_state.generation_queue):
# # #                     st.session_state.generation_queue.append((f"a {dep} layout", dep))
# # #             st.session_state.current_cell_data = None
# # #             st.success(f"Automatically approved '{cell_name}'. Continuing...")
# # #             time.sleep(1)
# # #             st.experimental_rerun()

# # # elif st.session_state.completed_cells:
# # #     st.balloons(); st.header("🎉 Generation Complete!"); st.write("All components have been successfully generated.")

# # import streamlit as st
# # import os
# # import shutil
# # import time
# # import matplotlib.pyplot as plt
# # import matplotlib.patches as patches
# # import random
# # import re
# # import asyncio
# # import nest_asyncio
# # import glob

# # # Fix for the asyncio event loop error in Streamlit's thread
# # nest_asyncio.apply()

# # from dotenv import load_dotenv
# # from langchain.prompts import PromptTemplate
# # from langchain_core.output_parsers import StrOutputParser
# # from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# # from langchain_community.vectorstores import FAISS
# # from langchain.docstore.document import Document

# # # --- Configuration ---
# # FAISS_INDEX_PATH = "faiss_mag_index_st"
# # GENERATED_MAG_DIR = "generated_mag_st"
# # MAG_FILES_PATH = 'std_cells_datasets/'

# # # ==============================================================================
# # # VISUALIZATION LOGIC
# # # ==============================================================================
# # _parsed_cell_cache = {}

# # def parse_mag_data_hierarchical(file_path, current_dir=None):
# #     abs_file_path = os.path.abspath(file_path)
# #     if abs_file_path in _parsed_cell_cache:
# #         return _parsed_cell_cache[abs_file_path]
# #     if current_dir is None: current_dir = os.path.dirname(abs_file_path)
# #     full_file_path = os.path.join(current_dir, os.path.basename(file_path))
# #     try:
# #         with open(full_file_path, 'r') as file: mag_content = file.read()
# #     except FileNotFoundError:
# #         st.warning(f"Sub-cell file not found: '{os.path.basename(full_file_path)}'. Instance will be skipped.")
# #         return None
# #     except Exception as e:
# #         st.error(f"Error reading file '{full_file_path}': {e}"); return None

# #     parsed_data = {"header": {}, "layers": {}, "instances": []}
# #     current_layer, current_instance = None, None
# #     for line in mag_content.strip().split('\n'):
# #         line = line.strip()
# #         if not line: continue
# #         parts = line.split()
# #         command = parts[0] if parts else ""
# #         if line.startswith("<<") and line.endswith(">>"):
# #             layer_name = line.strip("<<>> ").strip()
# #             if layer_name != "end":
# #                 current_layer = layer_name
# #                 if current_layer not in parsed_data["layers"]:
# #                     parsed_data["layers"][current_layer] = {"rects": [], "labels": []}
# #         elif command == "rect" and len(parts) == 5 and current_layer:
# #             try: parsed_data["layers"][current_layer]["rects"].append({"x1": int(parts[1]), "y1": int(parts[2]), "x2": int(parts[3]), "y2": int(parts[4])})
# #             except (ValueError, IndexError): pass
# #         elif command == "use":
# #             if current_instance: parsed_data["instances"].append(current_instance)
# #             if len(parts) >= 3:
# #                 sub_file_path = os.path.join(os.path.dirname(full_file_path), f"{parts[1]}.mag")
# #                 current_instance = {"cell_type": parts[1], "instance_name": parts[2], "parsed_content": parse_mag_data_hierarchical(sub_file_path, os.path.dirname(full_file_path)),"transform": [1, 0, 0, 0, 1, 0], "box": [0, 0, 0, 0]}
# #                 if not current_instance["parsed_content"]: current_instance = None
# #         elif command == "transform" and current_instance:
# #             try: current_instance["transform"] = [int(v) for v in parts[1:]]
# #             except (ValueError, IndexError): pass
# #         elif command == "box" and current_instance:
# #             try: current_instance["box"] = [int(v) for v in parts[1:]]
# #             except (ValueError, IndexError): pass
# #         elif line == "<< end >>" and current_instance:
# #             parsed_data["instances"].append(current_instance)
# #             current_instance = None
# #     if current_instance: parsed_data["instances"].append(current_instance)
# #     _parsed_cell_cache[abs_file_path] = parsed_data
# #     return parsed_data

# # def visualize_hierarchical_layout(file_path: str):
# #     _parsed_cell_cache.clear()
# #     parsed_data = parse_mag_data_hierarchical(file_path)
# #     if not parsed_data: return None
# #     fig, ax = plt.subplots(figsize=(15, 12))
# #     min_x, max_x, min_y, max_y = float('inf'), float('-inf'), float('inf'), float('-inf')
# #     layer_colors = {}
# #     def get_random_color(): return (random.random(), random.random(), random.random())
# #     def _apply_transform(x, y, T): return (T[0] * x + T[1] * y + T[2], T[3] * x + T[4] * y + T[5])
# #     def _draw_elements(data_to_draw, current_transform=[1, 0, 0, 0, 1, 0]):
# #         nonlocal min_x, max_x, min_y, max_y
# #         for layer_name, layer_data in data_to_draw["layers"].items():
# #             if layer_name not in layer_colors: layer_colors[layer_name] = get_random_color()
# #             color = layer_colors[layer_name]
# #             for rect in layer_data.get("rects", []):
# #                 tx1, ty1 = _apply_transform(rect["x1"], rect["y1"], current_transform)
# #                 tx2, ty2 = _apply_transform(rect["x2"], rect["y2"], current_transform)
# #                 width, height = abs(tx2 - tx1), abs(ty2 - ty1)
# #                 x_start, y_start = min(tx1, tx2), min(ty1, ty2)
# #                 min_x, max_x = min(min_x, x_start), max(max_x, x_start + width)
# #                 min_y, max_y = min(min_y, y_start), max(max_y, y_start + height)
# #                 ax.add_patch(patches.Rectangle((x_start, y_start), width, height, linewidth=1, edgecolor='black', facecolor=color, alpha=0.7))
# #         for instance in data_to_draw.get("instances", []):
# #             if instance.get("parsed_content"): _draw_elements(instance["parsed_content"], instance["transform"])
# #     _draw_elements(parsed_data)
# #     if not all(v != float('inf') and v != float('-inf') for v in [min_x, max_x, min_y, max_y]):
# #         plt.close(fig); return None
# #     padding = (max_x - min_x) * 0.1 if (max_x > min_x) else 10
# #     ax.set_xlim(min_x - padding, max_x + padding)
# #     ax.set_ylim(min_y - padding, max_y + padding)
# #     ax.set_aspect('equal', adjustable='box'); ax.set_title(f"Hierarchical Layout: {os.path.basename(file_path)}", fontsize=16); ax.grid(True, linestyle='--', alpha=0.6)
# #     return fig

# # def get_layout_bounds(mag_content):
# #     min_x, max_x, min_y, max_y = float('inf'), float('-inf'), float('inf'), float('-inf')
# #     for line in mag_content.split('\n'):
# #         if line.strip().startswith('rect'):
# #             try:
# #                 parts = line.split(); x1, y1, x2, y2 = map(int, parts[1:])
# #                 min_x, max_x = min(min_x, x1, x2), max(max_x, x1, x2)
# #                 min_y, max_y = min(min_y, y1, y2), max(max_y, y1, y2)
# #             except (ValueError, IndexError): continue
# #     return min_x, max_x, min_y, max_y

# # def visualize_layout_streamed(mag_content: str):
# #     if not mag_content: return
# #     min_x, max_x, min_y, max_y = get_layout_bounds(mag_content)
# #     if not all(v != float('inf') and v != float('-inf') for v in [min_x, max_x, min_y, max_y]): min_x, max_x, min_y, max_y = -10, 10, -10, 10
# #     padding = (max_x - min_x) * 0.15 if (max_x > min_x) else 20
# #     fig, ax = plt.subplots(figsize=(15, 12))
# #     ax.set_aspect('equal', adjustable='box')
# #     ax.set_xlim(min_x - padding, max_x + padding)
# #     ax.set_ylim(min_y - padding, max_y + padding)
# #     ax.set_title("Layout Generation Progress", fontsize=18)
# #     ax.grid(True, linestyle='--', alpha=0.6)
# #     layer_colors = {}
# #     def get_random_color(): return (random.random(), random.random(), random.random())
# #     current_layer = None
# #     yield fig 
# #     for line in mag_content.split('\n'):
# #         line = line.strip()
# #         if not line: continue
# #         parts = line.split()
# #         if line.startswith("<<") and line.endswith(">>"):
# #             layer_name = line.strip("<<>> ").strip()
# #             if layer_name != "end":
# #                 current_layer = layer_name
# #                 if current_layer not in layer_colors:
# #                     layer_colors[current_layer] = get_random_color()
# #                     legend_patches = [patches.Patch(color=color, label=name, alpha=0.7) for name, color in layer_colors.items()]
# #                     ax.legend(handles=legend_patches, loc='upper right', title="Layers")
# #         elif parts[0] == "rect" and len(parts) == 5 and current_layer:
# #             try: x1, y1, x2, y2 = map(int, parts[1:5])
# #             except (ValueError, IndexError): continue
# #             width, height = abs(x2 - x1), abs(y2 - y1)
# #             x_start, y_start = min(x1, x2), min(y1, y2)
# #             rect_patch = patches.Rectangle((x_start, y_start), width, height, linewidth=1.5, edgecolor='black', facecolor=layer_colors[current_layer], alpha=0.75)
# #             ax.add_patch(rect_patch)
# #             yield fig
# #     plt.close(fig)

# # # ==============================================================================
# # # DATA LOADING & AI GENERATION
# # # ==============================================================================
# # @st.cache_resource(show_spinner="Initializing Vector Store...")
# # def get_retriever(mag_files_directory: str):
# #     load_dotenv(); api_key = os.getenv("GOOGLE_API_KEY")
# #     if not api_key: st.error("GOOGLE_API_KEY not found."); st.stop()
# #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
# #     def _preprocess_mag_file(file_path: str):
# #         with open(file_path, 'r') as f: content = f.read()
# #         cell_name = os.path.splitext(os.path.basename(file_path))[0]
# #         return Document(page_content=f"Magic layout for standard cell. cell name is {cell_name}. function is {cell_name}. " + content, metadata={"source": file_path})
# #     if os.path.exists(FAISS_INDEX_PATH):
# #         vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
# #     else:
# #         if not os.path.exists(mag_files_directory): st.error(f"Dataset directory not found: '{mag_files_directory}'."); st.stop()
# #         mag_files = glob.glob(os.path.join(mag_files_directory, '**', '*.mag'), recursive=True)
# #         if not mag_files: st.error(f"No .mag files found in '{mag_files_directory}'."); st.stop()
# #         documents = [_preprocess_mag_file(file_path) for file_path in mag_files]
# #         vector_store = FAISS.from_documents(documents, embeddings)
# #         vector_store.save_local(FAISS_INDEX_PATH)
# #     return vector_store.as_retriever(search_kwargs={"k": 3})

# # class MagicLayoutGenerator:
# #     def __init__(self, retriever):
# #         load_dotenv(); api_key = os.getenv("GOOGLE_API_KEY")
# #         if not api_key: raise ValueError("GOOGLE_API_KEY not found.")
# #         self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=api_key, temperature=0.1)
# #         self.retriever = retriever
# #         self.synthesis_chain = PromptTemplate.from_template("CONTEXTS:\n{context}\n\nQUESTION:\n{question}\n\nBased on the contexts, generate a .mag file for the question. The response must be ONLY the raw .mag file content, starting with 'magic' and ending with '<< end >>'.") | self.llm | StrOutputParser()
# #         self.improvement_chain = PromptTemplate.from_template("ORIGINAL .MAG FILE:\n{original_mag}\n\nUSER'S IMPROVEMENT REQUEST:\n{improvement_request}\n\nRegenerate the .mag file to incorporate the request. The output MUST be a complete, valid .mag file.") | self.llm | StrOutputParser()
# #     def generate_single_cell(self, query: str):
# #         retrieved_docs = self.retriever.invoke(query)
# #         if not retrieved_docs: return {"content": None, "context": "No relevant contexts found.", "dependencies": set()}
# #         context_str = "".join([f"--- CONTEXT {i+1}: From file '{os.path.basename(doc.metadata.get('source', 'Unknown'))}' ---\n{doc.page_content}\n\n" for i, doc in enumerate(retrieved_docs)])
# #         mag_content = self.synthesis_chain.invoke({"context": context_str, "question": query})
# #         if not mag_content or not mag_content.strip().startswith("magic"): return {"content": None, "context": context_str, "dependencies": set()}
# #         dependencies = set(re.findall(r"^\s*use\s+([\w\d_]+)", mag_content, re.MULTILINE))
# #         return {"content": mag_content, "context": context_str, "dependencies": dependencies}
# #     def improve_single_cell(self, original_mag_content: str, improvement_request: str):
# #         new_mag_content = self.improvement_chain.invoke({"original_mag": original_mag_content, "improvement_request": improvement_request})
# #         dependencies = set(re.findall(r"^\s*use\s+([\w\d_]+)", new_mag_content, re.MULTILINE))
# #         return {"content": new_mag_content, "dependencies": dependencies}

# # # ==============================================================================
# # # STREAMLIT APPLICATION UI
# # # ==============================================================================
# # st.set_page_config(layout="wide", page_title="Interactive Chip Layout Designer")
# # try:
# #     retriever = get_retriever(MAG_FILES_PATH)
# #     generator = MagicLayoutGenerator(retriever)
# # except Exception as e:
# #     st.error(f"💥 **Initialization Error:** {e}"); st.stop()

# # if "generation_queue" not in st.session_state: st.session_state.generation_queue = []
# # if "completed_cells" not in st.session_state: st.session_state.completed_cells = {}
# # if "current_cell_data" not in st.session_state: st.session_state.current_cell_data = None
# # if "mode" not in st.session_state: st.session_state.mode = "Automatic"

# # st.title("🤖 Interactive Chip Layout Designer")
# # st.write("An advanced AI tool for generating, visualizing, and iteratively refining VLSI layouts.")

# # with st.container(border=True):
# #     st.subheader("⚙️ Control Panel")
# #     st.session_state.mode = st.radio("**Select Mode**", ["Automatic", "Strict Review"], horizontal=True)
# #     with st.form(key='design_form'):
# #         query = st.text_input("**Design Prompt**", "a 4-to-1 multiplexer named MY_MUX4")
# #         filename = st.text_input("**Top-level Filename**", "my_mux4.mag")
# #         if st.form_submit_button(label="🚀 Start New Generation", use_container_width=True):
# #             if query and filename:
# #                 st.session_state.generation_queue = [(query, os.path.splitext(filename)[0])]
# #                 st.session_state.completed_cells = {}
# #                 st.session_state.current_cell_data = None
# #                 os.makedirs(GENERATED_MAG_DIR, exist_ok=True)
# #             else: st.error("Please provide both a prompt and a filename.")
# # st.divider()

# # if st.session_state.generation_queue:
# #     current_query, current_cell_name = st.session_state.generation_queue[0]
# #     if st.session_state.current_cell_data is None:
# #         with st.spinner(f"AI is generating '{current_cell_name}'..."):
# #             result = generator.generate_single_cell(current_query)
# #             st.session_state.current_cell_data = {"name": current_cell_name, "query": current_query, **result}
    
# #     data = st.session_state.current_cell_data
# #     cell_name, mag_content = data['name'], data['content']
# #     st.header(f"Processing: `{cell_name}`")
    
# #     if not mag_content:
# #         st.error(f"Failed to generate content for `{cell_name}`. Skipping.")
# #         st.session_state.generation_queue.pop(0); st.session_state.current_cell_data = None
# #         if st.button("Continue"): st.experimental_rerun()
# #     else:
# #         file_path = os.path.join(GENERATED_MAG_DIR, f"{cell_name}.mag")
# #         with open(file_path, "w") as f: f.write(mag_content)
            
# #         plot_placeholder = st.empty()
# #         with st.container():
# #             col1, col2 = st.columns(2)
# #             with col1: st.subheader("📄 Generated Code"); st.code(mag_content, language='text', line_numbers=True)
# #             with col2: st.subheader("🧠 AI Context"); st.expander("View AI Context").text(data.get('context', 'N/A'))
        
# #         if "use " in mag_content:
# #             with plot_placeholder.container():
# #                 st.info("Hierarchical design detected. Rendering static layout.", icon="🏗️")
# #                 fig = visualize_hierarchical_layout(file_path)
# #                 if fig: st.pyplot(fig); plt.close(fig)
# #                 else: st.warning("Could not render hierarchical layout.")
# #         else:
# #             with plot_placeholder.container():
# #                 st.info("Flat design detected. Rendering animated layout.", icon="🎬")
# #                 plot_area = st.empty()
# #                 for fig in visualize_layout_streamed(mag_content):
# #                     plot_area.pyplot(fig)
# #                     time.sleep(0.02)
        
# #         if st.session_state.mode == "Strict Review":
# #             st.subheader("🔬 Review Component")
# #             # --- FIX: Added informational message to guide the user ---
# #             st.info("The generation is paused for your review. Please approve the component or request an improvement below to continue.", icon="✋")
# #             with st.container(border=True):
# #                 with st.form("review_form"):
# #                     improvement_prompt = st.text_area("Improvement Request (optional)", placeholder="e.g., Make the routing more compact.")
# #                     approve_button = st.form_submit_button("✅ Looks Good, Continue", use_container_width=True)
# #                     improve_button = st.form_submit_button("🔄 Improve This Component", use_container_width=True)
# #                 if approve_button:
# #                     st.session_state.completed_cells[cell_name] = mag_content
# #                     st.session_state.generation_queue.pop(0)
# #                     for dep in data.get('dependencies', set()):
# #                         if dep not in st.session_state.completed_cells and all(dep != item[1] for item in st.session_state.generation_queue):
# #                             st.session_state.generation_queue.append((f"a {dep} layout", dep))
# #                     st.session_state.current_cell_data = None
# #                     st.experimental_rerun()
# #                 if improve_button and improvement_prompt:
# #                     with st.spinner(f"AI is improving '{cell_name}'..."):
# #                         improved_result = generator.improve_single_cell(mag_content, improvement_prompt)
# #                         st.session_state.current_cell_data['content'] = improved_result['content']
# #                         st.session_state.current_cell_data['dependencies'] = improved_result['dependencies']
# #                     st.experimental_rerun()
# #         else:
# #             st.session_state.completed_cells[cell_name] = mag_content
# #             st.session_state.generation_queue.pop(0)
# #             for dep in data.get('dependencies', set()):
# #                 if dep not in st.session_state.completed_cells and all(dep != item[1] for item in st.session_state.generation_queue):
# #                     st.session_state.generation_queue.append((f"a {dep} layout", dep))
# #             st.session_state.current_cell_data = None
# #             st.success(f"Automatically approved '{cell_name}'. Continuing...")
# #             time.sleep(1)
# #             st.experimental_rerun()

# # elif st.session_state.completed_cells:
# #     st.balloons(); st.header("🎉 Generation Complete!"); st.write("All components have been successfully generated.")

# import streamlit as st
# import os
# import shutil
# import time
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import random
# import re
# import asyncio
# import nest_asyncio
# import glob

# # Fix for the asyncio event loop error in Streamlit's thread
# nest_asyncio.apply()

# from dotenv import load_dotenv
# from langchain.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.docstore.document import Document

# # --- Configuration ---
# FAISS_INDEX_PATH = "faiss_mag_index_st"
# GENERATED_MAG_DIR = "generated_mag_st"
# MAG_FILES_PATH = 'std_cells_datasets/'

# # ==============================================================================
# # VISUALIZATION LOGIC
# # ==============================================================================
# _parsed_cell_cache = {}

# def parse_mag_data_hierarchical(file_path, current_dir=None):
#     abs_file_path = os.path.abspath(file_path)
#     if abs_file_path in _parsed_cell_cache:
#         return _parsed_cell_cache[abs_file_path]
#     if current_dir is None: current_dir = os.path.dirname(abs_file_path)
#     full_file_path = os.path.join(current_dir, os.path.basename(file_path))
#     try:
#         with open(full_file_path, 'r') as file: mag_content = file.read()
#     except FileNotFoundError:
#         st.warning(f"Sub-cell file not found: '{os.path.basename(full_file_path)}'. Instance will be skipped.")
#         return None
#     except Exception as e:
#         st.error(f"Error reading file '{full_file_path}': {e}"); return None

#     parsed_data = {"header": {}, "layers": {}, "instances": []}
#     current_layer, current_instance = None, None
#     for line in mag_content.strip().split('\n'):
#         line = line.strip()
#         if not line: continue
#         parts = line.split()
#         command = parts[0] if parts else ""
#         if line.startswith("<<") and line.endswith(">>"):
#             layer_name = line.strip("<<>> ").strip()
#             if layer_name != "end":
#                 current_layer = layer_name
#                 if current_layer not in parsed_data["layers"]:
#                     parsed_data["layers"][current_layer] = {"rects": [], "labels": []}
#         elif command == "rect" and len(parts) == 5 and current_layer:
#             try: parsed_data["layers"][current_layer]["rects"].append({"x1": int(parts[1]), "y1": int(parts[2]), "x2": int(parts[3]), "y2": int(parts[4])})
#             except (ValueError, IndexError): pass
#         elif command == "use":
#             if current_instance: parsed_data["instances"].append(current_instance)
#             if len(parts) >= 3:
#                 sub_file_path = os.path.join(os.path.dirname(full_file_path), f"{parts[1]}.mag")
#                 current_instance = {"cell_type": parts[1], "instance_name": parts[2], "parsed_content": parse_mag_data_hierarchical(sub_file_path, os.path.dirname(full_file_path)),"transform": [1, 0, 0, 0, 1, 0], "box": [0, 0, 0, 0]}
#                 if not current_instance["parsed_content"]: current_instance = None
#         elif command == "transform" and current_instance:
#             try: current_instance["transform"] = [int(v) for v in parts[1:]]
#             except (ValueError, IndexError): pass
#         elif command == "box" and current_instance:
#             try: current_instance["box"] = [int(v) for v in parts[1:]]
#             except (ValueError, IndexError): pass
#         elif line == "<< end >>" and current_instance:
#             parsed_data["instances"].append(current_instance)
#             current_instance = None
#     if current_instance: parsed_data["instances"].append(current_instance)
#     _parsed_cell_cache[abs_file_path] = parsed_data
#     return parsed_data

# def visualize_hierarchical_layout(file_path: str):
#     _parsed_cell_cache.clear()
#     parsed_data = parse_mag_data_hierarchical(file_path)
#     if not parsed_data: return None
#     fig, ax = plt.subplots(figsize=(15, 12))
#     min_x, max_x, min_y, max_y = float('inf'), float('-inf'), float('inf'), float('-inf')
#     layer_colors = {}
#     def get_random_color(): return (random.random(), random.random(), random.random())
#     def _apply_transform(x, y, T): return (T[0] * x + T[1] * y + T[2], T[3] * x + T[4] * y + T[5])
#     def _draw_elements(data_to_draw, current_transform=[1, 0, 0, 0, 1, 0]):
#         nonlocal min_x, max_x, min_y, max_y
#         for layer_name, layer_data in data_to_draw["layers"].items():
#             if layer_name not in layer_colors: layer_colors[layer_name] = get_random_color()
#             color = layer_colors[layer_name]
#             for rect in layer_data.get("rects", []):
#                 tx1, ty1 = _apply_transform(rect["x1"], rect["y1"], current_transform)
#                 tx2, ty2 = _apply_transform(rect["x2"], rect["y2"], current_transform)
#                 width, height = abs(tx2 - tx1), abs(ty2 - ty1)
#                 x_start, y_start = min(tx1, tx2), min(ty1, ty2)
#                 min_x, max_x = min(min_x, x_start), max(max_x, x_start + width)
#                 min_y, max_y = min(min_y, y_start), max(max_y, y_start + height)
#                 ax.add_patch(patches.Rectangle((x_start, y_start), width, height, linewidth=1, edgecolor='black', facecolor=color, alpha=0.7))
#         for instance in data_to_draw.get("instances", []):
#             if instance.get("parsed_content"): _draw_elements(instance["parsed_content"], instance["transform"])
#     _draw_elements(parsed_data)
#     if not all(v != float('inf') and v != float('-inf') for v in [min_x, max_x, min_y, max_y]):
#         plt.close(fig); return None
#     padding = (max_x - min_x) * 0.1 if (max_x > min_x) else 10
#     ax.set_xlim(min_x - padding, max_x + padding)
#     ax.set_ylim(min_y - padding, max_y + padding)
#     ax.set_aspect('equal', adjustable='box'); ax.set_title(f"Hierarchical Layout: {os.path.basename(file_path)}", fontsize=16); ax.grid(True, linestyle='--', alpha=0.6)
#     return fig

# # ==============================================================================
# # DATA LOADING & AI GENERATION
# # ==============================================================================
# @st.cache_resource(show_spinner="Initializing Vector Store...")
# def get_retriever(mag_files_directory: str):
#     load_dotenv(); api_key = os.getenv("GOOGLE_API_KEY")
#     if not api_key: st.error("GOOGLE_API_KEY not found."); st.stop()
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
#     def _preprocess_mag_file(file_path: str):
#         with open(file_path, 'r') as f: content = f.read()
#         cell_name = os.path.splitext(os.path.basename(file_path))[0]
#         return Document(page_content=f"Magic layout for standard cell. cell name is {cell_name}. function is {cell_name}. " + content, metadata={"source": file_path})
#     if os.path.exists(FAISS_INDEX_PATH):
#         vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
#     else:
#         if not os.path.exists(mag_files_directory): st.error(f"Dataset directory not found: '{mag_files_directory}'."); st.stop()
#         mag_files = glob.glob(os.path.join(mag_files_directory, '**', '*.mag'), recursive=True)
#         if not mag_files: st.error(f"No .mag files found in '{mag_files_directory}'."); st.stop()
#         documents = [_preprocess_mag_file(file_path) for file_path in mag_files]
#         vector_store = FAISS.from_documents(documents, embeddings)
#         vector_store.save_local(FAISS_INDEX_PATH)
#     return vector_store.as_retriever(search_kwargs={"k": 3})

# class MagicLayoutGenerator:
#     def __init__(self, retriever):
#         load_dotenv(); api_key = os.getenv("GOOGLE_API_KEY")
#         if not api_key: raise ValueError("GOOGLE_API_KEY not found.")
#         self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=api_key, temperature=0.1)
#         self.retriever = retriever
#         self.synthesis_chain = PromptTemplate.from_template("CONTEXTS:\n{context}\n\nQUESTION:\n{question}\n\nBased on the contexts, generate a .mag file for the question. The response must be ONLY the raw .mag file content, starting with 'magic' and ending with '<< end >>'.") | self.llm
#         self.improvement_chain = PromptTemplate.from_template("ORIGINAL .MAG FILE:\n{original_mag}\n\nUSER'S IMPROVEMENT REQUEST:\n{improvement_request}\n\nRegenerate the .mag file to incorporate the request. The output MUST be a complete, valid .mag file.") | self.llm

#     def stream_single_cell(self, query: str):
#         """Yields context first, then streams the LLM response for the .mag file."""
#         retrieved_docs = self.retriever.invoke(query)
#         if not retrieved_docs:
#             yield {"type": "context", "data": "No relevant contexts found."}
#             yield {"type": "content_chunk", "data": ""}
#             return

#         context_str = "".join([f"--- CONTEXT {i+1}: From file '{os.path.basename(doc.metadata.get('source', 'Unknown'))}' ---\n{doc.page_content}\n\n" for i, doc in enumerate(retrieved_docs)])
#         yield {"type": "context", "data": context_str}
        
#         llm_stream = self.synthesis_chain.stream({"context": context_str, "question": query})
#         for chunk in llm_stream:
#             yield {"type": "content_chunk", "data": chunk.content}

#     def improve_single_cell(self, original_mag_content: str, improvement_request: str):
#         response = self.improvement_chain.invoke({"original_mag": original_mag_content, "improvement_request": improvement_request})
#         new_mag_content = response.content
#         dependencies = set(re.findall(r"^\s*use\s+([\w\d_]+)", new_mag_content, re.MULTILINE))
#         return {"content": new_mag_content, "dependencies": dependencies}

# # ==============================================================================
# # STREAMLIT APPLICATION UI
# # ==============================================================================
# st.set_page_config(layout="wide", page_title="Interactive Chip Layout Designer")
# try:
#     retriever = get_retriever(MAG_FILES_PATH)
#     generator = MagicLayoutGenerator(retriever)
# except Exception as e:
#     st.error(f"💥 **Initialization Error:** {e}"); st.stop()

# if "generation_queue" not in st.session_state: st.session_state.generation_queue = []
# if "completed_cells" not in st.session_state: st.session_state.completed_cells = {}
# if "current_cell_data" not in st.session_state: st.session_state.current_cell_data = None
# if "mode" not in st.session_state: st.session_state.mode = "Automatic"

# st.title("🤖 Interactive Chip Layout Designer")
# st.write("An AI tool for generating, visualizing, and iteratively refining VLSI layouts.")

# with st.container(border=True):
#     st.subheader("⚙️ Control Panel")
#     st.session_state.mode = st.radio("**Select Mode**", ["Automatic", "Strict Review"], horizontal=True, help="**Automatic**: Generate all components at once. **Strict Review**: Pause to review and improve each component.")
#     with st.form(key='design_form'):
#         query = st.text_input("**Design Prompt**", "a 2-input NAND gate")
#         filename = st.text_input("**Top-level Filename**", "my_nand.mag")
#         if st.form_submit_button(label="🚀 Start New Generation", use_container_width=True):
#             if query and filename:
#                 st.session_state.generation_queue = [(query, os.path.splitext(filename)[0])]
#                 st.session_state.completed_cells = {}
#                 st.session_state.current_cell_data = None
#                 os.makedirs(GENERATED_MAG_DIR, exist_ok=True)
#             else: st.error("Please provide both a prompt and a filename.")
# st.divider()

# if st.session_state.generation_queue:
#     if st.session_state.current_cell_data is None:
#         current_query, current_cell_name = st.session_state.generation_queue[0]
#         st.header(f"Processing: `{current_cell_name}`")
#         st.info("Live generation in progress... Code and visualization will appear side-by-side.", icon="⚡")

#         col1, col2 = st.columns(2)
#         with col1: st.subheader("📄 Live Generated Code"); code_area = st.empty()
#         with col2: st.subheader("🖼️ Live Visualization"); plot_area = st.empty()
#         context_area = st.empty()

#         fig, ax = plt.subplots(figsize=(15, 12))
#         ax.set_aspect('equal', adjustable='box'); ax.set_title("Live Layout Generation", fontsize=18); ax.grid(True, linestyle='--', alpha=0.6)
#         plot_area.pyplot(fig)

#         full_mag_content, line_buffer, current_layer = "", "", None
#         # --- FIX: Moved definition to two lines ---
#         layer_colors = {}
#         def get_random_color(): return (random.random(), random.random(), random.random())

#         response_stream = generator.stream_single_cell(current_query)
#         for response in response_stream:
#             if response["type"] == "context":
#                 context_area.expander("View AI Context Used for this Generation").text(response["data"])
#             elif response["type"] == "content_chunk":
#                 chunk = response["data"]
#                 full_mag_content += chunk
#                 line_buffer += chunk
#                 code_area.code(full_mag_content, language='text')
#                 if '\n' in line_buffer:
#                     lines, line_buffer = line_buffer.rsplit('\n', 1)
#                     for line in lines.split('\n'):
#                         line = line.strip()
#                         parts = line.split()
#                         if line.startswith("<<"):
#                             layer_name = line.strip("<<>> ").strip()
#                             if layer_name != "end" and layer_name not in layer_colors:
#                                 current_layer = layer_name
#                                 layer_colors[current_layer] = get_random_color()
#                                 ax.legend(handles=[patches.Patch(color=c, label=n, alpha=0.7) for n, c in layer_colors.items()], loc='upper right')
#                         elif parts and parts[0] == "rect" and len(parts) == 5 and current_layer:
#                             try:
#                                 x1, y1, x2, y2 = map(int, parts[1:5])
#                                 width, height = abs(x2 - x1), abs(y2 - y1)
#                                 x_start, y_start = min(x1, x2), min(y1, y2)
#                                 ax.add_patch(patches.Rectangle((x_start, y_start), width, height, linewidth=1.5, edgecolor='black', facecolor=layer_colors[current_layer], alpha=0.75))
#                                 ax.relim(); ax.autoscale_view()
#                                 plot_area.pyplot(fig)
#                             except (ValueError, IndexError): continue
#         plt.close(fig)
#         st.session_state.current_cell_data = {"name": current_cell_name, "content": full_mag_content}
#         st.rerun()

#     else:
#         data = st.session_state.current_cell_data
#         cell_name, mag_content = data['name'], data['content']
#         st.header(f"Reviewing: `{cell_name}`")
#         file_path = os.path.join(GENERATED_MAG_DIR, f"{cell_name}.mag")
#         with open(file_path, "w") as f: f.write(mag_content)
        
#         if "use " in mag_content:
#             st.info("Hierarchical design detected. Displaying final static layout.", icon="🏗️")
#             fig = visualize_hierarchical_layout(file_path)
#         else:
#             st.info("Flat design detected. Displaying final layout.", icon="🎬")
#             fig = visualize_hierarchical_layout(file_path)
#         if fig: st.pyplot(fig); plt.close(fig)

#         dependencies = set(re.findall(r"^\s*use\s+([\w\d_]+)", mag_content, re.MULTILINE))

#         if st.session_state.mode == "Strict Review":
#             st.info("The generation is paused. Please approve the component or request an improvement to continue.", icon="✋")
#             with st.container(border=True):
#                 with st.form("review_form"):
#                     st.subheader("🔬 Review Component")
#                     improvement_prompt = st.text_area("Improvement Request (optional)", placeholder="e.g., Make the routing more compact.")
#                     approve_button = st.form_submit_button("✅ Looks Good, Continue", use_container_width=True)
#                     improve_button = st.form_submit_button("🔄 Improve This Component", use_container_width=True)
                
#                 if approve_button:
#                     st.session_state.completed_cells[cell_name] = mag_content
#                     st.session_state.generation_queue.pop(0)
#                     for dep in dependencies:
#                         if dep not in st.session_state.completed_cells and all(dep != item[1] for item in st.session_state.generation_queue):
#                             st.session_state.generation_queue.append((f"a {dep} layout", dep))
#                     st.session_state.current_cell_data = None
#                     st.rerun()
                
#                 if improve_button and improvement_prompt:
#                     with st.spinner(f"AI is improving '{cell_name}'..."):
#                         improved_result = generator.improve_single_cell(mag_content, improvement_prompt)
#                         st.session_state.current_cell_data['content'] = improved_result['content']
#                     st.rerun()
#         else:
#             st.session_state.completed_cells[cell_name] = mag_content
#             st.session_state.generation_queue.pop(0)
#             for dep in dependencies:
#                 if dep not in st.session_state.completed_cells and all(dep != item[1] for item in st.session_state.generation_queue):
#                     st.session_state.generation_queue.append((f"a {dep} layout", dep))
#             st.session_state.current_cell_data = None
#             st.success(f"Automatically approved '{cell_name}'. Continuing...")
#             time.sleep(1)
#             st.rerun()

# elif st.session_state.completed_cells:
#     st.balloons(); st.header("🎉 Generation Complete!"); st.write("All components have been successfully generated.")

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
MAG_FILES_PATH = 'std_cells_datasets/'

# ==============================================================================
# VISUALIZATION LOGIC
# ==============================================================================
_parsed_cell_cache = {}

def parse_mag_data_hierarchical(file_path, current_dir=None):
    abs_file_path = os.path.abspath(file_path)
    if abs_file_path in _parsed_cell_cache:
        return _parsed_cell_cache[abs_file_path]
    if current_dir is None: current_dir = os.path.dirname(abs_file_path)
    full_file_path = os.path.join(current_dir, os.path.basename(file_path))
    try:
        with open(full_file_path, 'r') as file: mag_content = file.read()
    except FileNotFoundError:
        st.warning(f"Sub-cell file not found: '{os.path.basename(full_file_path)}'. Instance will be skipped.")
        return None
    except Exception as e:
        st.error(f"Error reading file '{full_file_path}': {e}"); return None

    parsed_data = {"header": {}, "layers": {}, "instances": []}
    current_layer, current_instance = None, None
    for line in mag_content.strip().split('\n'):
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
        elif command == "rect" and len(parts) == 5 and current_layer:
            try: parsed_data["layers"][current_layer]["rects"].append({"x1": int(parts[1]), "y1": int(parts[2]), "x2": int(parts[3]), "y2": int(parts[4])})
            except (ValueError, IndexError): pass
        elif command == "use":
            if current_instance: parsed_data["instances"].append(current_instance)
            if len(parts) >= 3:
                sub_file_path = os.path.join(os.path.dirname(full_file_path), f"{parts[1]}.mag")
                current_instance = {"cell_type": parts[1], "instance_name": parts[2], "parsed_content": parse_mag_data_hierarchical(sub_file_path, os.path.dirname(full_file_path)),"transform": [1, 0, 0, 0, 1, 0], "box": [0, 0, 0, 0]}
                if not current_instance["parsed_content"]: current_instance = None
        elif command == "transform" and current_instance:
            try: current_instance["transform"] = [int(v) for v in parts[1:]]
            except (ValueError, IndexError): pass
        elif command == "box" and current_instance:
            try: current_instance["box"] = [int(v) for v in parts[1:]]
            except (ValueError, IndexError): pass
        elif line == "<< end >>" and current_instance:
            parsed_data["instances"].append(current_instance)
            current_instance = None
    if current_instance: parsed_data["instances"].append(current_instance)
    _parsed_cell_cache[abs_file_path] = parsed_data
    return parsed_data

def visualize_hierarchical_layout(file_path: str):
    _parsed_cell_cache.clear()
    parsed_data = parse_mag_data_hierarchical(file_path)
    if not parsed_data: return None
    fig, ax = plt.subplots(figsize=(15, 12))
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
                tx1, ty1 = _apply_transform(rect["x1"], rect["y1"], current_transform)
                tx2, ty2 = _apply_transform(rect["x2"], rect["y2"], current_transform)
                width, height = abs(tx2 - tx1), abs(ty2 - ty1)
                x_start, y_start = min(tx1, tx2), min(ty1, ty2)
                min_x, max_x = min(min_x, x_start), max(max_x, x_start + width)
                min_y, max_y = min(min_y, y_start), max(max_y, y_start + height)
                ax.add_patch(patches.Rectangle((x_start, y_start), width, height, linewidth=1, edgecolor='black', facecolor=color, alpha=0.7))
        for instance in data_to_draw.get("instances", []):
            if instance.get("parsed_content"): _draw_elements(instance["parsed_content"], instance["transform"])
    _draw_elements(parsed_data)
    if not all(v != float('inf') and v != float('-inf') for v in [min_x, max_x, min_y, max_y]):
        plt.close(fig); return None
    padding = (max_x - min_x) * 0.1 if (max_x > min_x) else 10
    ax.set_xlim(min_x - padding, max_x + padding)
    ax.set_ylim(min_y - padding, max_y + padding)
    ax.set_aspect('equal', adjustable='box'); ax.set_title(f"Hierarchical Layout: {os.path.basename(file_path)}", fontsize=16); ax.grid(True, linestyle='--', alpha=0.6)
    return fig

# ==============================================================================
# DATA LOADING & AI GENERATION
# ==============================================================================
@st.cache_resource(show_spinner="Initializing Vector Store...")
def get_retriever(mag_files_directory: str):
    load_dotenv(); api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key: st.error("GOOGLE_API_KEY not found."); st.stop()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    def _preprocess_mag_file(file_path: str):
        with open(file_path, 'r') as f: content = f.read()
        cell_name = os.path.splitext(os.path.basename(file_path))[0]
        return Document(page_content=f"Magic layout for standard cell. cell name is {cell_name}. function is {cell_name}. " + content, metadata={"source": file_path})
    if os.path.exists(FAISS_INDEX_PATH):
        vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        if not os.path.exists(mag_files_directory): st.error(f"Dataset directory not found: '{mag_files_directory}'."); st.stop()
        mag_files = glob.glob(os.path.join(mag_files_directory, '**', '*.mag'), recursive=True)
        if not mag_files: st.error(f"No .mag files found in '{mag_files_directory}'."); st.stop()
        documents = [_preprocess_mag_file(file_path) for file_path in mag_files]
        vector_store = FAISS.from_documents(documents, embeddings)
        vector_store.save_local(FAISS_INDEX_PATH)
    return vector_store.as_retriever(search_kwargs={"k": 3})

class MagicLayoutGenerator:
    def __init__(self, retriever):
        load_dotenv(); api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key: raise ValueError("GOOGLE_API_KEY not found.")
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=api_key, temperature=0.1)
        self.retriever = retriever
        self.synthesis_chain = PromptTemplate.from_template("CONTEXTS:\n{context}\n\nQUESTION:\n{question}\n\nBased on the contexts, generate a .mag file for the question. The response must be ONLY the raw .mag file content, starting with 'magic' and ending with '<< end >>'.") | self.llm
        self.improvement_chain = PromptTemplate.from_template("ORIGINAL .MAG FILE:\n{original_mag}\n\nUSER'S IMPROVEMENT REQUEST:\n{improvement_request}\n\nRegenerate the .mag file to incorporate the request. The output MUST be a complete, valid .mag file.") | self.llm

    def stream_single_cell(self, query: str):
        """Yields context first, then streams the LLM response for the .mag file."""
        retrieved_docs = self.retriever.invoke(query)
        if not retrieved_docs:
            yield {"type": "context", "data": "No relevant contexts found."}
            yield {"type": "content_chunk", "data": ""}
            return

        context_str = "".join([f"--- CONTEXT {i+1}: From file '{os.path.basename(doc.metadata.get('source', 'Unknown'))}' ---\n{doc.page_content}\n\n" for i, doc in enumerate(retrieved_docs)])
        yield {"type": "context", "data": context_str}
        
        llm_stream = self.synthesis_chain.stream({"context": context_str, "question": query})
        for chunk in llm_stream:
            yield {"type": "content_chunk", "data": chunk.content}

    def improve_single_cell(self, original_mag_content: str, improvement_request: str):
        response = self.improvement_chain.invoke({"original_mag": original_mag_content, "improvement_request": improvement_request})
        new_mag_content = response.content
        dependencies = set(re.findall(r"^\s*use\s+([\w\d_]+)", new_mag_content, re.MULTILINE))
        return {"content": new_mag_content, "dependencies": dependencies}

# ==============================================================================
# STREAMLIT APPLICATION UI
# ==============================================================================
st.set_page_config(layout="wide", page_title="Interactive Chip Layout Designer")
try:
    retriever = get_retriever(MAG_FILES_PATH)
    generator = MagicLayoutGenerator(retriever)
except Exception as e:
    st.error(f"💥 **Initialization Error:** {e}"); st.stop()

if "generation_queue" not in st.session_state: st.session_state.generation_queue = []
if "completed_cells" not in st.session_state: st.session_state.completed_cells = {}
if "current_cell_data" not in st.session_state: st.session_state.current_cell_data = None
if "mode" not in st.session_state: st.session_state.mode = "Automatic"

st.title("🤖 Interactive Chip Layout Designer")
st.write("An AI tool for generating, visualizing, and iteratively refining VLSI layouts.")

with st.container(border=True):
    st.subheader("⚙️ Control Panel")
    st.session_state.mode = st.radio("**Select Mode**", ["Automatic", "Strict Review"], horizontal=True, help="**Automatic**: Generate all components at once. **Strict Review**: Pause to review and improve each component.")
    with st.form(key='design_form'):
        query = st.text_input("**Design Prompt**", "a 2-input NAND gate")
        filename = st.text_input("**Top-level Filename**", "my_nand.mag")
        if st.form_submit_button(label="🚀 Start New Generation", use_container_width=True):
            if query and filename:
                st.session_state.generation_queue = [(query, os.path.splitext(filename)[0])]
                st.session_state.completed_cells = {}
                st.session_state.current_cell_data = None
                os.makedirs(GENERATED_MAG_DIR, exist_ok=True)
            else: st.error("Please provide both a prompt and a filename.")
st.divider()

if st.session_state.generation_queue:
    # This block handles the two-stage process for a cell:
    # 1. Live-streaming generation
    # 2. Review and approval
    
    # Check if we need to start a new live generation
    if st.session_state.current_cell_data is None:
        current_query, current_cell_name = st.session_state.generation_queue[0]
        st.header(f"Processing: `{current_cell_name}`")
        st.info("Live generation in progress...", icon="⚡")

        col1, col2 = st.columns(2)
        with col1: st.subheader("📄 Live Generated Code"); code_area = st.empty()
        with col2: st.subheader("🖼️ Live Visualization"); plot_area = st.empty()
        context_area = st.empty()

        fig, ax = plt.subplots(figsize=(15, 12))
        ax.set_aspect('equal', adjustable='box'); ax.set_title("Live Layout Generation", fontsize=18); ax.grid(True, linestyle='--', alpha=0.6)
        plot_area.pyplot(fig)

        full_mag_content, line_buffer, current_layer = "", "", None
        layer_colors = {}
        def get_random_color(): return (random.random(), random.random(), random.random())

        response_stream = generator.stream_single_cell(current_query)
        for response in response_stream:
            if response["type"] == "context":
                context_area.expander("View AI Context Used for this Generation").text(response["data"])
            elif response["type"] == "content_chunk":
                chunk = response["data"]
                full_mag_content += chunk
                line_buffer += chunk
                code_area.code(full_mag_content, language='text')
                if '\n' in line_buffer:
                    lines, line_buffer = line_buffer.rsplit('\n', 1)
                    for line in lines.split('\n'):
                        line = line.strip()
                        parts = line.split()
                        if line.startswith("<<"):
                            layer_name = line.strip("<<>> ").strip()
                            if layer_name != "end" and layer_name not in layer_colors:
                                current_layer = layer_name
                                layer_colors[current_layer] = get_random_color()
                                ax.legend(handles=[patches.Patch(color=c, label=n, alpha=0.7) for n, c in layer_colors.items()], loc='upper right')
                        elif parts and parts[0] == "rect" and len(parts) == 5 and current_layer:
                            try:
                                x1, y1, x2, y2 = map(int, parts[1:5])
                                width, height = abs(x2 - x1), abs(y2 - y1)
                                x_start, y_start = min(x1, x2), min(y1, y2)
                                ax.add_patch(patches.Rectangle((x_start, y_start), width, height, linewidth=1.5, edgecolor='black', facecolor=layer_colors[current_layer], alpha=0.75))
                                ax.relim(); ax.autoscale_view()
                                plot_area.pyplot(fig)
                            except (ValueError, IndexError): continue
        
        st.session_state.current_cell_data = {"name": current_cell_name, "content": full_mag_content}
        plt.close(fig)

    # This block now runs immediately after the above block finishes, without a refresh.
    if st.session_state.current_cell_data:
        data = st.session_state.current_cell_data
        cell_name, mag_content = data['name'], data['content']
        
        file_path = os.path.join(GENERATED_MAG_DIR, f"{cell_name}.mag")
        with open(file_path, "w") as f: f.write(mag_content)

        dependencies = set(re.findall(r"^\s*use\s+([\w\d_]+)", mag_content, re.MULTILINE))

        if st.session_state.mode == "Strict Review":
            st.info("Generation complete. Please review the final layout below.", icon="✅")
            st.subheader("🔬 Review Component")
            with st.container(border=True):
                with st.form("review_form"):
                    improvement_prompt = st.text_area("Improvement Request (optional)", placeholder="e.g., Make the routing more compact.")
                    approve_button = st.form_submit_button("👍 Looks Good, Continue", use_container_width=True)
                    improve_button = st.form_submit_button("💡 Improve This Component", use_container_width=True)
                
                if approve_button:
                    st.session_state.completed_cells[cell_name] = mag_content
                    st.session_state.generation_queue.pop(0)
                    for dep in dependencies:
                        if dep not in st.session_state.completed_cells and all(dep != item[1] for item in st.session_state.generation_queue):
                            st.session_state.generation_queue.append((f"a {dep} layout", dep))
                    st.session_state.current_cell_data = None
                    st.rerun()
                
                if improve_button and improvement_prompt:
                    with st.spinner(f"AI is improving '{cell_name}'..."):
                        improved_result = generator.improve_single_cell(mag_content, improvement_prompt)
                        st.session_state.current_cell_data = {
                            "name": cell_name,
                            "content": improved_result['content']
                        }
                    st.rerun()
        else: # Automatic Mode
            st.session_state.completed_cells[cell_name] = mag_content
            st.session_state.generation_queue.pop(0)
            for dep in dependencies:
                # --- This is the start of the completed code ---
                if dep not in st.session_state.completed_cells and all(dep != item[1] for item in st.session_state.generation_queue):
                    st.session_state.generation_queue.append((f"a {dep} layout", dep))
            st.session_state.current_cell_data = None
            st.success(f"Automatically approved '{cell_name}'. Continuing...")
            time.sleep(1)
            st.rerun()

elif st.session_state.completed_cells:
    st.balloons(); st.header("🎉 Generation Complete!"); st.write("All components have been successfully generated.")