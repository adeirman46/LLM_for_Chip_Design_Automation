import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import subprocess
import os
import json
import shutil
import re
from IPython.display import SVG
from sootty import WireTrace, Visualizer, Style
import threading
import uuid

# --- Page Configuration ---
st.set_page_config(
    page_title="LLM Chip Design Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 LLM-Powered Chip Design Automation Flow")
st.write("A multi-module workflow to generate, simulate, and synthesize complex digital designs.")

# --- Constants ---
MODEL_NAME = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
HOME_DIR = os.path.expanduser("~")
OPENLANE_DIR = os.path.join(HOME_DIR, "OpenLane")
OPENLANE_IMAGE = "efabless/openlane:e73fb3c57e687a0023fcd4dcfd1566ecd478362a-amd64"
PDK_ROOT = os.path.join(HOME_DIR, ".volare")

# --- Session State Initialization ---
if 'model_lock' not in st.session_state:
    st.session_state.model_lock = threading.Lock()

# Initialize all session state keys to prevent errors on first run
defaults = {
    'model_loaded': False,
    'tokenizer': None,
    'model': None,
    'designs': [],  # List to hold all design modules
    'active_design_index': None, # Index of the module being worked on
    # --- UI state for the "new module" form ---
    'new_module_name': "my_module",
    'new_module_desc': "A simple module that passes input to output.",
    'new_module_ports': [],
    'new_module_params': [],
    'new_module_is_toplevel': False,
    'new_module_submodules': [], # For hierarchical design
    'form_reset_flag': False,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- Safe Form Reset Logic ---
# This runs at the start of a script run, before widgets are rendered.
if st.session_state.form_reset_flag:
    st.session_state.new_module_name = "my_module"
    st.session_state.new_module_desc = "A simple module that passes input to output."
    st.session_state.new_module_ports = []
    st.session_state.new_module_params = []
    st.session_state.new_module_is_toplevel = False
    st.session_state.new_module_submodules = []
    st.session_state.form_reset_flag = False # Reset the flag

# --- Helper Functions ---

def get_port_range(bits_val):
    """Helper to format the Verilog port range string."""
    bits_val_str = str(bits_val).strip()
    # Return empty string for 1-bit ports so they don't have a range
    if bits_val_str.isdigit() and int(bits_val_str) <= 1:
        return ""
    # Return calculated range for multi-bit numeric ports
    elif bits_val_str.isdigit():
        return f"[{int(bits_val_str)-1}:0]"
    # Return range based on parameter name
    else:
        return f"[{bits_val_str}-1:0]"

def extract_verilog_code(raw_output):
    """
    Extracts Verilog code cleanly, looking for markdown-style code blocks
    or falling back to the first 'module' and the last 'endmodule'.
    """
    verilog_match = re.search(r'```verilog(.*?)```', raw_output, re.DOTALL)
    if verilog_match:
        code = verilog_match.group(1).strip()
    else:
        code = raw_output

    try:
        start_index = code.index("module")
        end_index = code.rindex("endmodule") + len("endmodule")
        return code[start_index:end_index].strip()
    except ValueError:
        st.warning("Could not find 'module' and 'endmodule' keywords in the LLM output. Displaying raw output.")
        return raw_output.strip()

def load_model():
    """
    Loads the Hugging Face model and tokenizer. This function is designed to be
    called once and its results stored in the session state.
    """
    with st.session_state.model_lock:
        if not st.session_state.model_loaded:
            st.info(f"Loading model '{MODEL_NAME}'... This may take a few minutes and significant memory.")
            try:
                tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
                quantization_config = BitsAndBytesConfig(load_in_4bit=True)
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    device_map="auto",
                    quantization_config=quantization_config,
                    torch_dtype=torch.float16,
                )
                st.session_state.tokenizer = tokenizer
                st.session_state.model = model
                st.session_state.model_loaded = True
                st.success("✅ Model and Tokenizer loaded successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error loading model: {e}")
                st.error("This likely means your system is out of memory or you don't have a compatible GPU.")
                st.stop()

def generate_code_locally(prompt):
    """Generates code using the loaded local model from session state."""
    if not st.session_state.model_loaded:
        st.error("Model not loaded. Cannot generate code.")
        return None

    tokenizer = st.session_state.tokenizer
    model = st.session_state.model
    try:
        messages = [
            {"role": "system", "content": "You are an expert Verilog designer. Generate clean, correct, and complete Verilog code based on the user's request. Only output the Verilog code itself, without any explanations or markdown formatting."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=8192, pad_token_id=tokenizer.eos_token_id)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_start = generated_text.rfind("assistant")
        if response_start != -1:
            return generated_text[response_start + len("assistant"):].strip()
        return generated_text
    except Exception as e:
        st.error(f"Error during model inference: {e}")
        return None

# --- UI Component Callbacks ---
def add_port():
    name = st.session_state.get("new_port_name", "").strip()
    bits = st.session_state.get("new_port_bits", "1").strip()
    if name and bits:
        st.session_state.new_module_ports.append({
            "id": str(uuid.uuid4()),
            "direction": st.session_state.new_port_dir,
            "type": st.session_state.new_port_type,
            "bits": bits, # Store as a string
            "name": name.replace(" ", "_")
        })
        st.session_state.new_port_name = "" # Clear input
    else:
        st.warning("Port Name and Bits/Param cannot be empty.")

def remove_port(port_id):
    st.session_state.new_module_ports = [p for p in st.session_state.new_module_ports if p['id'] != port_id]

def add_param():
    name = st.session_state.get("new_param_name", "").strip()
    if name:
        st.session_state.new_module_params.append({
            "id": str(uuid.uuid4()),
            "name": name.replace(" ", "_"),
            "value": st.session_state.new_param_value
        })
        st.session_state.new_param_name = "" # Clear input
    else:
        st.warning("Parameter name cannot be empty.")

def remove_param(param_id):
    st.session_state.new_module_params = [p for p in st.session_state.new_module_params if p['id'] != param_id]

def select_design(index):
    st.session_state.active_design_index = index

# --- Main Application UI ---

if not st.session_state.model_loaded:
    st.write("Welcome! This tool uses a local LLM to help you design, test, and synthesize digital hardware.")
    with st.spinner("Initializing... The model will be downloaded and loaded into memory. This may take a while."):
        load_model()
    st.stop()

# --- Main Layout: Sidebar for navigation, Main area for content ---
st.sidebar.header("Design Units")
st.sidebar.write("Manage your Verilog modules here.")

if not st.session_state.designs:
    st.sidebar.info("No modules generated yet. Use the form in the main area to create one.")
else:
    for i, design in enumerate(st.session_state.designs):
        is_active = (st.session_state.active_design_index == i)
        label = f"» {design['name']}" if is_active else design['name']
        if design.get('is_toplevel', False):
            label += " (Top)"
        st.sidebar.button(label, key=f"select_design_{i}", on_click=select_design, args=(i,), use_container_width=True)

# --- Main Content Area ---
col1, col2 = st.columns([1, 1])

# --- Column 1: Module Generation ---
with col1:
    st.header("1. Create a New Module")
    with st.expander("Expand to create a new Verilog module", expanded=True):
        st.text_input("Module Name", key="new_module_name")
        st.checkbox("Set as Top-Level Module for Synthesis", key="new_module_is_toplevel")
        
        # HIERARCHY SUPPORT: Multi-select box for sub-modules
        available_modules = [d['name'] for d in st.session_state.designs]
        st.multiselect(
            "Instantiate existing modules (for hierarchical design):",
            options=available_modules,
            key="new_module_submodules"
        )
        
        st.text_area("Module Description", key="new_module_desc",
                     help="Describe the desired functionality. The LLM will use this to generate the internal logic.")

        # --- Parameters ---
        st.subheader("A. Define Parameters")
        param_cols = st.columns([2, 1, 1])
        param_cols[0].text_input("Parameter Name", key="new_param_name")
        param_cols[1].text_input("Default Value", "32", key="new_param_value")
        param_cols[2].button("➕ Add Param", on_click=add_param, use_container_width=True)

        if st.session_state.new_module_params:
            st.write("**Current Parameters:**")
            for p in st.session_state.new_module_params:
                p_col1, p_col2 = st.columns([5,1])
                p_col1.markdown(f"- `parameter` **{p['name']}** = `{p['value']}`")
                p_col2.button("🗑️", key=f"del_param_{p['id']}", on_click=remove_param, args=(p['id'],), help="Remove parameter")

        # --- Ports ---
        st.subheader("B. Define Ports")
        port_cols = st.columns([1, 1, 1, 2, 1])
        port_cols[0].selectbox("Direction", ["input", "output"], key="new_port_dir")
        port_cols[1].selectbox("Type", ["wire", "reg"], key="new_port_type")
        port_cols[2].text_input("Bits/Param", "1", key="new_port_bits", help="Enter a number (e.g., 8) or a parameter name (e.g., DATA_WIDTH).")
        port_cols[3].text_input("Port Name", key="new_port_name")
        port_cols[4].button("➕ Add Port", on_click=add_port, use_container_width=True)

        if st.session_state.new_module_ports:
            st.write("**Current Ports:**")
            for p in st.session_state.new_module_ports:
                p_col1, p_col2 = st.columns([5,1])
                range_str = get_port_range(p['bits'])
                range_display = f" `{range_str}`" if range_str else ""
                p_col1.markdown(f"- `{p['direction']}` `{p['type']}`{range_display} **{p['name']}**")
                p_col2.button("🗑️", key=f"del_port_{p['id']}", on_click=remove_port, args=(p['id'],), help="Remove port")

        st.divider()

        # --- Generation Button ---
        if st.button("🚀 Generate Verilog Module", type="primary", use_container_width=True):
            if not st.session_state.new_module_name.strip():
                st.error("Module name is required.")
            else:
                with st.spinner("Generating Verilog with LLM..."):
                    # --- Build Hierarchical Context ---
                    submodule_context = ""
                    if st.session_state.new_module_submodules:
                        submodule_context += "This module must instantiate and connect the following sub-modules, whose Verilog code is provided below:\n"
                        for sub_name in st.session_state.new_module_submodules:
                            sub_design = next((d for d in st.session_state.designs if d['name'] == sub_name), None)
                            if sub_design:
                                submodule_context += f"\n--- Verilog for {sub_name} ---\n"
                                submodule_context += f"```verilog\n{sub_design['code']}\n```\n"
                        submodule_context += "\n"
                    
                    # --- Build Parameter and Port Definitions ---
                    param_defs_str = ""
                    if st.session_state.new_module_params:
                        params = [f"parameter {p['name']} = {p['value']}" for p in st.session_state.new_module_params]
                        param_defs_str = "#(\n    " + ",\n    ".join(params) + "\n)"

                    port_defs = ",\n".join([f"    {p['direction']} {'' if p['direction'] == 'input' else p['type']} {get_port_range(p['bits'])} {p['name']}" for p in st.session_state.new_module_ports])

                    # --- Assemble Final Prompt ---
                    prompt = (
                        f"Generate a complete Verilog module named '{st.session_state.new_module_name}'.\n\n"
                        f"{submodule_context}"
                        f"Functional Description: {st.session_state.new_module_desc}\n\n"
                        f"If the module has parameters, define them using the `# (parameter ...)` syntax after the module name and before the port list. "
                        f"The module's parameters and ports are listed below.\n\n"
                        f"Module Definition:\n"
                        f"module {st.session_state.new_module_name} {param_defs_str}\n"
                        f"(\n{port_defs}\n);\n\n"
                        f"Based on all the information above, provide the complete Verilog code for the module, including the internal logic. "
                        f"Start with `module` and end with `endmodule`."
                    )


                    raw_code = generate_code_locally(prompt)
                    if raw_code:
                        new_design = {
                            "name": st.session_state.new_module_name.strip().replace(" ", "_"),
                            "code": extract_verilog_code(raw_code),
                            "testbench": "",
                            "is_toplevel": st.session_state.new_module_is_toplevel,
                            "vcd_path": None,
                            "sim_output": ""
                        }
                        if new_design['is_toplevel']:
                            for d in st.session_state.designs:
                                d['is_toplevel'] = False
                        st.session_state.designs.append(new_design)
                        st.session_state.active_design_index = len(st.session_state.designs) - 1
                        
                        st.success(f"Module '{new_design['name']}' generated!")
                        
                        # Set the flag to reset the form on the next run
                        st.session_state.form_reset_flag = True
                        st.rerun()
                    else:
                        st.error("Failed to generate Verilog code.")

# --- Column 2: Active Module Editor and Actions ---
with col2:
    if st.session_state.active_design_index is not None:
        idx = st.session_state.active_design_index
        active_design = st.session_state.designs[idx]

        st.header(f"Workspace: {active_design['name']}")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📝 Verilog Code", "🔬 Testbench", "🏁 Simulation", "🛠️ Synthesis"])

        with tab1:
            st.subheader("Verilog RTL Code")
            active_design['code'] = st.text_area("RTL Code", value=active_design['code'], height=400, key=f"code_{idx}")

        with tab2:
            st.subheader("Verilog Testbench")
            if st.button("🤖 Generate Testbench with LLM", key=f"gen_tb_{idx}"):
                with st.spinner("Generating testbench..."):
                    prompt = f"""Generate a comprehensive Verilog testbench for the module '{active_design['name']}'.
Instantiate the module, connect all ports, and provide meaningful stimulus to test its functionality.
If the module has parameters, declare them locally in the testbench (e.g., `parameter WIDTH = 32;`) and use them in the instantiation.
Include `timescale 1ns/100ps`, `$dumpfile`, `$dumpvars`, and `$finish`.

Module to be tested:
```verilog
{active_design['code']}
```
Provide only the Verilog code for the testbench."""
                    raw_tb = generate_code_locally(prompt)
                    if raw_tb:
                        active_design['testbench'] = extract_verilog_code(raw_tb)
                        st.success("Testbench generated!")
                        st.rerun()
                    else:
                        st.error("Failed to generate testbench.")
            
            active_design['testbench'] = st.text_area("Testbench Code", value=active_design['testbench'], height=400, key=f"tb_{idx}")

        with tab3:
            st.subheader("RTL Simulation")
            if st.button("🚦 Run Simulation", key=f"run_sim_{idx}", type="primary"):
                active_design['sim_output'] = ""
                active_design['vcd_path'] = None
                with st.spinner("Running Icarus Verilog simulation..."):
                    design_file = f"{active_design['name']}.v"
                    tb_file = f"{active_design['name']}_tb.v"
                    with open(design_file, "w") as f: f.write(active_design['code'])
                    with open(tb_file, "w") as f: f.write(active_design['testbench'])

                    output_file = f"{active_design['name']}_sim"
                    vcd_file = f"{active_design['name']}.vcd"
                    
                    try:
                        compile_cmd = ["iverilog", "-o", output_file, design_file, tb_file]
                        compile_res = subprocess.run(compile_cmd, capture_output=True, text=True, check=True)
                        run_cmd = ["vvp", output_file]
                        run_res = subprocess.run(run_cmd, capture_output=True, text=True, check=True)
                        active_design['sim_output'] = f"Compilation:\n{compile_res.stdout}{compile_res.stderr}\n\nExecution:\n{run_res.stdout}{run_res.stderr}"
                        if os.path.exists(vcd_file):
                            st.success("Simulation successful! VCD file created.")
                            active_design['vcd_path'] = vcd_file
                        else:
                            st.error("Simulation ran, but VCD file was not found.")
                    except subprocess.CalledProcessError as e:
                        st.error("Error during simulation:")
                        active_design['sim_output'] = e.stderr
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")

            if active_design.get('sim_output'):
                st.code(active_design['sim_output'], language='log')
            
            if active_design.get('vcd_path'):
                st.subheader("Waveform Viewer")
                if st.button("📈 Show Waveform", key=f"show_wave_{idx}"):
                    with st.spinner("Generating waveform image..."):
                        try:
                            svg_file = os.path.splitext(active_design['vcd_path'])[0] + ".svg"
                            wiretrace = WireTrace.from_vcd(active_design['vcd_path'])
                            image = Visualizer(Style.Dark).to_svg(wiretrace)
                            with open(svg_file, "w") as f: f.write(str(image))
                            st.image(svg_file)
                        except Exception as e:
                            st.error(f"Failed to display waveform: {e}")

        with tab4:
            st.subheader("Synthesize with OpenLane")
            st.warning("Prerequisites: Docker, OpenLane, and the SKY130 PDK must be installed correctly.", icon="⚠️")
            
            top_level_design = next((d for d in st.session_state.designs if d.get('is_toplevel')), None)

            if not top_level_design:
                st.error("No top-level module designated. Please create a new module or edit an existing one to be the top-level design.")
            else:
                st.info(f"Top-level module for synthesis: **{top_level_design['name']}**")
                
                all_verilog_files = [f"dir::src/{d['name']}.v" for d in st.session_state.designs]
                
                default_config = {
                    "DESIGN_NAME": top_level_design['name'],
                    "VERILOG_FILES": all_verilog_files,
                    "CLOCK_PORT": "clk",
                    "CLOCK_PERIOD": 10.0,
                    "DESIGN_IS_CORE": False,
                    "FP_PDN_CORE_RING": False,
                    "RT_MAX_LAYER": "met4"
                }
                
                config_str = json.dumps(default_config, indent=4)
                edited_config_str = st.text_area("OpenLane Configuration (config.json)", value=config_str, height=250,
                                                 help="This config is auto-generated. Edit if needed.")

                if st.button("🛠️ Synthesize Chip", key=f"synth_{idx}", type="primary"):
                    try:
                        user_config = json.loads(edited_config_str)
                        design_name = user_config["DESIGN_NAME"]
                        design_dir = os.path.join(OPENLANE_DIR, "designs", design_name)
                        src_dir = os.path.join(design_dir, "src")
                        
                        if os.path.exists(design_dir):
                            shutil.rmtree(design_dir)
                        os.makedirs(src_dir, exist_ok=True)
                        
                        with open(os.path.join(design_dir, "config.json"), "w") as f: json.dump(user_config, f, indent=4)
                        
                        for d in st.session_state.designs:
                            with open(os.path.join(src_dir, f"{d['name']}.v"), "w") as f: f.write(d['code'])
                        
                        st.success(f"Successfully set up design '{design_name}' in '{design_dir}'.")

                        docker_command = [
                            'docker', 'run', '--rm',
                            '-v', f'{HOME_DIR}:{HOME_DIR}', '-v', f'{OPENLANE_DIR}:/openlane',
                            '-e', f'PDK_ROOT={PDK_ROOT}', '-e', 'PDK=sky130A',
                            '--user', f'{os.getuid()}:{os.getgid()}',
                            OPENLANE_IMAGE,
                            './flow.tcl', '-design', design_name
                        ]

                        st.info("Running OpenLane flow... This will take a long time. See output below.")
                        st.code(' '.join(docker_command))

                        log_placeholder = st.empty()
                        log_content = ""
                        process = subprocess.Popen(docker_command, cwd=OPENLANE_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                        
                        for line in iter(process.stdout.readline, ''):
                            log_content += line
                            log_placeholder.code(log_content, language='log')
                        
                        process.stdout.close()
                        return_code = process.wait()

                        if return_code == 0:
                            st.success(f"✅ OpenLane flow for {design_name} completed successfully!")
                            st.info(f"Find results in: {os.path.join(design_dir, 'runs')}")
                        else:
                            st.error(f"❌ OpenLane flow failed. Check logs for errors.")

                    except json.JSONDecodeError:
                        st.error("Invalid JSON in OpenLane configuration.")
                    except Exception as e:
                        st.error(f"An unexpected error occurred during synthesis setup: {e}")

    else:
        st.info("Select a design unit from the sidebar or create a new one to begin.")
