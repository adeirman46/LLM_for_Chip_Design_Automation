import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import subprocess
import os
import json
import shutil
import re
import uuid
from IPython.display import SVG
from sootty import WireTrace, Visualizer, Style
import threading
import difflib

# --- Page Configuration ---
st.set_page_config(
    page_title="LLM Chip Design Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 LLM-Powered Chip Design Automation Flow")
st.write("A multi-module workflow with closed-loop feedback to generate, simulate, and synthesize complex digital designs.")

# --- Constants ---
MODEL_NAME = "Qwen/Qwen2.5-7B"
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
    'designs': [],
    'active_design_index': None,
    'form_module_name': "my_module",
    'form_module_desc': "A simple module that passes input to output.",
    'form_module_ports': [],
    'form_module_params': [],
    'form_module_is_toplevel': False,
    'form_module_submodules': [],
    'form_testbench_goal': "Test the main functionality with valid inputs.",
    'form_test_vectors': [],
    # State for the correction workflow
    'show_correction_ui_sim': False,
    'show_correction_ui_synth': False,
    'correction_prompt': "",
    'suggested_code': "",
    'suggested_tb': "",
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- Helper Functions ---

def clear_module_form():
    st.session_state.active_design_index = None
    st.session_state.form_module_name = "my_module"
    st.session_state.form_module_desc = "A simple module that passes input to output."
    st.session_state.form_module_ports = []
    st.session_state.form_module_params = []
    st.session_state.form_module_is_toplevel = False
    st.session_state.form_module_submodules = []
    st.session_state.form_testbench_goal = "Test the main functionality with valid inputs."
    st.session_state.form_test_vectors = []
    st.rerun()

def load_design_into_form(index):
    st.session_state.active_design_index = index
    design = st.session_state.designs[index]
    st.session_state.form_module_name = design.get('name', 'my_module')
    st.session_state.form_module_desc = design.get('description', '')
    st.session_state.form_module_ports = [p.copy() for p in design.get('ports', [])]
    st.session_state.form_module_params = [p.copy() for p in design.get('params', [])]
    st.session_state.form_module_is_toplevel = design.get('is_toplevel', False)
    st.session_state.form_module_submodules = design.get('submodules', [])
    st.session_state.form_testbench_goal = design.get('testbench_goal', "Test the main functionality with valid inputs.")
    st.session_state.form_test_vectors = [v.copy() for v in design.get('test_vectors', [])]
    # Reset correction UI when switching designs
    st.session_state.show_correction_ui_sim = False
    st.session_state.show_correction_ui_synth = False
    st.rerun()

def remove_design(index):
    # If the removed design is the currently active one, clear the form.
    if st.session_state.active_design_index == index:
        clear_module_form()
    # Adjust the active index if an item before it was removed.
    elif st.session_state.active_design_index is not None and st.session_state.active_design_index > index:
        st.session_state.active_design_index -= 1
    
    st.session_state.designs.pop(index)
    st.rerun()

def get_port_range(bits_val):
    bits_val_str = str(bits_val).strip()
    if bits_val_str.isdigit() and int(bits_val_str) <= 1: return ""
    elif bits_val_str.isdigit(): return f"[{int(bits_val_str)-1}:0]"
    else: return f"[{bits_val_str}-1:0]"

def extract_verilog_code(raw_output, module_name=""):
    """Extracts Verilog code from the LLM's raw output, with robust fallbacks."""
    verilog_matches = re.findall(r'```verilog(.*?)```', raw_output, re.DOTALL)

    if not verilog_matches:
        if 'module' in raw_output and 'endmodule' in raw_output:
            st.warning("LLM did not use Markdown format. Extracting content between 'module' and 'endmodule'.")
            match = re.search(r'module.*?endmodule', raw_output, re.DOTALL)
            if match: return match.group(0).strip()
        return ""

    if len(verilog_matches) == 1:
        return verilog_matches[0].strip()

    if module_name:
        for code_block in verilog_matches:
            if re.search(rf'\bmodule\s+{re.escape(module_name)}\b', code_block):
                return code_block.strip()

    return max(verilog_matches, key=len).strip()


def load_model():
    with st.session_state.model_lock:
        if not st.session_state.model_loaded:
            st.info(f"Loading model '{MODEL_NAME}'...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
                quantization_config = BitsAndBytesConfig(load_in_4bit=True)
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME, device_map="auto", quantization_config=quantization_config, torch_dtype=torch.float16
                )
                st.session_state.tokenizer, st.session_state.model, st.session_state.model_loaded = tokenizer, model, True
                st.success("✅ Model and Tokenizer loaded successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error loading model: {e}"); st.stop()

def generate_code_locally(prompt, is_correction=False):
    if not st.session_state.model_loaded:
        st.error("Model not loaded."); return None
    tokenizer, model = st.session_state.tokenizer, st.session_state.model
    system_prompt = "You are an expert Verilog designer. Generate clean, correct, and complete Verilog code. Only output the Verilog code itself, inside ```verilog ... ``` blocks. Do not add any explanations."
    if is_correction:
         system_prompt = "You are an expert Verilog designer and debugger. The user will provide Verilog code and an error log. Analyze the error and provide a fully corrected version of the Verilog code for all specified modules. Ensure the corrected code is complete and syntax-perfect. Only output the corrected code inside ```verilog ... ``` blocks, without any explanations."
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=8192, pad_token_id=tokenizer.eos_token_id)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_start = generated_text.rfind("assistant")
        return generated_text[response_start + len("assistant"):].strip() if response_start != -1 else generated_text
    except Exception as e:
        st.error(f"Error during model inference: {e}"); return None

# --- UI Component Callbacks (for form elements) ---
def add_port():
    name, bits = st.session_state.get("new_port_name", "").strip(), st.session_state.get("new_port_bits", "1").strip()
    if name and bits:
        st.session_state.form_module_ports.append({
            "id": str(uuid.uuid4()), "direction": st.session_state.new_port_dir,
            "type": st.session_state.new_port_type, "bits": bits, "name": name.replace(" ", "_")
        }); st.session_state.new_port_name = ""
    else: st.warning("Port Name and Bits/Param cannot be empty.")

def remove_port(port_id):
    st.session_state.form_module_ports = [p for p in st.session_state.form_module_ports if p['id'] != port_id]

def add_param():
    name = st.session_state.get("new_param_name", "").strip()
    if name:
        st.session_state.form_module_params.append({
            "id": str(uuid.uuid4()), "name": name.replace(" ", "_"), "value": st.session_state.new_param_value
        }); st.session_state.new_param_name = ""
    else: st.warning("Parameter name cannot be empty.")

def remove_param(param_id):
    st.session_state.form_module_params = [p for p in st.session_state.form_module_params if p['id'] != param_id]

def add_test_vector():
    vector = st.session_state.get("new_test_vector", "").strip()
    if vector:
        st.session_state.form_test_vectors.append({"id": str(uuid.uuid4()), "assignments": vector})
        st.session_state.new_test_vector = "" # Clear input
    else:
        st.warning("Test vector cannot be empty.")

def remove_test_vector(vector_id):
    st.session_state.form_test_vectors = [v for v in st.session_state.form_test_vectors if v['id'] != vector_id]


# --- Main Application UI ---
if not st.session_state.model_loaded:
    with st.spinner("Initializing..."): load_model()
    st.stop()

# --- Sidebar ---
st.sidebar.header("Design Units")
st.sidebar.button("➕ New Module", on_click=clear_module_form, use_container_width=True)
st.sidebar.divider()
for i, design in enumerate(st.session_state.designs):
    col1, col2 = st.sidebar.columns([4, 1])
    label = f"{design['name']}" + (" (Top)" if design.get('is_toplevel') else "")
    col1.button(label, key=f"select_design_{i}", on_click=load_design_into_form, args=(i,), use_container_width=True, type="primary" if st.session_state.active_design_index == i else "secondary")
    col2.button("🗑️", key=f"del_design_{i}", on_click=remove_design, args=(i,), use_container_width=True, help=f"Delete {design['name']}")

# --- Main Content Area ---
col1, col2 = st.columns([1, 1])

# --- Column 1: Module Generation / Editor ---
with col1:
    is_editing = st.session_state.active_design_index is not None
    header_text = f"Editing: {st.session_state.form_module_name}" if is_editing else "1. Create a New Module"
    st.header(header_text)
    with st.expander("Expand to define the Verilog module", expanded=True):
        st.text_input("Module Name", key="form_module_name")
        st.checkbox("Set as Top-Level Module for Synthesis", key="form_module_is_toplevel")
        available_modules = [d['name'] for i, d in enumerate(st.session_state.designs) if i != st.session_state.active_design_index]
        st.multiselect("Instantiate existing modules:", options=available_modules, key="form_module_submodules")
        st.text_area("Module Description", key="form_module_desc", help="Describe the desired functionality.")

        st.subheader("A. Define Parameters")
        param_cols = st.columns([2, 1, 1]); param_cols[0].text_input("Parameter Name", "", key="new_param_name"); param_cols[1].text_input("Default Value", "32", key="new_param_value"); param_cols[2].button("➕ Add Param", on_click=add_param, use_container_width=True)
        for p in st.session_state.form_module_params:
            p_col1, p_col2 = st.columns([5,1]); p_col1.markdown(f"- `parameter` **{p['name']}** = `{p['value']}`"); p_col2.button("🗑️", key=f"del_param_{p['id']}", on_click=remove_param, args=(p['id'],))

        st.subheader("B. Define Ports")
        port_cols = st.columns([1, 1, 1, 2, 1]); port_cols[0].selectbox("Direction", ["input", "output"], key="new_port_dir"); port_cols[1].selectbox("Type", ["wire", "reg"], key="new_port_type"); port_cols[2].text_input("Bits/Param", "1", key="new_port_bits"); port_cols[3].text_input("Port Name", "", key="new_port_name"); port_cols[4].button("➕ Add Port", on_click=add_port, use_container_width=True)
        for p in st.session_state.form_module_ports:
            p_col1, p_col2 = st.columns([5,1]); range_str = get_port_range(p['bits']); range_display = f" `{range_str}`" if range_str else ""; p_col1.markdown(f"- `{p['direction']}` `{p['type']}`{range_display} **{p['name']}**"); p_col2.button("🗑️", key=f"del_port_{p['id']}", on_click=remove_port, args=(p['id'],))

        st.divider()
        button_text = "🚀 Update Verilog Module" if is_editing else "🚀 Generate Verilog Module"
        if st.button(button_text, type="primary", use_container_width=True):
            if not st.session_state.form_module_name.strip(): st.error("Module name is required.")
            else:
                with st.spinner("Generating Verilog with LLM..."):
                    include_statements = "\n".join([f'`include "{name}.v"' for name in st.session_state.form_module_submodules])
                    submodule_context = "This module must instantiate sub-modules. Use named port connections and parameter passing if needed.\n" if st.session_state.form_module_submodules else ""
                    params = [f"parameter {p['name']} = {p['value']}" for p in st.session_state.form_module_params]
                    param_defs_str = "#(\n    " + ",\n    ".join(params) + "\n)" if params else ""
                    port_defs = ",\n".join([f"    {p['direction']} {'' if p['direction'] == 'input' else p['type']} {get_port_range(p['bits'])} {p['name']}" for p in st.session_state.form_module_ports])
                    prompt = (f"Generate a Verilog module named '{st.session_state.form_module_name}'.\n{submodule_context}"
                              f"Functional Description: {st.session_state.form_module_desc}\nInstructions:\n"
                              f"1. If instantiating sub-modules, start with:\n{include_statements}\n\n"
                              f"2. Module definition:\nmodule {st.session_state.form_module_name} {param_defs_str}\n(\n{port_defs}\n);\n\nProvide the complete Verilog code.")
                    raw_code = generate_code_locally(prompt)
                    if raw_code:
                        design_data = {
                            "name": st.session_state.form_module_name.strip().replace(" ", "_"), "description": st.session_state.form_module_desc,
                            "ports": [p.copy() for p in st.session_state.form_module_ports], "params": [p.copy() for p in st.session_state.form_module_params],
                            "submodules": st.session_state.form_module_submodules, "code": extract_verilog_code(raw_code, st.session_state.form_module_name),
                            "testbench": "", "is_toplevel": st.session_state.form_module_is_toplevel, "vcd_path": None, "sim_output": "", "openlane_config_str": "", "openlane_log": "",
                            "testbench_goal": st.session_state.form_testbench_goal, "test_vectors": [v.copy() for v in st.session_state.form_test_vectors]
                        }
                        if design_data['is_toplevel']:
                            for i, d in enumerate(st.session_state.designs):
                                if i != st.session_state.active_design_index: d['is_toplevel'] = False
                        if is_editing:
                            st.session_state.designs[st.session_state.active_design_index] = design_data
                            st.success(f"Module '{design_data['name']}' updated!")
                        else:
                            st.session_state.designs.append(design_data)
                            st.session_state.active_design_index = len(st.session_state.designs) - 1
                            st.success(f"Module '{design_data['name']}' generated!")
                        st.rerun()
                    else: st.error("Failed to generate Verilog code.")

# --- Column 2: Workspace for Active Module ---
with col2:
    if st.session_state.active_design_index is not None:
        idx = st.session_state.active_design_index
        active_design = st.session_state.designs[idx]
        st.header(f"Workspace: {active_design['name']}")
        tab1, tab2, tab3, tab4 = st.tabs(["📝 Verilog Code", "🔬 Testbench", "🏁 Simulation", "🛠️ Synthesis"])

        with tab1:
            edited_code = st.text_area("RTL Code", value=active_design['code'], height=400, key=f"code_editor_{idx}")
            if st.button("💾 Save Code", key=f"save_code_{idx}"):
                active_design['code'] = edited_code; st.success("Verilog code saved!"); st.toast("Saved!")

        with tab2:
            st.subheader("Testbench Configuration")
            st.text_area("High-Level Goal", key="form_testbench_goal", help="Describe the overall testing objective.")
            st.write("**Test Vectors**")
            tv_cols = st.columns([4, 1])
            tv_cols[0].text_input("Signal Assignments", placeholder="e.g., A=1; B=0; reset=1;", key="new_test_vector", label_visibility="collapsed")
            tv_cols[1].button("➕ Add Vector", on_click=add_test_vector, use_container_width=True)
            
            for v in st.session_state.form_test_vectors:
                v_cols = st.columns([4, 1])
                v_cols[0].markdown(f"- `{v['assignments']}`")
                v_cols[1].button("🗑️", key=f"del_vec_{v['id']}", on_click=remove_test_vector, args=(v['id'],))
            
            st.divider()

            if st.button("🤖 Generate Testbench", key=f"gen_tb_{idx}", type="primary", use_container_width=True):
                with st.spinner("Generating testbench..."):
                    tb_module_name = f"{active_design['name']}_tb"
                    
                    # Create parameter and signal declaration strings
                    param_declarations = "\n".join([f"    localparam {p['name']} = {p['value']};" for p in active_design.get('params', [])])
                    signal_declarations = []
                    for port in active_design.get('ports', []):
                        port_type = 'reg' if port['direction'] == 'input' else 'wire'
                        signal_declarations.append(f"    {port_type} {get_port_range(port['bits'])} {port['name']};")
                    signal_declarations_str = "\n".join(signal_declarations)

                    # Create DUT instantiation string
                    port_connections = ",\n".join([f"        .{p['name']}({p['name']})" for p in active_design.get('ports', [])])
                    param_assignments = "#(\n" + ",\n".join([f"        .{p['name']}({p['name']})" for p in active_design.get('params', [])]) + "\n    )" if active_design.get('params') else ""
                    dut_instantiation = f"""    {active_design['name']} {param_assignments} uut (\n{port_connections}\n    );"""

                    vector_steps = []
                    for i, v in enumerate(st.session_state.form_test_vectors):
                        vector_steps.append(f"    // Step {i+1}\n    #10 {v['assignments']};")
                    vector_instructions = "\n".join(vector_steps)

                    prompt = (
f"""**Objective:** Generate a comprehensive and correct Verilog testbench.

**Instructions:**

1.  **MANDATORY:** The file MUST begin with `` `timescale 1ns/100ps` `` on the very first line. No exceptions.
2.  The testbench module MUST be named `{tb_module_name}`.
3.  Declare local parameters for the testbench like this:
```verilog
{param_declarations if param_declarations else "// No parameters to declare"}
```
4.  Declare `reg` and `wire` signals to connect to the DUT, like this:
```verilog
{signal_declarations_str}
```
5.  Instantiate the device under test (DUT), `{active_design['name']}`, using named port connections, like this:
```verilog
{dut_instantiation}
```
6.  The overall high-level testing goal is: "{st.session_state.form_testbench_goal}".
7.  Create an `initial begin...end` block to drive the stimulus.
8.  Inside the `initial` block, apply the following test vectors sequentially:
```verilog
{vector_instructions if vector_instructions else "// No specific test vectors provided. Create your own simple stimulus."}
```
9.  **CRITICAL:** You MUST include this VCD generation block inside the testbench module:
    ```verilog
    initial begin
      $dumpfile("{active_design['name']}.vcd");
      $dumpvars(0, {tb_module_name});
    end
    ```
10. End the simulation with `$finish` after all stimulus has been applied.

**DUT Code (for reference):**
```verilog
{active_design['code']}
```

Provide only the complete, clean Verilog code for the testbench.
""")
                    raw_tb = generate_code_locally(prompt)
                    if raw_tb:
                        active_design['testbench'] = extract_verilog_code(raw_tb, tb_module_name)
                        active_design['testbench_goal'] = st.session_state.form_testbench_goal
                        active_design['test_vectors'] = [v.copy() for v in st.session_state.form_test_vectors]
                        st.success("Testbench generated!"); st.rerun()
                    else: st.error("Failed to generate testbench.")
            
            st.subheader("Testbench Code")
            edited_tb = st.text_area("Testbench Code", value=active_design.get('testbench', ''), height=400, key=f"tb_editor_{idx}", label_visibility="collapsed")
            if st.button("💾 Save Testbench", key=f"save_tb_{idx}"):
                active_design['testbench'] = edited_tb; st.success("Testbench code saved!"); st.toast("Saved!")

        with tab3:
            st.subheader("RTL Simulation")
            if st.button("🚦 Run Simulation", key=f"run_sim_{idx}", type="primary"):
                active_design['sim_output'], active_design['vcd_path'] = "", None
                with st.spinner("Running Icarus Verilog simulation..."):
                    design_files = [f"{d['name']}.v" for d in st.session_state.designs if f'`include "{d["name"]}.v"`' in active_design.get('code', '')]
                    design_files.append(f"{active_design['name']}.v")
                    tb_file = f"{active_design['name']}_tb.v"
                    for d in st.session_state.designs:
                        with open(f"{d['name']}.v", "w") as f: f.write(d.get('code', ''))
                    with open(tb_file, "w") as f: f.write(active_design.get('testbench', ''))
                    output_file, vcd_file = f"{active_design['name']}_sim", f"{active_design['name']}.vcd"
                    try:
                        compile_cmd = ["iverilog", "-o", output_file, tb_file] + list(set(design_files))
                        compile_res = subprocess.run(compile_cmd, capture_output=True, text=True)
                        if compile_res.returncode != 0: raise subprocess.CalledProcessError(compile_res.returncode, compile_cmd, stderr=compile_res.stderr)
                        run_res = subprocess.run(["vvp", output_file], capture_output=True, text=True, check=True)
                        active_design['sim_output'] = f"Compilation:\n{compile_res.stdout}{compile_res.stderr}\n\nExecution:\n{run_res.stdout}{run_res.stderr}"
                        if os.path.exists(vcd_file):
                            st.success("Simulation successful!"); active_design['vcd_path'] = vcd_file
                        else: st.warning("Simulation ran, but VCD file was not found.")
                    except subprocess.CalledProcessError as e:
                        st.error("Error during simulation:"); active_design['sim_output'] = e.stderr
                    except Exception as e: st.error(f"An unexpected error occurred: {e}")
                st.rerun()

            if active_design.get('sim_output'):
                st.code(active_design['sim_output'], language='log')
                if "error" in active_design.get('sim_output', "").lower() and not st.session_state.show_correction_ui_sim:
                    if st.button("🤖 Correct Simulation Error", key=f"start_correct_sim_{idx}"):
                        st.session_state.show_correction_ui_sim = True
                        st.session_state.suggested_code, st.session_state.suggested_tb = "", ""
                        st.session_state.correction_prompt = (
                            f"The Verilog simulation failed. Analyze the error log and the source code, then provide a complete, corrected version of the module and/or the testbench that fixes the error.\n\n"
                            f"**Error Log:**\n```\n{active_design['sim_output']}\n```\n\n"
                            f"**Original Module Code (`{active_design['name']}.v`):**\n```verilog\n{active_design['code']}\n```\n\n"
                            f"**Original Testbench Code (`{active_design['name']}_tb.v`):**\n```verilog\n{active_design['testbench']}\n```\n\n"
                            f"Your goal is to fix the bug so the simulation can run successfully. Provide the full code for any file you change."
                        )
                        st.rerun()

            if st.session_state.show_correction_ui_sim:
                with st.expander("🛠️ Simulation Correction Workspace", expanded=True):
                    st.text_area("LLM Correction Prompt", key="correction_prompt", height=250)
                    if st.button("🤖 Generate Fix", key="generate_fix_btn_sim"):
                        with st.spinner("Asking LLM for a fix..."):
                            full_response = generate_code_locally(st.session_state.correction_prompt, is_correction=True)
                            if full_response:
                                st.session_state.suggested_code = extract_verilog_code(full_response, active_design['name'])
                                st.session_state.suggested_tb = extract_verilog_code(full_response, f"{active_design['name']}_tb")
                            else: st.error("LLM failed to provide a correction.")
                        st.rerun()

                    if st.session_state.suggested_code or st.session_state.suggested_tb:
                        st.write("#### Proposed Changes")
                        if st.session_state.suggested_code:
                            st.write(f"**Module: `{active_design['name']}.v`**")
                            diff = difflib.unified_diff(active_design['code'].splitlines(keepends=True), st.session_state.suggested_code.splitlines(keepends=True), fromfile='Original', tofile='Suggested')
                            st.code("".join(diff), language='diff')
                        if st.session_state.suggested_tb:
                            st.write(f"**Testbench: `{active_design['name']}_tb.v`**")
                            diff_tb = difflib.unified_diff(active_design['testbench'].splitlines(keepends=True), st.session_state.suggested_tb.splitlines(keepends=True), fromfile='Original', tofile='Suggested')
                            st.code("".join(diff_tb), language='diff')

                        c1, c2 = st.columns(2)
                        if c1.button("✅ Accept Changes", use_container_width=True, key="accept_sim_changes"):
                            if st.session_state.suggested_code: active_design['code'] = st.session_state.suggested_code
                            if st.session_state.suggested_tb: active_design['testbench'] = st.session_state.suggested_tb
                            st.session_state.show_correction_ui_sim = False
                            st.success("Code updated with accepted changes.")
                            st.rerun()
                        if c2.button("❌ Reject Changes", use_container_width=True, key="reject_sim_changes"):
                            st.session_state.show_correction_ui_sim = False
                            st.rerun()

            if active_design.get('vcd_path'):
                st.subheader("Waveform Viewer")
                if st.button("📈 Show Waveform", key=f"show_wave_{idx}"):
                    with st.spinner("Generating waveform image..."):
                        try:
                            wiretrace = WireTrace.from_vcd(active_design['vcd_path'])
                            image = Visualizer(Style.Dark).to_svg(wiretrace, start=0, length=1000)
                            st.image(str(image))
                        except Exception as e: st.error(f"Failed to display waveform: {e}")

        with tab4:
            st.subheader("Synthesize with OpenLane")
            st.warning("Prerequisites: Docker, OpenLane, and SKY130 PDK must be installed.", icon="⚠️")
            top_level_design = next((d for d in st.session_state.designs if d.get('is_toplevel')), None)
            if not top_level_design:
                st.error("No top-level module designated. Please set one.")
            else:
                st.info(f"Top-level module for synthesis: **{top_level_design['name']}**")
                if not active_design.get('openlane_config_str'):
                    all_verilog_files = [f"dir::src/{d['name']}.v" for d in st.session_state.designs]
                    active_design['openlane_config_str'] = json.dumps({
                        "DESIGN_NAME": top_level_design['name'], "VERILOG_FILES": list(set(all_verilog_files)),
                        "CLOCK_PORT": "clk", "CLOCK_PERIOD": 10.0,
                    }, indent=4)
                
                edited_config_str = st.text_area("OpenLane Configuration (config.json)", value=active_design['openlane_config_str'], height=250, key=f"json_editor_{idx}")
                if st.button("💾 Save Config", key=f"save_json_{idx}"):
                    active_design['openlane_config_str'] = edited_config_str
                    st.success("OpenLane config saved!"); st.toast("Saved!")

                if st.button("🛠️ Synthesize Chip", key=f"synth_{idx}", type="primary"):
                    try:
                        user_config = json.loads(active_design['openlane_config_str'])
                        design_name = user_config["DESIGN_NAME"]
                        design_dir = os.path.join(OPENLANE_DIR, "designs", design_name)
                        if os.path.exists(design_dir): shutil.rmtree(design_dir)
                        os.makedirs(os.path.join(design_dir, "src"), exist_ok=True)
                        with open(os.path.join(design_dir, "config.json"), "w") as f: json.dump(user_config, f, indent=4)
                        for d in st.session_state.designs:
                            with open(os.path.join(design_dir, "src", f"{d['name']}.v"), "w") as f: f.write(d['code'])
                        st.success(f"Set up design '{design_name}' in '{design_dir}'.")
                        docker_command = ['docker', 'run', '--rm', '-v', f'{HOME_DIR}:{HOME_DIR}', '-v', f'{OPENLANE_DIR}:/openlane', '-e', f'PDK_ROOT={PDK_ROOT}', '-e', 'PDK=sky130A', '--user', f'{os.getuid()}:{os.getgid()}', OPENLANE_IMAGE, './flow.tcl', '-design', design_name]
                        st.info("Running OpenLane flow..."); st.code(' '.join(docker_command))
                        log_placeholder, log_content = st.empty(), ""
                        process = subprocess.Popen(docker_command, cwd=OPENLANE_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                        for line in iter(process.stdout.readline, ''):
                            log_content += line
                            log_placeholder.code(log_content, language='log')
                        process.stdout.close()
                        active_design['openlane_log'] = log_content
                        if process.wait() == 0: st.success(f"✅ OpenLane flow for {design_name} completed!")
                        else: st.error(f"❌ OpenLane flow failed. Check logs.")
                        st.rerun()
                    except Exception as e: st.error(f"An error occurred during synthesis setup: {e}")

                if active_design.get('openlane_log'):
                    st.code(active_design.get('openlane_log'), language='log')
                    if "error" in active_design.get('openlane_log', "").lower() and not st.session_state.show_correction_ui_synth:
                        if st.button("🤖 Correct Synthesis Error", key=f"start_correct_synth_{idx}"):
                            st.session_state.show_correction_ui_synth = True
                            st.session_state.suggested_code = ""
                            all_designs_code = ""
                            for d in st.session_state.designs:
                                all_designs_code += f"\n--- Verilog for {d['name']} ---\n```verilog\n{d['code']}\n```\n"
                            
                            log_lines = active_design.get('openlane_log', '').splitlines()[-50:]
                            log_for_prompt = "\n".join(log_lines)
                            
                            st.session_state.correction_prompt = (
                                f"The OpenLane synthesis flow failed for top-level design '{top_level_design['name']}'. Analyze the error log and all provided Verilog files, then provide a complete, corrected version of ONLY the file(s) that need to be fixed.\n\n"
                                f"**Error Log (last 50 lines):**\n```\n{log_for_prompt}\n```\n\n"
                                f"**All Verilog Modules in Project:**\n{all_designs_code}\n\n"
                                f"Your goal is to fix the bug so the synthesis can run successfully. Provide the full code for any file you change."
                            )
                            st.rerun()

                if st.session_state.show_correction_ui_synth:
                    with st.expander("🛠️ Synthesis Correction Workspace", expanded=True):
                        st.text_area("LLM Correction Prompt", key="correction_prompt", height=300)
                        if st.button("🤖 Generate Fix", key="generate_fix_btn_synth"):
                            with st.spinner("Asking LLM for a fix..."):
                                full_response = generate_code_locally(st.session_state.correction_prompt, is_correction=True)
                                if full_response:
                                    st.session_state.suggested_code = full_response
                                else:
                                    st.error("LLM failed to provide a correction.")
                            st.rerun()

                        if st.session_state.suggested_code:
                            st.write("#### Proposed Changes")
                            any_change_found = False
                            for design in st.session_state.designs:
                                suggested_module_code = extract_verilog_code(st.session_state.suggested_code, design['name'])
                                if suggested_module_code:
                                    any_change_found = True
                                    st.write(f"**Module: `{design['name']}.v`**")
                                    diff = difflib.unified_diff(design['code'].splitlines(keepends=True), suggested_module_code.splitlines(keepends=True), fromfile='Original', tofile='Suggested')
                                    st.code("".join(diff), language='diff')
                            
                            if not any_change_found:
                                st.warning("The LLM provided a response, but I couldn't match it to any of your existing modules. Here is the raw response:")
                                st.code(st.session_state.suggested_code)

                            c1, c2 = st.columns(2)
                            if c1.button("✅ Accept Changes", use_container_width=True, key="accept_synth_changes", disabled=not any_change_found):
                                for design in st.session_state.designs:
                                    suggested_module_code = extract_verilog_code(st.session_state.suggested_code, design['name'])
                                    if suggested_module_code:
                                        design['code'] = suggested_module_code
                                st.session_state.show_correction_ui_synth = False
                                st.success("Code updated with accepted changes.")
                                st.rerun()
                            if c2.button("❌ Reject Changes", use_container_width=True, key="reject_synth_changes"):
                                st.session_state.show_correction_ui_synth = False
                                st.rerun()

    else:
        st.info("Select a design unit from the sidebar or create a new one to begin.")
