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

# --- Page Configuration ---
st.set_page_config(
    page_title="LLM Chip Design Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 LLM-Powered Chip Design Automation Flow")

# --- Constants ---
MODEL_NAME = "Qwen/Qwen2.5-7B" # Using the instruct-tuned model is often better for prompts
HOME_DIR = os.path.expanduser("~")
OPENLANE_DIR = os.path.join(HOME_DIR, "OpenLane")
# IMPORTANT: Update this to your OpenLane Docker image if it's different
OPENLANE_IMAGE = "efabless/openlane:e73fb3c57e687a0023fcd4dcfd1566ecd478362a-amd64"
PDK_ROOT = os.path.join(HOME_DIR, ".volare") # Common path for Volare-installed PDKs

# --- Session State Initialization ---
# This lock ensures that model loading is thread-safe.
if 'model_lock' not in st.session_state:
    st.session_state.model_lock = threading.Lock()

# Initialize all session state keys to prevent errors on first run
defaults = {
    'stage': 0,
    'module_name': "my_module",
    'ports': [],
    'design_code': "",
    'testbench_code': "",
    'vcd_path': None,
    'simulation_output': "",
    'openlane_config': {},
    'model_loaded': False,
    'tokenizer': None,
    'model': None
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --- Helper Functions ---

def set_stage(stage):
    """Callback function to set the current stage of the app."""
    st.session_state.stage = stage

def extract_verilog_code(raw_output):
    """
    Extracts Verilog code cleanly, looking for markdown-style code blocks
    or falling back to the first 'module' and the last 'endmodule'.
    """
    # Use regex to find content between ```verilog and ```
    verilog_match = re.search(r'```verilog(.*?)```', raw_output, re.DOTALL)
    if verilog_match:
        code = verilog_match.group(1).strip()
    else:
        # Fallback to finding module/endmodule if no markdown block is found
        code = raw_output

    # Clean up by finding the core module definition
    try:
        start_index = code.index("module")
        # rindex finds the last occurrence
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
    # Use a lock to prevent race conditions if multiple sessions/threads try to load
    with st.session_state.model_lock:
        if not st.session_state.model_loaded:
            st.info(f"Loading model '{MODEL_NAME}'... This may take a few minutes and significant memory.")
            try:
                tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
                # Configure quantization to reduce memory usage
                quantization_config = BitsAndBytesConfig(load_in_4bit=True)
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    device_map="auto",  # Let transformers handle device placement
                    quantization_config=quantization_config,
                    torch_dtype=torch.float16, # Use float16 for further optimization
                )
                st.session_state.tokenizer = tokenizer
                st.session_state.model = model
                st.session_state.model_loaded = True
                st.success("✅ Model and Tokenizer loaded successfully!")
                # Rerun the script to remove the loading message and show the main app
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error loading model: {e}")
                st.error("This likely means your system is out of memory or you don't have a compatible GPU. Try restarting the app or checking your hardware setup.")
                st.stop()


def generate_code_locally(prompt):
    """Generates code using the loaded local model from session state."""
    if not st.session_state.model_loaded:
        st.error("Model not loaded. Cannot generate code.")
        return None

    tokenizer = st.session_state.tokenizer
    model = st.session_state.model

    try:
        # Format the prompt for the instruct-tuned model
        messages = [
            {"role": "system", "content": "You are a helpful assistant that generates Verilog code."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        outputs = model.generate(**inputs, max_new_tokens=8192, pad_token_id=tokenizer.eos_token_id)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # The response includes the prompt, so we need to extract only the assistant's part
        response_start = generated_text.rfind("assistant")
        if response_start != -1:
            return generated_text[response_start + len("assistant"):].strip()
        return generated_text # Fallback if template isn't found
    except Exception as e:
        st.error(f"Error during model inference: {e}")
        return None


# --- Main Application UI ---

# This is the "lock". The app UI will only be shown if the model is loaded.
if not st.session_state.model_loaded:
    st.write("""
    Welcome to the interactive chip design assistant. This tool uses a Large Language Model
    to help you generate, simulate, and synthesize a digital chip design.
    """)
    with st.spinner("Initializing... The model will be downloaded and loaded into memory."):
        load_model()
    st.stop() # Stop execution here until load_model() reruns the script

# --- THE REST OF THE APP RUNS ONLY IF THE MODEL IS LOADED ---

st.write("""
Welcome to the interactive chip design assistant. The model is loaded and ready.
Follow the steps below to generate, simulate, and synthesize your chip.
""")

# --- STAGE 1: VERILOG DESIGN GENERATION ---
st.header("1. Generate Verilog RTL Design")

# Part 1: Port Management (outside the form)
st.subheader("A. Define Ports")
st.write("Add all the necessary input and output ports for your module here.")

def add_port_callback():
    """Callback to add a new port to the session state."""
    port_name = st.session_state.get("new_port_name", "")
    if port_name:
        st.session_state.ports.append({
            "direction": st.session_state.new_port_dir,
            "type": st.session_state.new_port_type,
            "bits": int(st.session_state.new_port_bits),
            "name": port_name.replace(" ", "_") # Sanitize port name on creation
        })
        # Clear input for next entry
        st.session_state.new_port_name = ""
    else:
        st.warning("Port name cannot be empty.")

# Use different keys for the "new port" inputs to avoid conflicts
port_cols = st.columns([1, 1, 1, 2, 1])
port_cols[0].selectbox("Direction", ["input", "output"], key="new_port_dir")
port_cols[1].selectbox("Type", ["wire", "reg"], key="new_port_type", help="Use 'reg' for outputs that hold state.")
port_cols[2].number_input("Bits", min_value=1, max_value=128, value=1, key="new_port_bits")
port_cols[3].text_input("Port Name", key="new_port_name", help="Name for the new port.")
port_cols[4].button("➕ Add Port", on_click=add_port_callback, help="Click to add the port defined above to the list.", use_container_width=True)

# Display current ports
if st.session_state.ports:
    st.write("**Current Ports:**")
    for i, port in enumerate(st.session_state.ports):
        st.markdown(f"- `{port['direction']}` `{port['type']}` `[{port['bits']-1}:0]` **{port['name']}**")
else:
    st.caption("No ports defined yet.")

st.divider()

# Part 2: Module Name and Generation (inside a form)
st.subheader("B. Generate Module")
st.write("Once you have added all ports, provide a module name and generate the Verilog code.")

with st.form("verilog_design_form"):
    st.text_input(
        "Module Name",
        value=st.session_state.module_name,
        key="input_module_name_temp",
        help="Enter the name for your Verilog module. Spaces will be replaced with underscores on submission."
    )
    submitted_generate = st.form_submit_button("🚀 Generate Verilog Code", type="primary", use_container_width=True)

if submitted_generate:
    module_name_clean = st.session_state.input_module_name_temp.replace(" ", "_").strip()
    
    if not module_name_clean:
        st.error("Module name is required.")
    elif not st.session_state.ports:
        st.error("You must add at least one port before generating the module.")
    else:
        st.session_state.module_name = module_name_clean
        
        with st.spinner("Generating Verilog code with the LLM..."):
            port_definitions = ",\n".join([f"    {p['direction']} {'' if p['direction'] == 'input' else p['type']} [{p['bits']-1}:0] {p['name']}" for p in st.session_state.ports])
            design_prompt = (
                f"Generate a complete and clean Verilog code for a module named '{st.session_state.module_name}'. "
                f"The module should have the following ports:\n"
                f"{port_definitions}\n"
                f"The functionality should be a simple pass-through or basic logic based on the port names. "
                f"Provide only the Verilog code itself, starting with `module` and ending with `endmodule`."
            )
            raw_code = generate_code_locally(design_prompt)
            if raw_code:
                st.session_state.design_code = extract_verilog_code(raw_code)
                set_stage(1)
                st.rerun() 
            else:
                st.error("Failed to generate Verilog code.")

if st.session_state.stage >= 1:
    st.subheader("📝 Review and Edit Generated Verilog")
    st.write("You can modify the generated code below. Any changes will be used in the next steps.")
    st.session_state.design_code = st.text_area(
        "Verilog Design Code", value=st.session_state.design_code, height=300, key="design_code_editor"
    )
    if st.button("Proceed to Testbench Generation ▶️"):
        set_stage(2)

st.divider()

# --- STAGE 2: TESTBENCH GENERATION ---
if st.session_state.stage >= 2:
    st.header("2. Generate Verilog Testbench")
    if st.button("🤖 Generate Testbench with LLM", type="primary"):
        with st.spinner("Generating testbench... This may take a moment."):
            testbench_prompt = f"""Generate a comprehensive Verilog testbench for the following module named '{st.session_state.module_name}'.
Ensure the testbench correctly instantiates the module, connects to all of its ports, and provides meaningful stimulus to verify its functionality.
Include a `timescale 1ns/100ps` directive at the top. The testbench should also include a `$dumpfile` and `$dumpvars` call to generate a VCD waveform file. Finally, it must have a `$finish` statement to end the simulation.

Here is the Verilog code for the module to be tested:
```verilog
{st.session_state.design_code}
```
Please provide only the Verilog code for the testbench, starting with `module` and ending with `endmodule`."""
            raw_tb_code = generate_code_locally(testbench_prompt)
            if raw_tb_code:
                st.session_state.testbench_code = extract_verilog_code(raw_tb_code)
                set_stage(3)
            else:
                st.error("Failed to generate testbench code.")

if st.session_state.stage >= 3:
    st.subheader("📝 Review and Edit Generated Testbench")
    st.write("You can modify the testbench code below.")
    st.session_state.testbench_code = st.text_area(
        "Verilog Testbench Code", value=st.session_state.testbench_code, height=300, key="testbench_code_editor"
    )
    if st.button("Proceed to RTL Simulation ▶️"):
        set_stage(4)

st.divider()

#--- STAGE 3: RTL SIMULATION (CORRECTED) ---
if st.session_state.stage >= 4:
    st.header("3. Run RTL Simulation (Icarus Verilog)")
    if st.button("🚦 Run Simulation", type="primary"):
        # Reset previous output before running a new simulation
        st.session_state.simulation_output = ""
        with st.spinner("Running simulation..."):
            design_file = f"{st.session_state.module_name}.v"
            tb_file = f"{st.session_state.module_name}_tb.v"
            with open(design_file, "w") as f: f.write(st.session_state.design_code)
            with open(tb_file, "w") as f: f.write(st.session_state.testbench_code)

            base_name = os.path.splitext(tb_file)[0]
            output_file = f"{base_name}_sim"
            vcd_file = f"{base_name}.vcd"
            
            compile_command = ["iverilog", "-o", output_file, design_file, tb_file]
            run_command = ["vvp", output_file]

            try:
                # Compile
                compile_result = subprocess.run(compile_command, capture_output=True, text=True, check=True)
                sim_output = f"Compilation successful:\n{compile_result.stdout}\n{compile_result.stderr}"
                # Run
                run_result = subprocess.run(run_command, capture_output=True, text=True, check=True)
                sim_output += f"\nSimulation run successful:\n{run_result.stdout}\n{run_result.stderr}"
                
                st.session_state.simulation_output = sim_output
                
                if os.path.exists(vcd_file):
                    st.success(f"✅ Simulation successful! Waveform file '{vcd_file}' created.")
                    st.session_state.vcd_path = vcd_file
                    set_stage(5)
                else:
                    st.error(f"Simulation ran, but the VCD file '{vcd_file}' was not created as expected.")
                    st.session_state.vcd_path = None

            except subprocess.CalledProcessError as e:
                st.error("❌ Error during simulation:")
                # If there's an error, save the stderr to the output state so it can be displayed
                st.session_state.simulation_output = e.stderr
                st.session_state.vcd_path = None
    
    # *** FIX: Display the log OUTSIDE the button's if block. ***
    # This ensures the log is shown after the automatic rerun on button click.
    if st.session_state.simulation_output:
        st.subheader("Simulation Log")
        st.code(st.session_state.simulation_output, language='log')

st.divider()

#--- STAGE 4: DISPLAY WAVEFORM ---
if st.session_state.stage >= 5 and st.session_state.vcd_path:
    st.header("4. View Simulation Waveform")
    if st.button("📈 Show Waveform", type="primary"):
        with st.spinner("Generating waveform image..."):
            try:
                svg_file = os.path.splitext(st.session_state.vcd_path)[0] + ".svg"
                wiretrace = WireTrace.from_vcd(st.session_state.vcd_path)
                # Render the image using the Visualizer
                image = Visualizer(Style.Dark).to_svg(wiretrace)

                with open(svg_file, "w") as f: f.write(str(image))
                st.image(svg_file)
                set_stage(6)
            except Exception as e:
                st.error(f"An error occurred while displaying the waveform: {e}")

st.divider()

#--- STAGE 5: OPENLANE SYNTHESIS ---
if st.session_state.stage >= 6:
    st.header("5. Synthesize a Chip with OpenLane")
    st.warning("Prerequisites: Docker, OpenLane, and the SKY130 PDK must be installed correctly on your system.", icon="⚠️")

    if not st.session_state.openlane_config or st.session_state.openlane_config.get("DESIGN_NAME") != st.session_state.module_name:
        st.session_state.openlane_config = {
            "DESIGN_NAME": st.session_state.module_name,
            "VERILOG_FILES": f"dir::src/{st.session_state.module_name}.v",
            "CLOCK_PORT": "clk", # User might need to change this
            "CLOCK_PERIOD": 10.0,
            "DESIGN_IS_CORE": False,
            "FP_PDN_CORE_RING": False,
            "RT_MAX_LAYER": "met4"
        }

    config_str = json.dumps(st.session_state.openlane_config, indent=4)
    edited_config_str = st.text_area(
        "OpenLane Configuration (config.json)", value=config_str, height=250,
        help="Edit the JSON configuration for the OpenLane flow. Ensure CLOCK_PORT is correct."
    )

    if st.button("🛠️ Synthesize Chip with OpenLane", type="primary"):
        try:
            user_config = json.loads(edited_config_str)
            st.session_state.openlane_config = user_config
            design_name = user_config.get("DESIGN_NAME", st.session_state.module_name)

            design_dir = os.path.join(OPENLANE_DIR, "designs", design_name)
            src_dir = os.path.join(design_dir, "src")
            os.makedirs(src_dir, exist_ok=True)
            
            with open(os.path.join(design_dir, "config.json"), "w") as f: json.dump(user_config, f, indent=4)
            
            source_verilog_file = f"{st.session_state.module_name}.v"
            dest_verilog_path = os.path.join(src_dir, os.path.basename(source_verilog_file))
            shutil.copy(source_verilog_file, dest_verilog_path)
            
            st.success(f"Successfully set up design '{design_name}' in '{design_dir}'.")

            docker_command = [
                'docker', 'run', '--rm',
                '-v', f'{HOME_DIR}:{HOME_DIR}', '-v', f'{OPENLANE_DIR}:/openlane',
                '-e', f'PDK_ROOT={PDK_ROOT}', '-e', 'PDK=sky130A',
                '--user', f'{os.getuid()}:{os.getgid()}',
                OPENLANE_IMAGE,
                './flow.tcl', '-design', design_name
            ]

            st.info("Running OpenLane flow. This will take a long time. See output below for progress.")
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
                st.info(f"Find your results in: {os.path.join(design_dir, 'runs')}")
            else:
                st.error(f"❌ OpenLane flow failed with return code {return_code}. Check the logs above for errors.")

        except json.JSONDecodeError:
            st.error("Invalid JSON in OpenLane configuration. Please correct it.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
