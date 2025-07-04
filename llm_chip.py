import streamlit as st
import subprocess
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Configuration ---
# The model identifier. Transformers will use the cached (downloaded) version.
MODEL_NAME = "Qwen/Qwen2.5-7B"

# --- Load Local Model and Tokenizer ---
# This function is cached so the model is only loaded once.
@st.cache_resource
def load_local_model():
    """
    Loads the model and tokenizer from the local cache.
    """
    st.info(f"Loading model '{MODEL_NAME}' from local cache... This may take a moment.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    st.success(f"Model '{MODEL_NAME}' loaded successfully.")
    return tokenizer, model

# --- Local Hugging Face Model Interaction ---
def generate_verilog_locally(prompt, tokenizer, model):
    """
    Generates Verilog code using the loaded local model.
    """
    try:
        # Encode the prompt text into tokens
        inputs = tokenizer(prompt, return_tensors="pt")

        # Generate output tokens from the model
        outputs = model.generate(**inputs, max_new_tokens=8096, pad_token_id=tokenizer.eos_token_id)
        
        # Decode the tokens back to text, skipping special tokens
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Often, the model's output includes the original prompt, so we remove it.
        # This step might need adjustment depending on the model's specific output format.
        if generated_text.startswith(prompt):
            return generated_text[len(prompt):].strip()
        return generated_text.strip()

    except Exception as e:
        st.error(f"Error during local model inference: {e}")
        return None

# --- Icarus Verilog Simulation ---
def run_simulation(design_file, testbench_file):
    """
    Compiles and simulates Verilog files using Icarus Verilog and returns the VCD output.
    """
    output_file = "simulation_output"
    vcd_file = "waveform.vcd"
    compile_command = ["iverilog", "-o", output_file, design_file, testbench_file]
    run_command = ["vvp", output_file]

    try:
        # Compile the Verilog code
        compile_result = subprocess.run(compile_command, capture_output=True, text=True, check=True)
        st.text("Compilation Output:")
        st.code(compile_result.stdout or "No output", language='log')
        if compile_result.stderr:
            st.text("Compilation Errors:")
            st.code(compile_result.stderr, language='log')

        # Run the simulation
        run_result = subprocess.run(run_command, capture_output=True, text=True, check=True)
        st.text("Simulation Output:")
        st.code(run_result.stdout or "No output", language='log')
        if run_result.stderr:
            st.text("Simulation Errors:")
            st.code(run_result.stderr, language='log')

        return vcd_file if os.path.exists(vcd_file) else None

    except FileNotFoundError:
        st.error("Icarus Verilog (`iverilog` and `vvp`) not found. Please ensure it is installed and in your system's PATH.")
        return None
    except subprocess.CalledProcessError as e:
        st.error(f"An error occurred during simulation:")
        st.text("Command:")
        st.code(' '.join(e.cmd), language='bash')
        st.text("Return Code:")
        st.code(e.returncode)
        st.text("Stdout:")
        st.code(e.stdout or "No output", language='log')
        st.text("Stderr:")
        st.code(e.stderr or "No output", language='log')
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

# --- Waveform Visualization ---
def display_waveform(vcd_file):
    """
    Converts a VCD file to an SVG using sootty and displays it.
    """
    svg_file = "waveform.svg"
    try:
        # Use sootty to convert VCD to SVG
        sootty_command = ["sootty", "-i", vcd_file, "-o", svg_file]
        subprocess.run(sootty_command, check=True)

        if os.path.exists(svg_file):
            st.image(svg_file)
        else:
            st.warning("Could not generate waveform image.")

    except FileNotFoundError:
        st.error("`sootty` is not installed. Please install it using `pip install sootty`.")
    except subprocess.CalledProcessError as e:
        st.error(f"Error running sootty: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred during waveform visualization: {e}")

# --- Streamlit App ---
st.title("Verilog Generator and Simulator (Local Model)")

# Load the model and tokenizer
tokenizer, model = load_local_model()

st.write(
    f"This application uses the local `{MODEL_NAME}` model "
    "to generate Verilog code for a specified module and its testbench. "
    "The generated code is then simulated using Icarus Verilog, and the waveform is displayed."
)

# Input for the Verilog module
module_name = st.text_input("Enter the name of the Verilog file to create (e.g., 'd_flip_flop'):")

if st.button("Generate and Simulate"):
    if module_name:
        design_filename = f"{module_name}.v"
        testbench_filename = f"{module_name}_tb.v"

        # Prompt for generating the design
        design_prompt = f"Generate the Verilog code for a {module_name}."
        st.info(f"Generating Verilog for: {module_name}...")
        design_code = generate_verilog_locally(design_prompt, tokenizer, model)

        if design_code:
            st.subheader("Generated Verilog Design")
            st.code(design_code, language='verilog')
            with open(design_filename, "w") as f:
                f.write(design_code)

            # Prompt for generating the testbench
            testbench_prompt = f"Generate a testbench for the {module_name}."
            st.info(f"Generating testbench for: {module_name}...")
            testbench_code = generate_verilog_locally(testbench_prompt, tokenizer, model)

            if testbench_code:
                st.subheader("Generated Verilog Testbench")
                st.code(testbench_code, language='verilog')
                with open(testbench_filename, "w") as f:
                    f.write(testbench_code)

                # Run the simulation
                st.subheader("Simulation Results")
                vcd_file = run_simulation(design_filename, testbench_filename)

                # Display the waveform
                if vcd_file:
                    st.subheader("Waveform")
                    display_waveform(vcd_file)
    else:
        st.warning("Please enter a file name for the Verilog module.")

# import streamlit as st
# import subprocess
# import os
# import torch
# # Import the main model classes and the new config class
# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# # --- Configuration ---
# # The model identifier. Transformers will use the cached (downloaded) version.
# MODEL_NAME = "Qwen/Qwen2.5-7B"

# # --- Load Local Model and Tokenizer with 8-bit Quantization ---
# @st.cache_resource
# def load_local_model():
#     """
#     Loads the model and tokenizer from the local cache using 8-bit quantization.
#     """
#     st.info(f"Loading model '{MODEL_NAME}' in 8-bit mode... This may take a moment.")
    
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
#     # 1. Define the 8-bit quantization configuration
#     quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    
#     # 2. Load the model with the quantization config and automatic device placement
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_NAME,
#         quantization_config=quantization_config,
#         device_map="auto",
#     )
    
#     st.success(f"Model '{MODEL_NAME}' loaded successfully.")
#     return tokenizer, model

# # --- Local Hugging Face Model Interaction ---
# def generate_verilog_locally(prompt, tokenizer, model):
#     """
#     Generates Verilog code using the loaded local model.
#     """
#     try:
#         # Encode the prompt text into tokens
#         inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

#         # Generate output tokens from the model
#         outputs = model.generate(**inputs, max_new_tokens=8096, pad_token_id=tokenizer.eos_token_id)
        
#         # Decode the tokens back to text, skipping special tokens
#         generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
#         # Often, the model's output includes the original prompt, so we remove it.
#         if generated_text.startswith(prompt):
#             return generated_text[len(prompt):].strip()
#         return generated_text.strip()

#     except Exception as e:
#         st.error(f"Error during local model inference: {e}")
#         return None

# # --- Icarus Verilog Simulation ---
# def run_simulation(design_file, testbench_file):
#     """
#     Compiles and simulates Verilog files using Icarus Verilog and returns the VCD output.
#     """
#     output_file = "simulation_output"
#     vcd_file = "waveform.vcd"
#     compile_command = ["iverilog", "-o", output_file, design_file, testbench_file]
#     run_command = ["vvp", output_file]

#     try:
#         # Compile the Verilog code
#         compile_result = subprocess.run(compile_command, capture_output=True, text=True, check=True)
#         st.text("Compilation Output:")
#         st.code(compile_result.stdout or "No output", language='log')
#         if compile_result.stderr:
#             st.text("Compilation Errors:")
#             st.code(compile_result.stderr, language='log')

#         # Run the simulation
#         run_result = subprocess.run(run_command, capture_output=True, text=True, check=True)
#         st.text("Simulation Output:")
#         st.code(run_result.stdout or "No output", language='log')
#         if run_result.stderr:
#             st.text("Simulation Errors:")
#             st.code(run_result.stderr, language='log')

#         return vcd_file if os.path.exists(vcd_file) else None

#     except FileNotFoundError:
#         st.error("Icarus Verilog (`iverilog` and `vvp`) not found. Please ensure it is installed and in your system's PATH.")
#         return None
#     except subprocess.CalledProcessError as e:
#         st.error(f"An error occurred during simulation:")
#         st.text("Command:")
#         st.code(' '.join(e.cmd), language='bash')
#         st.text("Return Code:")
#         st.code(e.returncode)
#         st.text("Stdout:")
#         st.code(e.stdout or "No output", language='log')
#         st.text("Stderr:")
#         st.code(e.stderr or "No output", language='log')
#         return None
#     except Exception as e:
#         st.error(f"An unexpected error occurred: {e}")
#         return None

# # --- Waveform Visualization ---
# def display_waveform(vcd_file):
#     """
#     Converts a VCD file to an SVG using sootty and displays it.
#     """
#     svg_file = "waveform.svg"
#     try:
#         # Use sootty to convert VCD to SVG
#         sootty_command = ["sootty", "-i", vcd_file, "-o", svg_file]
#         subprocess.run(sootty_command, check=True)

#         if os.path.exists(svg_file):
#             st.image(svg_file)
#         else:
#             st.warning("Could not generate waveform image.")

#     except FileNotFoundError:
#         st.error("`sootty` is not installed. Please install it using `pip install sootty`.")
#     except subprocess.CalledProcessError as e:
#         st.error(f"Error running sootty: {e}")
#     except Exception as e:
#         st.error(f"An unexpected error occurred during waveform visualization: {e}")

# # --- Streamlit App ---
# st.title("Verilog Generator and Simulator (Local Model)")

# # Load the model and tokenizer
# tokenizer, model = load_local_model()

# st.write(
#     f"This application uses the local `{MODEL_NAME}` model "
#     "to generate Verilog code for a specified module and its testbench. "
#     "The generated code is then simulated using Icarus Verilog, and the waveform is displayed."
# )

# # Input for the Verilog module
# module_name = st.text_input("Enter the name of the Verilog file to create (e.g., 'd_flip_flop'):")

# if st.button("Generate and Simulate"):
#     if module_name:
#         design_filename = f"{module_name}.v"
#         testbench_filename = f"{module_name}_tb.v"

#         # Prompt for generating the design
#         design_prompt = f"Generate the Verilog code for a {module_name}."
#         st.info(f"Generating Verilog for: {module_name}...")
#         design_code = generate_verilog_locally(design_prompt, tokenizer, model)

#         if design_code:
#             st.subheader("Generated Verilog Design")
#             st.code(design_code, language='verilog')
#             with open(design_filename, "w") as f:
#                 f.write(design_code)

#             # Prompt for generating the testbench
#             testbench_prompt = f"Generate a testbench for the {module_name}."
#             st.info(f"Generating testbench for: {module_name}...")
#             testbench_code = generate_verilog_locally(testbench_prompt, tokenizer, model)

#             if testbench_code:
#                 st.subheader("Generated Verilog Testbench")
#                 st.code(testbench_code, language='verilog')
#                 with open(testbench_filename, "w") as f:
#                     f.write(testbench_code)

#                 # Run the simulation
#                 st.subheader("Simulation Results")
#                 vcd_file = run_simulation(design_filename, testbench_filename)

#                 # Display the waveform
#                 if vcd_file:
#                     st.subheader("Waveform")
#                     display_waveform(vcd_file)
#     else:
#         st.warning("Please enter a file name for the Verilog module.")