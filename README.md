# LLM for Chip Design Automation

This project leverages Large Language Models (LLMs) to automate chip design, transforming high-level prompts and specifications into synthesized chip designs. It integrates tools like Icarus Verilog and OpenLane to facilitate the design and verification process.

## Features
- Converts natural language prompts into Verilog code.
- Automates chip design synthesis and place-and-route using OpenLane.
- Supports simulation and verification with Icarus Verilog.
- Scalable for various chip design workflows.

## Prerequisites
- **Operating System**: Ubuntu 22.04 (or compatible Linux distribution)
- **Hardware**: Minimum 8GB RAM, 20GB free disk space
- **Software**:
  - Anaconda or Miniconda
  - Python 3.11
  - Icarus Verilog
  - OpenLane 1
  - Git

## Installation

Follow these steps to set up the environment for the LLM for Chip Design Automation project.

1. **Install Anaconda or Miniconda**
   - Download and install Anaconda or Miniconda from [the official website](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
   - Follow the installation instructions for your system.

2. **Create a Conda Environment**
   ```bash
   conda create -n llm_env python=3.11
   ```

3. **Activate the Environment**
   ```bash
   conda activate llm_env
   ```

4. **Install Python Dependencies**
   - Ensure you have a `requirements.txt` file with necessary dependencies.
   - Run:
     ```bash
     pip install -r requirements.txt
     ```
   - Example `requirements.txt` might include:
     ```
     numpy
     pandas
     torch
     transformers
     ```

5. **Install Icarus Verilog**
   ```bash
   sudo apt update
   sudo apt install iverilog
   ```

6. **Install OpenLane**
   ```bash
   cd $HOME
   git clone https://github.com/The-OpenROAD-Project/OpenLane.git
   cd OpenLane
   make
   make test
   ```
   - Ensure Docker is installed, as OpenLane requires it. Follow [Docker's official installation guide](https://docs.docker.com/engine/install/ubuntu/) if needed.
   - The `make test` command verifies the OpenLane installation.

## Usage
1. Activate the Conda environment:
   ```bash
   conda activate llm_env
   ```
2. Run the Streamlit application:
   ```bash
   streamlit run chip_design_automation_v2.py
   ```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- [OpenLane](https://github.com/The-OpenROAD-Project/OpenLane) for chip design automation tools.
- [Icarus Verilog](http://iverilog.icarus.com/) for Verilog simulation.