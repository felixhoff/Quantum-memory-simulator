# EIT Quantum Memory Simulator

A web-based simulator for Electromagnetically Induced Transparency (EIT) quantum memory, allowing users to visualize and understand the process of light storage and retrieval in atomic ensembles.

## Overview

This simulator implements a numerical model of photon storage and retrieval in a Λ-type atomic ensemble, based on the formalism developed in Gorshkov *et al.*, *Phys. Rev. A 76, 033805* [(link)](https://arxiv.org/abs/quant-ph/0612083). It provides an interactive way to explore:

- Light pulse propagation through an atomic medium
- Storage of light in atomic coherence
- Retrieval of stored light
- Comparison between slow light and memory protocols

[Webapp screenshot](Webapp_Screenshot.PNG)

## Features

- **Interactive Parameters**: Adjust various physical parameters through an intuitive interface:
  - Atomic properties (medium length, optical depth)
  - Field parameters (detuning, control field strength)
  - Timing parameters (pulse width, control field timing)
  - Simulation parameters (grid resolution, duration)

- **Real-time Visualization**:
  - Temporal evolution of field intensities
  - Spatiotemporal evolution of field and spin wave intensities
  - Visual indication of the atomic medium region
  - Progress tracking during simulation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Simulator_EIT_memory.git
cd Simulator_EIT_memory
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Adjust the simulation parameters in the sidebar:
   - Set atomic properties (medium length, optical depth)
   - Configure field parameters (detuning, control field)
   - Adjust timing parameters
   - Fine-tune simulation settings

2. Click "Run Simulation" to start the calculation

3. Observe the results:
   - Top plot shows temporal evolution of field intensities
   - Bottom plots show spatiotemporal evolution of field and spin wave intensities
   - The atomic medium region is marked in the spatiotemporal plots

## Physics Background

The simulator implements the Maxwell-Bloch equations for a Λ-type atomic system:

- Propagation equation for the probe field
- Coupled atomic equations for polarization and spin wave
- Control field protocols for storage and retrieval

For detailed theoretical background, see the "More information about EIT memory" section in the application.

## Requirements

- Python 3.7+
- Streamlit
- NumPy
- SciPy
- Plotly

