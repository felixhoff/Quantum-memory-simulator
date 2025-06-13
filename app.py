import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import plotly.graph_objects as go

# Initialize session state for tracking simulation results
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None
if 'simulation_results_on' not in st.session_state:
    st.session_state.simulation_results_on = None
if 'simulation_results_spin' not in st.session_state:
    st.session_state.simulation_results_spin = None
if 'last_params' not in st.session_state:
    st.session_state.last_params = None

# -------------------- Physical scaling (for display only) --------------------
time_unit = 1e-6  # s
length_unit = 1e-2  # m
Gamma = 2 * np.pi * 6e6  # rad/s

# -------------------- UI Setup --------------------
st.set_page_config(page_title="Quantum memory", layout="wide")
st.title("EIT Quantum Memory Simulator")
st.caption("Simulate light pulse storage and retrieval using Electromagnetically Induced Transparency (EIT).")

with st.expander("More information about EIT memory"):
    st.markdown("""
This webapp presents a numerical simulation modeling **photon storage and retrieval in a Λ-type atomic ensemble**, based on the formalism developed in Gorshkov *et al.*, *Phys. Rev. A 76, 033805* [(link)](https://arxiv.org/abs/quant-ph/0612083).

We consider a one-dimensional Λ-type three-level atomic medium with states $|g\\rangle$, $|s\\rangle$, $|e\\rangle$, interacting with:
- A weak quantum **probe field** $\\hat{E}(z, t)$ resonant with $|g\\rangle \\leftrightarrow |e\\rangle$,
- A classical **control field** $\\Omega(t)$ coupling $|s\\rangle \\leftrightarrow |e\\rangle$.

The Maxwell-Bloch equations (in the slowly-varying envelope approximation and in dimensionless units normalized by $\\Gamma$) are:

$$\\hat{E}(z + \\delta z, t) = \\hat{E}(z, t) + i \\sqrt{d} \\, \\delta z \\cdot \\hat{P}(z, t)$$

$$\\frac{d}{dt} \\hat{P}(z, t) = - (1 + i \\Delta) \\hat{P}(z, t) + i \\sqrt{d} \\, \\hat{E}(z, t) + i \\Omega(t) \\hat{S}(z, t)$$

$$\\frac{d}{dt} \\hat{S}(z, t) = i \\Omega^*(t) \\hat{P}(z, t)$$

Where:
- $\\hat{P}(z, t)$ is the optical polarization ($\\sim \\hat{\\rho}_{eg}$),
- $\\hat{S}(z, t)$ is the spin coherence ($\\sim \\hat{\\rho}_{sg}$),
- $\\Delta$ is the detuning (normalized to $\\Gamma$),
- $d$ is the resonant optical depth.

**Numerical Implementation:** Time and space are discretized into uniform grids: $t \\in [0, t_\\text{max}]$, $z \\in [0, L]$. The electric field $\\hat{E}(z, t)$ is initialized with an incoming Gaussian wavepacket:

$$\\hat{E}(0, t) = \\exp\\left[-\\frac{(t - t_0)^2}{2\\sigma^2}\\right]$$

For each spatial step $z_i \\to z_{i+1}$, we:
1. **Interpolate** $\\hat{E}(z_i, t)$ over time.
2. **Solve** the local ODE system in time for $\\hat{P}(z_i, t)$, $\\hat{S}(z_i, t)$ using a Runge-Kutta solver.
3. **Update** the probe field using:

$$\\hat{E}(z_{i+1}, t) = \\hat{E}(z_i, t) + i \\sqrt{d} \\delta z \\cdot \\hat{P}(z_i, t)$$

This yields a full spatiotemporal evolution of $\\hat{E}(z, t)$, $\\hat{P}(z, t)$, and $\\hat{S}(z, t)$.

**EIT memory protocol** (Write → Hold → Read):

$$
\\Omega(t) = \\left\\{
\\begin{array}{ll}
\\Omega_0 & \\text{if } t < t_\\text{off} \\\\
0 & \\text{if } t_\\text{off} \\leq t < t_\\text{on} \\\\
\\Omega_0 & \\text{if } t \\geq t_\\text{on}
\\end{array}
\\right.
$$

This models **storage** by switching off the control field (closing the transparency window), causing the probe to be mapped into the spin wave $\\hat{S}(z, t)$, and **retrieval** by switching the control back on.

The transmitted probe intensity is computed as:

$$I_\\text{out}(t) = |\\hat{E}(z = L, t)|^2$$

**Remarks:**
- The model assumes perfect coherence and no decay of the spin wave $\\hat{S}$,
- The discretized propagation approximates continuous-space evolution via first-order Euler stepping,
- The use of time-local ODE integration (via `solve_ivp`) is equivalent to solving the coupled Maxwell-Bloch equations under the SVEA.
    """)

# -------------------- Sidebar Parameters --------------------
st.sidebar.header("Simulation Controls")

with st.sidebar.expander("🧪 Atomic Properties", expanded=False):
    L = st.slider("Medium Length [cm]", 0.5, 2.0, 1.0, 0.1,
                  help="Length of the medium in centimeters.")
    d = st.slider("Optical Depth d", 10, 300, 150, 10,
                  help="Effective optical depth of the medium.")
    L_vacuum = st.slider("Vacuum Region Length [cm]", 0.1, 1.0, 0.5, 0.1,
                        help="Length of vacuum regions before and after the medium.")

with st.sidebar.expander("🌈 Field Parameters", expanded=False):
    Delta = st.slider("Detuning Δ [Γ units]", -10.0, 10.0, 0.0, 0.1,
                      help="Detuning between light and atom resonance, in units of Γ.")
    Omega_const = st.slider("Control Field Ω [Γ units]", 1.0, 50.0, 8.0, 1.0,
                            help="Rabi frequency of the control field, in units of Γ.")
    control_edge_width = st.slider("Control Edge Width [ns]", 10, 300, 240, 5,
                                  help="Width of the Gaussian edges for control field transitions (in nanoseconds).")

with st.sidebar.expander("⏱️ Timings", expanded=False):
    t0 = st.slider("Input Pulse Center t₀ [μs]", 0.0, 5.0, 1.5, 0.1,
                   help="Center time of the input pulse.")
    sigma = st.slider("Input Pulse Width σ [μs]", 0.1, 1.0, 0.65, 0.05,
                      help="Width (std dev) of the input Gaussian pulse.")
    control_off = st.slider("Control OFF [μs]", 1.0, 5.0, 2.25, 0.05,
                            help="Time when control field is turned OFF.")
    control_on = st.slider("Control ON Resume [μs]", 3.0, 9.0, 5.4, 0.05,
                           help="Time when control field resumes.")
    

with st.sidebar.expander("⚙️ Advanced Simulation Parameters", expanded=False):
    Nz = st.slider("Spatial Steps Nz", 100, 2000, 500, 100,
                   help="Number of spatial grid points.")
    Nt = st.slider("Time Steps Nt", 100, 2000, 300, 100,
                   help="Number of time points in simulation.")
    t_max = st.slider("Simulation Duration tₘₐₓ [μs]", 5.0, 20.0, 10.0, 0.5,
                      help="Total simulation time window in microseconds.")

st.sidebar.markdown("---")
st.sidebar.info("Adjust parameters and click **Run Simulation** to begin.")
run_simulation = st.sidebar.button("🚀 Run Simulation", use_container_width=True)

# -------------------- Grids --------------------
# Create extended grid including vacuum regions
z_grid = np.linspace(-L_vacuum, L + L_vacuum, Nz)
dz = z_grid[1] - z_grid[0]
t_grid = np.linspace(0, t_max, Nt)

# Calculate the indices for different regions
vacuum_before_idx = np.where(z_grid < 0)[0]
medium_idx = np.where((z_grid >= 0) & (z_grid <= L))[0]
vacuum_after_idx = np.where(z_grid > L)[0]

# Initial field at z = -L_vacuum
Ein = np.exp(- (t_grid - t0)**2 / (2 * sigma**2))

def Omega_t(t):
    # Convert edge width from ns to μs
    edge_width = control_edge_width * 1e-3  # convert to μs
    
    # Calculate the Gaussian transitions
    off_transition = np.exp(-(t - control_off)**2 / (2 * edge_width**2))
    on_transition = np.exp(-(t - control_on)**2 / (2 * edge_width**2))
    
    # Combine the transitions
    if t < control_off:
        return Omega_const
    elif t < control_on:
        return Omega_const * off_transition
    else:
        return Omega_const * (1 - on_transition + off_transition)

# Calculate control field values
Omega_vals = np.array([Omega_t(t) for t in t_grid])

# -------------------- Plot Function --------------------
def create_plot(show_results=False):
    # Create two subplots
    fig = go.Figure()
    
    # Plot input field at z = -L_vacuum
    fig.add_trace(go.Scatter(x=t_grid, y=np.abs(Ein)**2, fill='tozeroy',
                             name="Input", line=dict(color='steelblue')))
    fig.add_trace(go.Scatter(x=t_grid, y=Omega_vals / np.max(Omega_vals), name="Control",
                             line=dict(color='firebrick', dash='dash')))
    
    if show_results:
        if st.session_state.simulation_results is not None:
            # Plot output at z = L + L_vacuum
            E_out = np.abs(st.session_state.simulation_results[-1, :])**2
            fig.add_trace(go.Scatter(x=t_grid, y=E_out, fill='tozeroy',
                                    name="Output (Memory)", line=dict(color='darkorange')))
        
        if st.session_state.simulation_results_on is not None:
            E_out_on = np.abs(st.session_state.simulation_results_on[-1, :])**2
            fig.add_trace(go.Scatter(x=t_grid, y=E_out_on, fill='tozeroy',
                                    name="Output (Slow light)", line=dict(color='seagreen')))

    fig.update_layout(xaxis_title="Time [μs]",
                      yaxis_title="Normalized Intensity",
                      template="plotly_white",
                      legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"))
    
    # Create color plots if we have results
    if show_results and st.session_state.simulation_results is not None:
        # Create figures for both field and spin wave
        fig_field = go.Figure()
        fig_spin = go.Figure()
        
        # Calculate the field and spin wave intensities
        field_intensity = np.abs(st.session_state.simulation_results)**2
        spin_intensity = np.abs(st.session_state.simulation_results_spin)**2
        
        # Add the field intensity plot
        fig_field.add_trace(go.Heatmap(
            z=field_intensity,
            x=t_grid,
            y=z_grid,
            colorscale='Reds',
            colorbar=dict(title='|E|²', len=0.45, y=0.5)
        ))
        
        # Add the spin wave intensity plot
        fig_spin.add_trace(go.Heatmap(
            z=spin_intensity,
            x=t_grid,
            y=z_grid,
            colorscale='Blues',
            colorbar=dict(title='|S|²', len=0.45, y=0.5)
        ))
        
        # Add semi-transparent rectangles to mark the medium in both plots
        for fig_plot in [fig_field, fig_spin]:
            fig_plot.add_shape(
                type="rect",
                x0=t_grid[0],
                y0=0,
                x1=t_grid[-1],
                y1=L,
                line=dict(width=2, color="white"),
                fillcolor="rgba(0, 0, 0, 0.3)"
            )
            
            # Add label for the medium
            fig_plot.add_annotation(
                x=t_grid[0] + 0.5 * (t_grid[-1] - t_grid[0]),  # Center horizontally
                y=L/2,
                text="Atomic ensemble",
                showarrow=False,
                font=dict(color="black", size=12, family="Arial Black")
            )
            
            fig_plot.update_layout(
                xaxis_title="Time [μs]",
                yaxis_title="Position [cm]",
                template="plotly_white",
                margin=dict(l=0, r=0, t=30, b=0)  # Reduce margins for side-by-side layout
            )
        
        # Set titles for each plot
        fig_field.update_layout(title="Field Intensity |E|²")
        fig_spin.update_layout(title="Spin Wave Intensity |S|²")
        
        return fig, (fig_field, fig_spin)
    else:
        return fig, None

# -------------------- Simulation Function --------------------
def simulate(Omega_func, case_name="EIT"):
    E_z = np.zeros((Nz, Nt), dtype=complex)
    P_z = np.zeros((Nz, Nt), dtype=complex)
    S_z = np.zeros((Nz, Nt), dtype=complex)
    
    # Initialize field at the start of vacuum region
    E_z[0, :] = Ein

    def rhs(t, y, E_interp, in_medium=True):
        P, S = y[0] + 1j*y[1], y[2] + 1j*y[3]
        E = E_interp(t)
        if in_medium:
            Omega_val = Omega_func(t)
            dP = -(1 + 1j * Delta) * P + 1j * np.sqrt(d) * E + 1j * Omega_val * S
            dS = 1j * np.conj(Omega_val) * P
        else:
            # In vacuum, no atomic response
            dP = 0
            dS = 0
        return [dP.real, dP.imag, dS.real, dS.imag]

    # Create progress bar with color based on case
    bar_color = "green" if case_name == "Slow light" else "orange"
    progress_text = f"Running {case_name} simulation..."
    progress_bar = st.sidebar.progress(0.0, text=progress_text)
    
    # Propagate through the entire grid
    for i in range(Nz - 1):
        # Determine if we're in the medium
        in_medium = (z_grid[i] >= 0) and (z_grid[i] <= L)
        
        E_current = E_z[i, :]
        E_interp = interp1d(t_grid, E_current, kind='cubic', fill_value="extrapolate")
        y0 = [0.0, 0.0, 0.0, 0.0]
        
        # Solve atomic equations
        sol = solve_ivp(rhs, [t_grid[0], t_grid[-1]], y0, t_eval=t_grid, 
                       args=(E_interp, in_medium), method='RK23')
        
        P_next = sol.y[0] + 1j * sol.y[1]
        S_next = sol.y[2] + 1j * sol.y[3]
        
        # Update fields
        P_z[i + 1, :] = P_next
        S_z[i + 1, :] = S_next
        
        if in_medium:
            # In medium: full EIT dynamics
            E_z[i + 1, :] = E_z[i, :] + 1j * np.sqrt(d) * dz * P_next
        else:
            # In vacuum: free propagation
            E_z[i + 1, :] = E_z[i, :]
        
        if i % 10 == 0:
            progress = i / (Nz - 1)
            progress_bar.progress(progress, text=f"{progress_text} {int(progress*100)}%")

    progress_bar.progress(1.0, text=f"{case_name} simulation complete!")
    return E_z, S_z

# -------------------- Run Simulation --------------------
current_params = (L, d, Delta, Omega_const, sigma, t0, control_off, control_on, Nz, Nt, t_max)

# Check if parameters have changed
if st.session_state.last_params != current_params:
    st.session_state.simulation_results = None
    st.session_state.simulation_results_on = None
    st.session_state.simulation_results_spin = None
    st.session_state.last_params = current_params

# Create placeholders for the plots
time_plot_placeholder = st.empty()
plots_placeholder = st.empty()

# Always show the preview plot first
fig, _ = create_plot(show_results=False)
time_plot_placeholder.plotly_chart(fig, use_container_width=True, key="time_evolution_preview")

if run_simulation:
    # Run first simulation (Slow light case)
    E_z_on, S_z_on = simulate(lambda t: Omega_const, "Slow light")
    st.session_state.simulation_results_on = E_z_on
    st.session_state.simulation_results_spin_on = S_z_on
    
    # Update temporal plot with slow light results
    fig, _ = create_plot(show_results=True)
    time_plot_placeholder.plotly_chart(fig, use_container_width=True, key="time_evolution_slow")
    
    # Run second simulation (Memory case)
    E_z, S_z = simulate(Omega_t, "Memory")
    st.session_state.simulation_results = E_z
    st.session_state.simulation_results_spin = S_z
    
    # Update plot with both results and show color plots
    fig, plots = create_plot(show_results=True)
    time_plot_placeholder.plotly_chart(fig, use_container_width=True, key="time_evolution_memory")
    if plots is not None:
        fig_field, fig_spin = plots
        # Create two columns for the color plots
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_field, use_container_width=True, key="field_memory")
        with col2:
            st.plotly_chart(fig_spin, use_container_width=True, key="spin_memory")
