# %% [markdown]
# # N Parallel-Coupled Plane Pendula
# 
# This notebook explores the dynamics of two coupled plane pendula connected by a spring. Using both analytical and numerical methods, we examine the normal modes, angular frequencies, and total motion of the system. We'll also include visualizations to help understand the motion.
# 
# ## Objectives
# 1. Derive and solve the equations of motion.
# 2. Analyze the system's normal modes and angular frequencies.
# 3. Visualize the motion through plots and animations.
# 4. Gain insight into the physical interpretation of coupled oscillations.

# %%
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from IPython import display

# Get parameters from the user
g = float(input("Enter gravitational acceleration (m/s^2) [default 9.8]: ") or 9.8)
l = float(input("Enter length (m) [default 1.0]: ") or 1.0)
n = int(input("Enter number of masses (e.g., 2) [default 2]: ") or 2)

# Display entered values for verification
print(f"Parameters set:\n- Gravitational acceleration: {g}\n- Length: {l}\n- Number of masses: {n}")

# Initialize lists to store initial velocities, positions, masses
initial_positions = []
initial_velocities = []
masses = []

# Get initial positions, velocities, and masses for each particle
for i in range(n):
    position = float(input(f"Enter initial position for mass {i+1} [default 0]: ") or 0)
    velocity = float(input(f"Enter initial velocity for mass {i+1} [default 0]: ") or 0)
    mass = float(input(f"Enter mass for particle {i+1} [default 1.0]: ") or 1.0)
    
    initial_positions.append(position)
    initial_velocities.append(velocity)
    masses.append(mass)

# Convert lists to numpy arrays for further processing
initial_positions = np.array(initial_positions)
initial_velocities = np.array(initial_velocities)
masses = np.array(masses)

# Display entered values for verification
print(f"Initial positions: {initial_positions}")
print(f"Initial velocities: {initial_velocities}")
print(f"Masses: {masses}")

# Define spring constants, one for each pair of adjacent masses
spring_constants = []
for i in range(n - 1):
    spring_constant = float(input(f"Enter spring constant for the spring between particle {i+1} and particle {i+2} [default 1.0]: ") or 1.0)
    spring_constants.append(spring_constant)

# Display entered spring constants
spring_constants = np.array(spring_constants)
print(f"Spring constants: {spring_constants}")

# Initial conditions for the system (positions and velocities)
initial_val = np.concatenate([initial_positions, initial_velocities])

# Get time parameters from the user
t_start = float(input("Enter start time (e.g., 0) [default 0]: ") or 0)
t_end = float(input("Enter end time (e.g., 20) [default 20]: ") or 20)

# Define time span and evaluation points
t_span = (t_start, t_end)
num_points = int((t_end - t_start) * 100)
t_eval = np.linspace(*t_span, num_points)

# Display entered values for verification
print(f"Time parameters set:\n- Start time: {t_start}\n- End time: {t_end}\n- Number of time points: {num_points}")

# %% [markdown]
# ## System of Equations
# 
# The equations of motion for the coupled pendula system are derived considering the forces acting on each mass. Assuming small oscillations ($ \sin \theta \approx \theta $), the governing equations are:
# 
# $$
# m \ddot{x}_i + k(x_i - x_{i-1}) - k(x_{i+1} - x_i) + \frac{mg}{l} x_i = 0
# $$
# 
# ### Coupling Matrix
# Rewriting the system in matrix form:
# 
# $$
# \mathbf{\ddot{x}} = \mathbf{MK^{-1}} \mathbf{x}
# $$
# 
# where the coupling matrix $$ \mathbf{MK^{-1}} $$ is defined as:
# 
# $$
# \mathbf{MK^{-1}} =
# \begin{bmatrix}
# \frac{g}{l} + \frac{k}{m} & -\frac{k}{m} & 0 & \cdots & 0 \\
# -\frac{k}{m} & \frac{g}{l} + \frac{2k}{m} & -\frac{k}{m} & \cdots & 0 \\
# 0 & -\frac{k}{m} & \frac{g}{l} + \frac{2k}{m} & \cdots & -\frac{k}{m} \\
# \vdots & \vdots & \vdots & \ddots & \vdots \\
# 0 & \cdots & 0 & -\frac{k}{m} & \frac{g}{l} + \frac{k}{m}
# \end{bmatrix}
# $$
# 
# ### Assumptions
# 1. Small oscillations $( \sin \theta \approx \theta $).
# 2. The masses and spring constants are identical for simplicity.
# 
# ## Step-by-Step Calculation of Eigenvalues and Eigenvectors
# 
# ### Understanding the Matrix Equation
# To find the normal modes and angular frequencies of the coupled system, we solve the eigenvalue problem:
# 
# $$
# \mathbf{MK^{-1}} \mathbf{A} = \mathbf{\Lambda} \mathbf{A}
# $$
# 
# Where:
# - $ \mathbf{MK^{-1}} $: The coupling matrix.
# - $ \mathbf{A} $: A matrix whose columns are eigenvectors.
# - $ \mathbf{\Lambda} $: A diagonal matrix containing eigenvalues.
# 
# Each eigenvalue $ \lambda $ corresponds to the square of the angular frequency ($ \omega^2 $):
# 
# $$
# \lambda = \omega^2
# $$
# 
# The eigenvectors $ \mathbf{A} $ describe the relative motion (or "shape") of each normal mode.

# %%
# Define the matrix representing the coupled system
MK = np.zeros((n, n))
for i in range(n):
    if i == 0:
        # Handle the first mass (no spring before it)
        MK[i, i] = (g/l) + (spring_constants[i]) / masses[i]
        if i < n-1:
            MK[i, i+1] = -spring_constants[i] / masses[i]  # Coupling with the next mass
    elif i == n-1:
        # Handle the last mass (no spring after it)
        MK[i, i] = (g/l) + (spring_constants[i-1]) / masses[i]
        MK[i, i-1] = -spring_constants[i-1] / masses[i]  # Coupling with the previous mass
    else:
        # Handle the middle masses (coupled with both neighbors)
        MK[i, i] = (g/l) + (spring_constants[i-1] + spring_constants[i]) / masses[i]
        MK[i, i-1] = -spring_constants[i-1] / masses[i]  # Coupling with the previous mass
        MK[i, i+1] = -spring_constants[i] / masses[i]  # Coupling with the next mass


# Solve the eigenvalue problem to find normal modes
w2, A = np.linalg.eig(MK)  # Eigenvalues and eigenvectors
freq = np.sqrt(w2)  # Frequencies (sqrt of eigenvalues)

# Amplitudes of the normal modes
amp = np.dot(np.linalg.inv(A), initial_positions)

# Construct normal modes
modes = np.zeros((n, len(t_eval)))
for i in range(n):
    modes[i, :] = amp[i] * np.cos(freq[i] * t_eval)  # Normal modes as a function of time

# Reconstruct the motion of each mass from the normal modes
X_sum = np.dot(A, modes)

# %% [markdown]
# ### Steps to Calculate Eigenvalues and Eigenvectors
# 1. **Construct the Coupling Matrix $ \mathbf{MK^{-1}} $**:
#    - Each diagonal element represents the restoring force proportional to displacement.
#    - The off-diagonal elements describe the coupling effect between adjacent pendula.
# 
# 2. **Solve the Eigenvalue Problem**:
#    - Use numerical methods (e.g., `np.linalg.eig`) to compute eigenvalues and eigenvectors.
#    - Extract angular frequencies as $ \omega = \sqrt{\lambda} $.
# 
# 3. **Interpret the Results**:
#    - The eigenvalues determine the oscillation frequencies.
#    - The eigenvectors describe the motion pattern of each normal mode.
# 
# ### Final Expression for Normal Modes
# The displacement of the system in terms of normal modes is:
# 
# $$
# \mathbf{X}(t) = \sum_{i=1}^{n} A_i \cos(\omega_i t + \phi_i)
# $$
# 
# Where:
# - $ A_i $: The amplitude of the $ i $-th mode.
# - $ \omega_i $: The angular frequency of the $ i $-th mode.
# - $ \phi_i $: The phase of the $ i $-th mode.

# %% [markdown]
# ## Numerical Solution
# 
# To solve the equations of motion numerically, we use the Runge-Kutta method implemented in `solve_ivp` from SciPy.
# 
# ### State Vector
# We define the state vector as:
# 
# $$
# \mathbf{y} = [\mathbf{x}, \mathbf{\dot{x}}]
# $$
# 
# The second-order ODEs are transformed into a system of first-order ODEs:
# 
# $$
# \dot{\mathbf{y}} = 
# \begin{bmatrix}
# \mathbf{\dot{x}} \\
# \mathbf{MK^{-1}} \mathbf{x}
# \end{bmatrix}
# $$
# 
# ### Numerical Method
# 1. Use the `solve_ivp` function to integrate the system over the given time span.
# 2. Specify the initial conditions for positions and velocities.
# 3. Evaluate the solution at discrete time points for visualization.
# 
# This method provides an approximate solution to the coupled pendula's motion.
# 

# %%
def coupled_odes(t, y):
    # Split y into positions (x) and velocities (v)
    x = y[:n]
    v = y[n:]
    
    # Define the coupling matrix
    MK = np.zeros((n, n))
    for i in range(n):
        if i == 0:
            # Handle the first mass (no spring before it)
            MK[i, i] = (g/l) + (spring_constants[i]) / masses[i]
            if i < n-1:
                MK[i, i+1] = -spring_constants[i] / masses[i]  # Coupling with the next mass
        elif i == n-1:
            # Handle the last mass (no spring after it)
            MK[i, i] = (g/l) + (spring_constants[i-1]) / masses[i]
            MK[i, i-1] = -spring_constants[i-1] / masses[i]  # Coupling with the previous mass
        else:
            # Handle the middle masses (coupled with both neighbors)
            MK[i, i] = (g/l) + (spring_constants[i-1] + spring_constants[i]) / masses[i]
            MK[i, i-1] = -spring_constants[i-1] / masses[i]  # Coupling with the previous mass
            MK[i, i+1] = -spring_constants[i] / masses[i]  # Coupling with the next mass
            
    # Compute second derivatives
    dv_dt = -MK @ x
    dx_dt = v  # First derivatives are velocities
    
    return np.concatenate([dx_dt, dv_dt])

# Solve the system
solution = solve_ivp(fun=coupled_odes,
                     t_span=t_span,
                     y0=initial_val,
                     method='RK45',
                     t_eval=t_eval)

# Extract results
t = solution.t
x = np.zeros((n, len(t_eval)))
for i in range(n):
    x[i,:] = solution.y[i]  # Position

# %% [markdown]
# ## Visualizing the Motion
# 
# ### Individual Normal Modes
# Each normal mode represents a unique motion pattern at a specific frequency. By plotting these modes, we can observe the isolated behavior of each mode.
# 
# ### Reconstructed Motion
# The motion of the system is a combination of all normal modes. This is expressed as:
# 
# $$
# \mathbf{X}(t) = \mathbf{A} \cos(\mathbf{\omega} t + \phi)
# $$
# 
# 
# ### Plots
# - **Individual Modes:** Show the isolated oscillations of each normal mode.
# - **Total Motion:** Display the reconstructed motion of the system, which is the sum of all normal modes.
# - **Coupled Motion:** Illustrate how the coupled pendula move over time.
# 
# ### Insights
# - Normal modes reveal the intrinsic frequencies and patterns of the system.
# - The reconstructed motion demonstrates the interplay between modes, leading to periodic and complex behavior.
# 

# %%
# Plotting the results
plt.figure(figsize=(10, 6))
for i in range(n):
    plt.plot(t, x[i], label=f"x_{i+1}(t)", linestyle='-.')
plt.plot(t, np.sum(x, axis=0), label="Sum", linestyle='solid')
plt.title(f"Solutions of the Coupled System for {n} Variables")
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
for i in range(n):
    plt.plot(t_eval, x[i,:], label=f'$x_{i+1}$')    # solutions found by brute-force
    plt.plot(t_eval, X_sum[i, :], 'k:')  # Motion of each mass
    
plt.ylim(-1.5, 1.5)
plt.xlabel(r'$t$')
plt.ylabel(r'$x_i(t)$')
plt.title(f"Motion of {n} Coupled Masses Using Normal Modes")
plt.legend(ncol=2)
plt.grid()
plt.show()

fig, axes = plt.subplots(1, n, figsize=(18, 4))  # Create subplots dynamically based on n
for mode_index in range(n):  # Iterate over all modes
    for i in range(n):  # For each mode, plot all variables x_i
        x_i = A[i, mode_index] * modes[mode_index, :]
        axes[mode_index].plot(t_eval, x_i, label=f'$x_{i+1}$')
    axes[mode_index].set_ylim(-1, 1)
    axes[mode_index].set_xlabel(r'$t$')
    axes[mode_index].set_ylabel(r'$x_i$')
    axes[mode_index].legend(ncol=2)
    axes[mode_index].set_title(rf'Mode {mode_index+1}: $\omega_{mode_index+1} = {freq[mode_index]:.3f}$')

plt.tight_layout()
plt.show()

# %% [markdown]
# # Analysis of Coupled Plane Pendula
# 
# ## 1. Frequency Analysis
# The natural frequencies of the system are computed as the square root of the eigenvalues of the matrix representing the coupled system. These frequencies depend on:
# - **Mass ($m$)**: Higher mass leads to lower frequencies.
# - **Spring Constant ($k$)**: Higher spring constant results in higher frequencies.
# - **Gravitational Acceleration ($g$)**: Increasing $g$ raises the frequencies.
# 
# ### Example Frequencies:
# For the provided setup, the system has the following natural frequencies:
# $$\omega_1 = \text{value}, \; \omega_2 = \text{value}, \; \dots$$
# 
# ---
# 
# ## 2. Normal Modes
# Each normal mode corresponds to a distinct pattern of motion for the coupled system:
# - **Mode 1**: All masses oscillate in phase (symmetric motion).
# - **Mode 2**: Adjacent masses oscillate out of phase (antisymmetric motion).
# 
# **Plots for Normal Modes:**
# - Mode 1: $\omega_1 = 3.435$
# - Mode 2: $\omega_2 = 3.130$
# 
# <center><img src="image1_x(t).png"/></center>
# <img src="image1_mode.png">
# 
# ---
# 
# ## 3. Effect of Changing Parameters
# 
# ### Changing the Mass
# - Increasing the mass decreases the frequencies, resulting in slower oscillations.
# - Decreasing the mass increases the frequencies, leading to faster oscillations.
# 
# **Example Plot: Effect of increasing $m$ by 50% (m1 = 1.5, m2 = 1.)**
# <center><img src="image2_x(t).png"/></center>
# <img src="image2_mode.png">
# 
# ### Changing the Spring Constant
# - Increasing the spring constant strengthens the coupling, resulting in faster oscillations.
# - Decreasing the spring constant weakens the coupling, leading to slower oscillations.
# 
# **Example Plot: Effect of increasing $k$ by 50% (k = 1.5)**
# <center><img src="image3_x(t).png"/></center>
# <img src="image3_mode.png">
# 
# ---
# 
# ## 4. Effect of Changing Initial Conditions
# 
# ### Changing Initial Position
# Altering the initial position redistributes energy between the normal modes, potentially amplifying or diminishing specific modes.
# 
# ### Changing Initial Velocity
# A larger initial velocity for one mass increases its amplitude, making the energy transfer between masses more pronounced.
# 
# **Example Plot: Effect of increasing initial velocity for $x_1$ ($v_1$ = 1.0)**
# <center><img src="image4_x(t).png"/></center>
# <img src="image4_mode.png">

# %% [markdown]
# ## Interactive Simulations
# 
# ### Animation Description
# To better visualize the system's behavior:
# 1. **Mode Animations:** Animate the motion of each normal mode separately.
# 2. **Combined Animation:** Show the total motion combining all modes.
# 
# ### Key Observations
# - Observe how each mode contributes to the overall motion.
# - Analyze how the system transitions between different configurations over time.
# 
# ### Diagram
# Below is a live animation of the system's behavior, including individual modes and the total motion.
# 
# The plots display:
# 1. The motion of each pendulum.
# 2. The superimposed contributions of all modes.
# 

# %%
matplotlib.rcParams['animation.embed_limit'] = 2**128
# Set up the figure
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-1, n + 2)
ax.set_ylim(-0.5, n + 1)
ax.axis('off')

# Add text for labels
ax.text(-1, 0.2, 'all modes', fontsize=10)
for mode_index in range(n):
    ax.text(-1, mode_index + 1.2, f'mode {mode_index + 1}', fontsize=10)

# Create plots for each mode and the total motion
plots = []
for mode_index in range(n):
    plots.append(ax.plot([], [], 'o-', label=f'Mode {mode_index + 1}')[0])
p_all, = ax.plot([], [], 'o-', label='All modes')  # Total motion plot

# Offset for spring visualization
offset = np.arange(n)

# Animation function
def animate(t):
    for mode_index in range(n):
        plots[mode_index].set_data(A[:, mode_index] * modes[mode_index, t] + offset, [mode_index + 1] * n)
    p_all.set_data(X_sum[:, t] + offset, [0] * n)

# Create the animation
mov = anim.FuncAnimation(fig, animate, frames=len(t_eval), interval=50)

# Display the animation
plt.title(f"Visualization of {n} Coupled Masses with Normal Modes")
plt.close()
display.HTML(mov.to_jshtml())

# %% [markdown]
# ### Explanation of the Plot Output:
# 
# #### 1. **Animation of Masses and Modes:**
# The main plot you're generating shows the oscillation of multiple masses that are coupled by springs, and it does so in the following way:
# 
# - **"All modes" plot (combined motion):**
#   - This plot represents the **total motion of all masses** in the system as a sum of their individual normal modes.
#   - Over time, you see how the positions of all masses change together. The total motion shows the **overall displacement** of each mass, as if all normal modes are interacting and contributing to the motion.
#   - The motion is oscillatory and reflects the fact that the system of coupled masses is undergoing **harmonic oscillations**, influenced by both the spring constants and masses.
# 
# - **Individual Mode Plots:**
#   - For each **mode (mode 1, mode 2, etc.)**, you see a separate plot that shows the **motion of the masses** as they oscillate in that particular mode.
#   - Each **mode** corresponds to a specific frequency and pattern of movement that is characteristic of the system's natural vibrations. In these plots:
#     - **Mode 1**: Typically, all masses move in unison (in-phase) with the same displacement, but possibly with different magnitudes.
#     - **Mode 2**: The masses might oscillate **out of phase**, meaning that some masses move in opposite directions. For example, the masses at the ends might move in one direction, while the masses in the middle move in the opposite direction.
#     - Higher modes (mode 3, mode 4, etc.) involve more complex patterns where the masses move with different phase shifts and amplitudes.
# 
# #### 2. **What You See in the Animation:**
# 
# - **Spring Visualization:**
#   - The masses are shown as points (denoted by 'o'), and these points move according to their oscillations. The springs between the masses create a visual connection between them, which helps you understand how the coupling works.
#   - As the animation plays, you see each mass moving back and forth. The **relative movement** between the masses reflects the **force transmission** through the springs. This motion is determined by the normal modes of vibration, which are essentially the natural ways the masses can move when disturbed.
# 
# - **Mode-specific Motion:**
#   - As each mode is displayed in the animation, you'll see different patterns of motion. For example:
#     - In **mode 1**, all masses might move in the same direction at the same time (or with slight variations if the masses have different spring constants or masses).
#     - In **mode 2**, the masses could move in alternating directions: the leftmost mass might move in one direction, the middle mass in the opposite direction, and the rightmost mass in yet another direction.
#     - Higher modes have more complex, often sinusoidal, oscillations where different parts of the system may oscillate with different amplitudes and phases.
# 
# - **Overall Behavior:**
#   - The combined motion (shown in the "All modes" plot) is the result of **adding up** the individual mode motions. The total motion reflects how the system behaves as a whole, considering all normal modes simultaneously. You can imagine it as a **superposition** of simple harmonic motions (each corresponding to a different mode).
# 
# #### 3. **Key Observations During Animation:**
# - The animation helps visualize the **evolution of the system over time**. You’ll notice that the masses oscillate in patterns that are dictated by the system’s natural frequencies.
# - The frequencies and amplitude relationships between the masses in different modes give you an insight into how the system behaves based on its physical properties (such as mass and spring constants).
#   
#   For example, if the masses have different spring constants (`k1, k2, ...`), the oscillations might appear more complex or irregular because each mode will involve different restoring forces acting on each mass.
# 
# #### 4. **Interpretation of the "All modes" Plot:**
# - The plot that shows the motion of all modes together represents the **combined oscillation** of the system. The total displacement of the masses at any given time is a result of all the modes contributing to the system’s motion.
#   - **At certain moments in time**, you may see the masses move more strongly in some directions, indicating that those particular normal modes have higher amplitudes at those times.
#   - The **total motion** will often be a more complex oscillation compared to the simple oscillations in individual modes.
# 
# ### Conclusion:
# - The animation illustrates how the coupled masses oscillate, both individually and as a system. It shows the normal mode frequencies and how these modes contribute to the overall motion.
# - Each mode displays a unique pattern of oscillation, and the combined motion of all modes gives a comprehensive view of how the entire system behaves dynamically.
# 


