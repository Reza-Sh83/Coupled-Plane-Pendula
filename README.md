# N Parallel-Coupled Plane Pendula Simulation

This project simulates the dynamics of **N Parallel-Coupled Plane Pendula** using both analytical and numerical methods. It is designed to help users explore the behavior of coupled oscillatory systems and visualize the normal modes, angular frequencies, and overall motion through interactive plots and animations.

## Overview

The simulation models a set of pendula connected in parallel by springs. It covers:
- **Derivation of the Equations of Motion:** Starting from the small-angle approximation, the system is represented by a set of coupled differential equations.
- **Eigenvalue Analysis:** Calculation of normal modes and angular frequencies by solving the eigenvalue problem.
- **Numerical Solution:** Implementation of the Runge-Kutta method (via `solve_ivp` from SciPy) to obtain the time evolution of the system.
- **Visualization:** Use of Matplotlib for plotting individual pendulum motions, the superposition of modes, and dynamic animations to illustrate how energy transfers between modes.

## Features

- **User Input:** Customize gravitational acceleration, pendulum length, number of masses, initial positions, velocities, and spring constants.
- **Analytical & Numerical Methods:** Combines eigenvalue analysis with numerical integration to display both the underlying theory and the computed solution.
- **Interactive Animation:** The animation shows both the individual normal modes and the overall motion of the coupled pendula.
- **Visualization of Results:** Plots illustrate the motion of each pendulum, the combined displacement, and the impact of changing system parameters.

## Installation

Ensure you have Python 3 installed along with the following packages:
- **NumPy**
- **SciPy**
- **Matplotlib**
- **IPython** (for enhanced display and interactivity)

You can install these dependencies using pip:

```bash
pip install numpy scipy matplotlib ipython
```

## Usage

1. **Run the Script:**
   - Open your terminal or command prompt.
   - Navigate to the directory containing the project files.
   - Run the simulation with:
     ```bash
     python Coupled Plane Pendula.py
     ```

2. **Input Parameters:**
   - When prompted, enter the desired values for gravitational acceleration, pendulum length, number of masses, and the respective initial conditions.
   - If you press Enter without input, the script uses default values.

3. **Visualization and Animation:**
   - The script generates several plots:
     - Time-series plots for each mass.
       <img src="Solutions of the Coupled System for 2 Variables.png" width="450">
     - Combined motion plots showcasing the superposition of normal modes.
       <img src="Motion of 2 Coupled Masses Using Normal Modes.png" width="450">
     - Subplots for individual normal modes with corresponding angular frequencies.
       <img src="Each Normal Modes.png" width="450">
   - An animation is created to visually represent the movement of the masses over time. If running in a Jupyter Notebook or Canvas environment that supports HTML display, the animation will be embedded.

## Contributing

Contributions are welcome! If you'd like to suggest improvements or new features, feel free to fork the repo and open a pull request.

---

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- Inspired by classical mechanics and coupled oscillation theory.
- The project leverages numerical methods from the SciPy library.
- Visualization techniques are implemented using Matplotlib.

