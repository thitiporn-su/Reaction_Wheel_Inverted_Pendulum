# **Reaction Wheel Balanced Inverted Pendulum - Control System Simulation**

## Overview
This repository contains code for simulating the control system of a Reaction Wheel Balanced Inverted Pendulum. The system's dynamics are modeled using state-space equations, and a PID controller is implemented to control its position.

## Contents
- `README.md` : Introduction and instructions.
- `Reaction Wheel Simulation.py` : Python code implementing the Continuous simulation.
- `Characteristic of Reaction Wheel.ipynb` : Python code implementing the Discrete simulation.

## Usage

1. **Clone the Repository :**
   
   Open Command Prompt and navigate to the directory where you want to clone the repository :

    ```
    git clone https://github.com/thakdanaiza/Project_FRA333_66.git
    cd reaction-wheel-pendulum
    ```
    
2. **Install Requirement library :**
   
    Ensure you have the necessary Python libraries by installing them via :
    
    For NumPy :
    ```
    pip install numpy
    ```
    
    For Matplotlib :
    ```
    pip install matplotlib
    ```
    
    If you're using Anaconda, you can also install these libraries via conda :
    
    ```
    conda install numpy matplotlib
    ```

4. **Run the Simulation:**

    Execute the simulation code to observe the control system in action :

    ```
    Reaction Wheel Simulation.py
    ```
    ```
    Characteristic of Reaction Wheel.ipynb
    ```

## Methodology

### Inverted Pendulum

![image](https://github.com/thakdanaiza/Project_FRA333_66/assets/61357514/26f7ee69-38af-46eb-be5f-939cf32332b2)

Using the Newton Euler approach and under some predefined assumptions, the equations
of the motion of the pendulum can described by : 

$$J\ddot{\theta} = mgL\sin(\theta)-\mu L\cos(\theta)$$

Where :
- J represents the moment of inertia of the pendulum.
- θ¨ represents the angular acceleration of the pendulum.
- m is the mass of the pendulum bob.
- g is the acceleration due to gravity.
- L is the length of the pendulum arm.
- θ represents the angular displacement of the pendulum from its equilibrium position.
- μ is the coefficient of friction.

### Kinetic Energy (K) :

   $$\ K = \frac{J_{p}\dot{\theta}^{2}}{2} \$$

   - K represents the kinetic energy of the pendulum system.
   - Jp is the total moment of inertia of the pendulum.
   - θ˙ is the angular velocity of the pendulum.
This equation demonstrates that the kinetic energy is calculated based on the rotational motion of the pendulum. It's proportional to the square of the angular velocity and the total moment of inertia of the system.

### Potential Energy (V) :

   $$\ V = mgL(\cos(\theta) - 1) \$$
   
   - V represents the potential energy of the pendulum system.
   - m is the mass of the pendulum bob.
   - g is the acceleration due to gravity.
   - L is the length of the pendulum.
   - θ is the angle the pendulum makes with the vertical.
This equation accounts for the gravitational potential energy of the pendulum. It's dependent on the height of the pendulum bob above the reference level where `θ = 0` and the gravitational acceleration g.

The total mechanical energy (E) of the system, which was previously mentioned as :

$$E = K + V$$

$$E = \frac{J_{p}\dot{\theta}^{2}}{2} + mgL(\cos(\theta) - 1) \$$

This energy expression combines both the kinetic and potential energy terms, encapsulating the total energy of the pendulum system at any given position or angle θ.


### State-Space Equations

The system dynamics are represented by the state-space equations :

$$\begin{align*}
\dot{X} &= AX + Bu(t) \\
\\
y &= CX + Du(t)
\end{align*}$$

### Explanation
- **State Equation X_dot = AX + Bu(t) :**
  - Describes how the system's states change over time.
  - A determines the rate of change of the state variables.
  - B influences how the input u(t) affects the state variables.

- **Output Equation y = CX + Du(t) :**
  - Defines the system's output in terms of its internal state and input.
  - C maps the state vector X to the output y(t).
  - D describes any direct influence of the input u(t) on the output y(t) without passing through the state variables.

These equations are fundamental in describing the continuous-time behavior of the system in terms of its internal states, inputs, and outputs.

Where, X is the state of this state space system. Three states are the angle around the balancing edge (θp), the angular velocity around the edge (ωp), and angular velocity of the reaction wheel (ωr)

$$\
X = \begin{bmatrix}
{\theta}_p \\
{\omega}_p \\
{\omega}_r \\
\end{bmatrix}
\$$

$${U}={V}$$

### Equations that defines the states :

$$\\dot{{\theta}}_p = {\omega}_p\$$

This equation represents the rate of change of the angle around the balancing edge (θp). It states that the time derivative of θp (θ˙p) is equal to the angular velocity (ωp).

$$\\dot{{\omega}}_p = \frac{{mgL}}{{J}_t}{\theta}_p - \frac{T}{{J}_t}\$$

Describes the rate of change of the angular velocity around the edge (ωp). It's influenced by two terms:
     - The first term represents the effect of gravity (mgL) and the angle (θp) on the change in angular velocity.
     - The second term is the torque (T) applied divided by the moment of inertia (Jt). It's represents the effect of the applied torque on the change in angular velocity.
     
$$\\dot{{\omega}}_r = \frac{T}{{J}_r}\$$

Indicates the rate of change of the angular velocity of the reaction wheel (ωr). It's determined by the applied torque (T) divided by the moment of inertia of the reaction wheel (Jr).

### Motor Model :

Motor can be modeled as a first order system,

$$\frac{{\omega}_r}{{V}} = \frac{k}{{\tau}s+1}$$

This equation describes the relationship between the angular velocity of the reaction wheel (omega_r) and the applied voltage (V) through a transfer function. It's a standard representation of a first-order system with time constant (τ) and gain (k).


$$\\dot{{\omega}}_r = -\frac{1}{\tau}{\omega}_r + \frac{k}{\tau}{V}$$

1. **First Term :**
   - This term represents the decay or damping effect on the angular velocity. It's proportional to the current angular velocity \(\omega_r\) and inversely proportional to the time constant \(\tau\). It indicates that the angular velocity will tend to decrease over time if no input voltage is applied (V = 0).

2. **Second Term :**
   - This term signifies the influence of the applied voltage on the angular velocity. It's proportional to the input voltage (V) and scaled by the gain (k) and time constant (τ). Increasing the applied voltage (V) will increase the angular velocity (omega_r), and the rate of change will depend on the gain (k) and time constant (τ).

This equation captures how the motor's angular velocity responds to changes in the input voltage, providing insight into the motor's dynamics and aiding in control system design for tasks like speed regulation or trajectory tracking.

Where :
- Jt - Moment of inertia of the reaction wheel
- Jm - Moment of inertia of the cube + motor + reaction wheel
- m - Total mass (cube + motor + wheel)
- L - Center of gravity of the entire pendulum system from the pivot point
- I - Motor current
- k - Motor gain
- tau - Motor time constant
- τ - Current motor torque
- θp - Cube’s angle around an edge
- θr - Reaction wheel rotation angle
- ωp - Cube’s angular velocity around an edge
- ωr - Reaction wheels angular velocity
- g - Gravitational acceleration
- v - Motor input voltage

So the state space model of my system is as below

$$A = \begin{bmatrix}
0 & 1 & 0 \\
\frac{mgL}{{J}_t} & 0 & \frac{{J}_r}{{J}_t{\tau}} \\
0 & 0 & -\frac{1}{\tau}
\end{bmatrix}
$$

$$B = \begin{bmatrix}
0 \\
-\frac{{{J}_r}{k}}{{{J}_t}{\tau}} \\
\frac{k}{\tau}
\end{bmatrix}
$$


### PID Controller

![image](https://github.com/thakdanaiza/Project_FRA333_66/assets/61357514/e065ad29-cef2-4c87-b7ff-7b31f6057abe)

PID (Proportional-Integral-Derivative) control is a feedback control mechanism widely used in engineering and control systems to achieve and maintain a desired setpoint. It works by continuously calculating an error value as the difference between a desired setpoint and the current state of a system. This error value is then used to adjust the system's control input to minimize the error and maintain stability.

### Components of PID Control :

1. **Proportional (P) term :** This term responds in proportion to the current error. It drives the control output based on the current magnitude of the error. Adjusting the proportional constant \(K_p\) changes the proportional influence on the output.

2. **Integral (I) term :** The integral term considers the accumulation of past errors over time. It helps in eliminating any residual error that persists over time by integrating the error values. The integral constant \(K_i\) determines how aggressively the controller reacts to past error values.

3. **Derivative (D) term :** This term anticipates future errors by considering the rate of change of the error. It helps dampen rapid changes in the error, preventing overshoot or oscillations in the system. The derivative constant \(K_d\) controls the impact of the rate of change of the error.

### PID Control Equation :  

$$ \text{PID Output} = K_p \cdot \text{error} + K_i \cdot \int \text{error} \cdot dt + K_d \cdot \frac{d(\text{error})}{dt} $$

Where:
- PID Output is the output of the controller.
- K_p,  K_i, and K_d are the proportional, integral, and derivative constants, respectively.
- error is the difference between the setpoint (desired value) and the process variable (measured value).
- Integrate error represents the integral of the error with respect to time.
- Derivative error represents the derivative of the error with respect to time.

We defined specific values for (Kp), Ki), and (Kd) as 1.6, 0.08, and 0.0, respectively. These values determine the relative impact of each term in the controller equation. For instance, a higher (Kp) value would amplify the proportional response, while a non-zero (Ki) value would address accumulated errors over time.

## Simulation
### Discrete Time

![image](https://github.com/thakdanaiza/Project_FRA333_66/assets/61357514/ac9e64ba-52f4-470d-bbfc-b195f8dc880c)

The discrete-time simulation runs over a specified duration, where the system's state dynamics are calculated at discrete time steps using the PID controller. The simulation results in three plots :
- Arm Theta : Shows the arm's physical movement as it changes angle over time.
- Arm Omega : The arm motor starts slow and then ramps up its speed.
- Wheel Omega : Starts at high speed and gradually slows down, Indicates the wheel motor is receiving power and slowing down from initial acceleration.

### Continuous-Time

https://github.com/thitiporn-su/Reaction_Wheel_Inverted_Pendulum/assets/61357514/6fc89576-d09a-411e-815f-6fdda977e95e

To perform a continuous-time simulation :
1. **Derivation of Continuous-Time Equations :** Derive the continuous-time equations from the discrete-time model, considering differential equations.
2. **Integration Method :** Select a suitable numerical integration method (Euler's method, Runge-Kutta methods, etc.) to simulate the continuous-time behavior.
3. **Simulation and Visualization :** Implement the continuous-time simulation and visualize the system's behavior over time.

## Contribution

Feel free to contribute by forking the repository and creating pull requests with improvements or additional features.

## Reference

- http://ise.ait.ac.th/wp-content/uploads/sites/57/2020/12/Development-and-Motion-Control-of-Single-Axis-Reaction-Wheel-Based-Cubic-Robot.pdf
- https://ri.itba.edu.ar/server/api/core/bitstreams/50e43ae0-2e69-4bde-8dd1-f1158305a291/content
