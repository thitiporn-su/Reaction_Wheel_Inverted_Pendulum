import numpy as np
import matplotlib.pyplot as plt
# from numpy import sin, cos, pi
import time
import math

# Define system matrices
A = np.array([[0, 1, 0],          
              [99.67, 0, 7.78],
              [0, 0, -82.64]])
B = np.array([[0],
              [-74.701],
              [792.56]])
C = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
D = np.array([[0], [0],[0]])

# PID controller parameters
kp_pos = 1.6
ki_pos = 0.08
kd_pos = 0.0


dt = 0.001  # time step
t_sim = np.arange(0,100, dt)  # time vector


# Initial conditions
x0 = np.array([-90, 0, 0])  # initial state
u = np.zeros_like(t_sim)  # input signal (initially zero)

setpoint_pos = np.array([0, 0, 0])

# PID controller variables
prev_error_pos = setpoint_pos[0] - x0[0]
integral_pos = 0.0

x0_history = []  # Store x0 values for plotting
error_pos_history = []
u_pos_history = []
Stop_time = 0
m = -1
round = 0
a1 = 1.5
fig, ax = plt.subplots()

# Simulation loop
for i in range(len(t_sim)):
    x0_history.append(x0.copy())  
    error_pos_history.append(prev_error_pos) 
    error_pos = setpoint_pos[0] - x0[0] 
    
    #Store integral term
    integral_pos += error_pos * dt
    #Calculate pid 
    u_pos = kp_pos * error_pos + ki_pos * integral_pos + kd_pos * (error_pos - prev_error_pos) / dt
    #set direction of motor
    u_pos = m*u_pos
    u_pos_history.append(u_pos) 
    #input to plant
    u[i] = u_pos 
    #limit Voltage input 
    if (u[i] > 24 ):
        u[i] = 24 
    elif (u[i] < -24 ):
        u[i] = -24 
    #break system at set point and clear integral term
    if (error_pos <= 0.5 and error_pos >= -0.5 and round == 0):
        x0[1] = 0
        x0[2] = 0
        integral_pos = 0
        round = 1
    #calculate next state
    x_dot = np.matmul(A, x0) + np.matmul(B, np.array([u[i]]))
    #Switch direction of cube 
    if(x_dot[2] >0 and setpoint_pos[0] >= 0):
        x_dot[1] = x_dot[1]*-1
    if(x_dot[2] <0 and setpoint_pos[0] <= 0):
        x_dot[1] = x_dot[1]*-1
    # Update state using Euler method
    x0 = x0 + x_dot * dt
    x0 = x0.flatten() 
    # Update previous error
    prev_error_pos = error_pos
    
    # Visualization
    x = a1 * math.sin((x0[0])*2*math.pi/360)
    y = a1 * np.cos((x0[0])*2*np.pi/360)
    ax.clear()
    
    #find position at wheel
    Pwheel = np.array([0])
    for i in range(len(x0_history)):
        Pwheel = np.append(Pwheel, Pwheel[i] + x0_history[i][2]*dt) 
    Wx = np.cos(Pwheel[i])
    Wy = np.sin(Pwheel[i])
    ax.plot([0, x], [0, y], 'o-')
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    plt.plot( x,y, color = 'red', marker = 'o') #Draw center
    r = 3
    angles = np.linspace(0 , 2 * np.pi, 100) 
    xs = np.cos(angles)
    ys = np.sin(angles)
    plt.plot(xs+x, ys+y, color = 'green')
    plt.xlim(-r, r)
    plt.ylim(-r, r)
    plt.gca().set_aspect('equal')
    plt.plot([x, Wx +x], [y, Wy+y], color = "black")
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("2R Robot with End-effector Velocity")
    plt.pause(0.01) 
