import math, copy
from stringprep import b3_exceptions # For mathematical operations and deep copying
import numpy as np # For numerical operations
import matplotlib.pyplot as plt # For plotting

x_train = np.array([1.0, 2.0]) # Training input data
y_train = np.array([300.0, 500.0]) # Training output data

# Function to compute the cost for linear regression
def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0

    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i])**2
    total_cost = cost / (2*m)

    return total_cost

# Function to compute the gradient for linear regression
def compute_gradient(x, y, w, b):
    """
    Computes the gradient for linear regression 
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters  
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
    """
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = f_wb - y[i]
        dj_db += dj_db_i
        dj_dw += dj_dw_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db


def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    """
    Performs gradient descent to fit w,b. Updates w,b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      x (ndarray (m,))  : Data, m examples 
      y (ndarray (m,))  : target values
      w_in,b_in (scalar): initial values of model parameters  
      alpha (float):     Learning rate
      num_iters (int):   number of iterations to run gradient descent
      cost_function:     function to call to produce cost
      gradient_function: function to call to produce gradient
      
    Returns:
      w (scalar): Updated value of parameter after running gradient descent
      b (scalar): Updated value of parameter after running gradient descent
      J_history (List): History of cost values
      p_history (List): History of parameter values

    """
    # Array to store cost J and w's at each iteration primarily for graphing later
    J_history = [] # Use a python list to save cost J at each iteration
    p_history = [] # Use a python list to save parameters w and b at each iteration
    b = b_in
    w = w_in

    for i in range(num_iters):
        # Calculate the gradient and update the parameters using gradient_function
        dj_dw, dj_db = gradient_function(x, y, w, b)

        # Update the parameters using equation w = w - alpha * dj_dw, b = b - alpha * dj_db
        b = b - alpha * dj_db
        w = w - alpha * dj_dw

        # Save cost J at each iration

        if i<100000: # prevent resource exhaustion 
            J_history.append(cost_function(x, y, w, b)) # save cost J at each iteration #append adds to the end of the list
            p_history.append([w,b]) # save w and b at each iteration
            # Print cost every at intervals 10 times or as many iterations if < 10

        if i%math.ceil(num_iters/10) == 0: # math.ceil is used to round up to the nearest integer
            print (f"Iteration {i:4}: Cost {J_history[-1]:0.2e}",#Print the last cost
                   f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}",#Print the gradients
                   f"w:{w: 0.3e}, b:{b: 0.5e}" )#Print the parameters

    return w, b, J_history, p_history # return w and b and J history for graphing