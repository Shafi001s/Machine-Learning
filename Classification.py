import numpy as np
import matplotlib.pyplot as plt
import copy, math




x_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])

w_tmp = np.array([2,3])
b_tmp = -1

#Input array
input_array = np.array([1,2,3])
exp_array = np.exp(input_array) # Exponential of each element

print("Input to exp: ", input_array)
print("Output of exp: ", exp_array)


#input is a single number
input_val = 1
exp_val = np.exp(input_val)

print("Input to exp: ", input_val)
print("Output of exp: ", exp_val)

##SIGMOID Function
def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z (ndarray): a scalr, numpy array of any size.

    Returns:
     g (ndarray): sigmoid (z), with the same shape as z

    """

    g = 1/ (1 + np.exp(-z))

    return g

## Graph for the data plot
def plot_data(x, y, ax=None):
    if ax is None:
        ax = plt.gca()#Get current axis    
    pos = y.flatten() == 1#Positive examples
    neg = y.flatten() == 0# Negative examples
    ax.scatter(x[neg, 0], x[neg, 1], marker='o', s=80, label="y=0", facecolors='none', edgecolors='blue') #positive examples
    ax.scatter(x[pos, 0], x[pos, 1], marker='x', s=80, label="y=1", c='red') #negative examples
    ax.legend()


##Compute cost/loss 
def compute_cost_logistic(X, y, w, b):
    """
        Computes cost

    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      cost (scalar): cost

    """
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i],w) + b
        f_wb_i = sigmoid(z_i)
        cost += -y[i]*np.log(f_wb_i) - (i - y[i])*np.log(1-f_wb_i)
    cost = cost/m
    return cost
    

print(compute_cost_logistic(x_train, y_train, w_tmp, b_tmp))


##Compute Gradient of logistic function
def compute_gradient_logistic(X, y, w, b):
    """
    Computes the gradient for logistic regression 
 
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
    Returns
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar)      : The gradient of the cost w.r.t. the parameter b. 

    """
    m,n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i],w) + b)
        err_i = f_wb_i - y[i]

        for j in range(n):
            dj_dw[j] = dj_dw[j] = err_i * X[i,j]
        dj_db = dj_db + err_i
    dj_db = dj_db/m
    dj_dw = dj_dw/m

    return dj_dw, dj_db

dj_dw_tmp, dj_db_tmp = compute_gradient_logistic(x_train, y_train, w_tmp, b_tmp)
print(f"dj_db: {dj_db_tmp}" )
print(f"dj_dw: {dj_dw_tmp.tolist()}" ) #tolist for listing the numbers

def gradient_descent(X, y, w_in, b_in, alpha, num_iters): 
    """
    Performs batch gradient descent
    
    Args:
      X (ndarray (m,n)   : Data, m examples with n features
      y (ndarray (m,))   : target values
      w_in (ndarray (n,)): Initial values of model parameters  
      b_in (scalar)      : Initial values of model parameter
      alpha (float)      : Learning rate
      num_iters (scalar) : number of iterations to run gradient descent
      
    Returns:
      w (ndarray (n,))   : Updated values of parameters
      b (scalar)         : Updated value of parameter 
    """
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):
        # Calculate the gradient and update the parameters
        dj_dw, dj_db = compute_gradient_logistic(X, y, w, b)   

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               
      
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( compute_cost_logistic(X, y, w, b) )

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")
        
    return w, b, J_history         #return final w,b and J history for graphing
alph = 0.1
iters = 10000

w_out, b_out, _ = gradient_descent(x_train, y_train, w_tmp, b_tmp, alph, iters) 
print(f"\nupdated parameters: w:{w_out}, b:{b_out}")




"""
#Generate an array of evenly spaced value from -10 to 10
z_tmp = np.arange(-10,11)

#Use the sigmoid function to compute the sigmoid of each value
y = sigmoid(z_tmp)

np.set_printoptions(precision=3) #Set precision to 3 decimal places
print("Input (z), Output(sigmoid(z))")
print(np.c_[z_tmp, y]) #Print z_tmp and y side by side



#Plotting the data points
fig,ax = plt.subplots(1,1,figsize=(4,4))
plot_data(x_train , y_train , ax)

ax.axis([0, 4, 0, 3.5])
ax.set_ylabel('$x_1$')
ax.set_xlabel('$x_0$')
plt.show()





#Choose values between 0 and 6
x0 = np.arange(0,6)

x1 = 3 - x0 # since z = 0 = -3 + x1 + x2
x1_other = 4 - x0 # just for example, not real outcome

fig,ax = plt.subplots(1, 1, figsize=(4,4))
# Plot the decision boundary
ax.plot(x0,x1, c="blue", label="$b$=-3")
ax.plot(x0,x1_other, c="magenta", label="$b$=-4")
ax.axis([0, 4, 0, 4])

# Plot the original data
plot_data(x_train,y_train,ax)
ax.axis([0, 4, 0, 4])
ax.set_ylabel('$x_1$', fontsize=12)
ax.set_xlabel('$x_0$', fontsize=12)
plt.legend(loc="upper right")
plt.title("Decision Boundary")
plt.show()

#Plot z vs sigmoid(z)

fig,ax = plt.subplots(1,1, figsize = (5,3))
ax.plot(z_tmp, y, c ='b')

ax.set_title("Sigmoid Function")
ax.set_ylabel('Sigmoid(z)')
ax.set_xlabel('z')

ax.axvline(0, color ='k', ls='--', lw=0.5) # Add vertical line at x=0 # ls is line style, lw is line width
ax.axhline(0.5, color ='k', ls='--', lw=0.5) # Add horizontal line at y=0.5
plt.show()



fig,ax = plt.subplots(1,1, figsize = (5,4))

#Plot the decision boundary

ax.plot(x0, x1, c ="b")
ax.axis([0, 4, 0, 3.5])

#Fill the region below the line
ax.fill_between(x0,x1, alpha = 0.2)

# Plot the original data
plot_data(x_train,y_train,ax)
ax.set_ylabel(r'$x_1$')
ax.set_xlabel(r'$x_0$')
plt.show()

"""

    
