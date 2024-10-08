# %%
import numpy as np

# %%
def f(x):
    return x * np.sin(3*x) - np.exp(x)

def df(x):
    return np.sin(3*x) + 3*x*np.cos(3*x) - np.exp(x)

# %%
# Newton-Raphson method
def newton_raphson(x0, tol=1e-6):
    x_val = [x0]
    max_iter = 1000
    iter_num = 0
    for i in range(max_iter):
        iter_num += 1
        x_new = x_val[i] - f(x_val[i])/df(x_val[i])
        x_val.append(x_new)
        if (np.abs(f(x_val[i])) < tol):
            break
    return x_val[-1], iter_num, x_val

# %%
# Bisection method
def bisection(left, right, tol=1e-6):
    mid = []
    max_iter = 1000
    iter_num = 0
    for i in range(max_iter):
        iter_num += 1
        mid.append((left + right)/2)
        if (f(mid[i]) > 0):
            left = mid[i]
        else:
            right = mid[i]
        if (np.abs(f(mid[i])) < tol):
            break
    return mid[-1], iter_num, mid

# %%
newton_x_final_val, newton_iter_num, newton_x_val = newton_raphson(-1.6)
A1 = newton_x_val
bisec_x_final_val, bisec_iter_num, bisec_x_val = bisection(-0.7, -0.4)
A2 = bisec_x_val
A3 = [newton_iter_num, bisec_iter_num]
print(A1, A2, A3)

# %%
# Question 2
A = np.array([[1, 2], [-1, 1]])
B = np.array([[2, 0], [0, 2]])
C = np.array([[2, 0, -3], [0, 0, -1]])
D = np.array([[1, 2], [2, 3], [-1, 0]])
x = np.array([1, 0])
y = np.array([0, 1])
z = np.array([1, 2, -1])

# a)
A4 = A + B

# b)
A5 = (3*x - 4*y)
print(A5)

# c)
A6 = (np.dot(A, x))

# d)
A7 = (np.dot(B, (x-y)))

# e)
A8 = (np.dot(D, x))

# f)
A9 = (np.dot(D, y) + z)

# g)
A10 = np.dot(A, B)

# h)
A11 = np.dot(B, C)

# i)
A12 = np.dot(C, D)


