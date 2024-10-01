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
    for i in range(max_iter):
        x_new = x_val[i] - f(x_val[i])/df(x_val[i])
        x_val.append(x_new)
        if np.abs(f(x_new)) < tol:
            break
    return x_val[-1], len(x_val), x_val

# %%
# Bisection method
def bisection(left, right, tol=1e-6):
    mid = []
    max_iter = 1000
    for i in range(max_iter):
        mid.append((left + right)/2)
        if np.abs(f(mid[i]) > 0):
            left = mid[i]
        else:
            right = mid[i]
        if (np.abs(f(mid[i])) < tol):
            break
    return mid[-1], len(mid), mid

# %%
newton_x_final_val, newton_iter_num, newton_x_val = newton_raphson(-1.6)
A1 = newton_x_val
np.save('/Users/chris/Library/CloudStorage/OneDrive-UW/4. Seinor/AMATH 481/HW/HW 1/A1.npy', newton_x_val)
bisec_x_final_val, bisec_iter_num, bisec_x_val = bisection(-0.7, -0.4)
A2 = bisec_x_val
np.save('/Users/chris/Library/CloudStorage/OneDrive-UW/4. Seinor/AMATH 481/HW/HW 1/A2.npy', bisec_x_val)
A3 = [newton_iter_num, bisec_iter_num]
np.save('/Users/chris/Library/CloudStorage/OneDrive-UW/4. Seinor/AMATH 481/HW/HW 1/A3.npy', [newton_iter_num, bisec_iter_num])

# %%
# Question 2
A = np.array([[1, 2], [-1, 1]])
B = np.array([[2, 0], [0, 2]])
C = np.array([[2, 0, -3], [0, 0, -1]])
D = np.array([[1, 2], [2, 3], [-1, 0]])
x = np.array([[1], [0]])
y = np.array([[0], [1]])
z = np.array([[1], [2], [-1]])

# a)
A4 = A + B
np.save('/Users/chris/Library/CloudStorage/OneDrive-UW/4. Seinor/AMATH 481/HW/HW 1/A4.npy', A4)

# b)
A5 = 3*x - 4*y
np.save('/Users/chris/Library/CloudStorage/OneDrive-UW/4. Seinor/AMATH 481/HW/HW 1/A5.npy', A5)

# c)
A6 = np.dot(A, x)
np.save('/Users/chris/Library/CloudStorage/OneDrive-UW/4. Seinor/AMATH 481/HW/HW 1/A6.npy', A6)

# d)
A7 = np.dot(B, (x-y))
np.save('/Users/chris/Library/CloudStorage/OneDrive-UW/4. Seinor/AMATH 481/HW/HW 1/A7.npy', A7)

# e)
A8 = np.dot(D, x)
np.save('/Users/chris/Library/CloudStorage/OneDrive-UW/4. Seinor/AMATH 481/HW/HW 1/A8.npy', A8)

# f)
A9 = np.dot(D, y) + z
np.save('/Users/chris/Library/CloudStorage/OneDrive-UW/4. Seinor/AMATH 481/HW/HW 1/A9.npy', A9)

# g)
A10 = np.dot(A, B)
np.save('/Users/chris/Library/CloudStorage/OneDrive-UW/4. Seinor/AMATH 481/HW/HW 1/A10.npy', A10)

# h)
A11 = np.dot(B, C)
np.save('/Users/chris/Library/CloudStorage/OneDrive-UW/4. Seinor/AMATH 481/HW/HW 1/A11.npy', A11)

# i)
A12 = np.dot(C, D)
np.save('/Users/chris/Library/CloudStorage/OneDrive-UW/4. Seinor/AMATH 481/HW/HW 1/A12.npy', A12)


