import numpy as np

A = np.array([[1, 2],[3, 4]])
print(A)
A_flat = A.flatten('F')
print(type(A_flat))
A2 = np.array([A_flat])
AT = A2.T

print(np.shape(AT))