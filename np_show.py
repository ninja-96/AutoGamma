import numpy as np

rec = np.load('test.npy')
print(rec)

y = range(np.max(rec[:, 0].astype(np.int)))
import matplotlib.pyplot as plt

x = rec[:, 1]
print(x)
plt.hist(x, y)
plt.show()
