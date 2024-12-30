import matplotlib.pyplot as plt
import numpy as np

losses = np.load('./models/loss/1735540711.188275.npy')
plt.plot(losses)
plt.show()
