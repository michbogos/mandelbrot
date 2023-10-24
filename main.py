import numpy as np
import matplotlib.pyplot as plt

N=10000
x = np.linspace(-2.7,2.7,N)

img = x[:,None] + complex('j') * x

xy = img

for _ in range(10):
    img = img*img+xy

img = np.where(abs(img)<2, 255, 0)

plt.imshow(img)
plt.show()