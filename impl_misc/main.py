import numpy as np
import matplotlib.pyplot as plt
import tqdm

scale = 0.00000000000000002

for frame in range(120):
    N=1024
    x = np.linspace(-1,1,N, dtype=np.complex256)

    img = x[:,None] + complex('j') * x
    img *= scale
    img += complex(-0.0700432019411218 - 0.8224676332988761j)
    xy = img
    for _ in tqdm.trange(1000):
        img = img*img+xy

    img = np.where(abs(img)<2, 255, 0)

    scale *= 2

    plt.imshow(img)
    plt.savefig(f"{frame}.png", bbox_inches='tight')
    print(f"Saved frame: {frame} @ scale: {scale}")