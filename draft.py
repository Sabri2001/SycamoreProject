import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()

ims = []
for _ in range(10):
    im1, = plt.plot([random.randrange(10), random.randrange(10)], [random.randrange(10), random.randrange(10)])
    im2, = plt.plot([random.randrange(10), random.randrange(10)], [random.randrange(10), random.randrange(10)])
    ims.append([im1, im2])
ani = animation.ArtistAnimation(fig, ims)

plt.show()
