from oslo import *
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science"])

# for system_size in [4, 8, 16, 32, 64, 128, 256]:
#     model = OsloModel(32)

model = OsloModel(256)
result = []

for t in range(3000):
    model.run()
    result.append(sum(model.heights))

plt.plot(np.arange(3000), result)
plt.show()

