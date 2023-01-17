from oslo import *
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science"])

#%% task 2a
"""In this Oslo model, heights for all sites are stored already, and hard-coded to be updated in the run method.
Hence, the optimisation that is recording changes in height is not required, because looking up model.heights[0] is O(1)
anyway. I am not using eq (3) to calculate the height at the leftmost site.
"""

plt.figure()
for system_size in [4, 8, 16, 32, 64, 128, 256]:
    model = OsloModel(system_size)
    result = []
    for t in tqdm(range(80000)):
        model.run()
        result.append(model.heights[0])
    plt.plot(np.arange(80000), result, label=f"L={system_size}")

plt.legend()
plt.xlabel("Time t")
plt.ylabel("Height of the pile h(t; L)")
plt.show()

#%% task 2b for L=16
"""This shows a distribution the numerically estimated cross-over times for L=16 only using histograms."""
plt.figure()

model2b = OsloModel(16)
result2ba = []

for _ in tqdm(range(200)):  # 100 runs
    model2b.reset()
    prev_total_height = 0
    while sum(model2b.heights) == prev_total_height:
        model2b.run()
        total_height = sum(model2b.heights)
        prev_total_height += 1
    result2ba.append(model2b.time)

plt.hist(result2ba, bins=20)
plt.xlabel("Cross-over time for L=16")
plt.ylabel("Count")
plt.show()

#%% task2b for varying L
"""This takes a while to run"""
plt.figure()

result2b = []

for system_size in [4, 8, 16, 32, 64, 128, 256]:
    model = OsloModel(system_size)
    individual_time = []
    for _ in tqdm(range(10)):  # 10 runs to compute the average
        prev_total_height = 0
        while sum(model.heights) == prev_total_height:
            model.run()
            total_height = sum(model.heights)
            prev_total_height += 1
        individual_time.append(model.time)
    result2b.append(np.mean(individual_time))

plt.plot([4, 8, 16, 32, 64, 128, 256], result2b, "o")
plt.xlabel("System size L")
plt.ylabel("Numerically estimated crossover time")
plt.show()

#%% task 2d
plt.figure()
M = 10  # M is the iterations to average over

for system_size in [4, 8, 16, 32]:
    for _ in range(M):
        model = OsloModel(system_size)
        result = []
        for t in tqdm(range(80000)):
            model.run()
            result.append(model.heights[0])
        plt.plot(np.arange(80000), result, label=f"L={system_size}")

plt.legend()
plt.xlabel("Time t")
plt.ylabel("Height at leftmost site h(t; L)")
plt.show()