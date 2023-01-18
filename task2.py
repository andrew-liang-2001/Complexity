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
for system_size in [4, 8, 16, 32, 64, 128, 256, 512]:
    model = OsloModel(system_size)
    result = []
    for t in tqdm(range(250000)):
        model.run()
        result.append(model.heights[0])
    plt.plot(np.arange(250000), result, label=f"L={system_size}")

plt.legend()
plt.xlabel("Time t")
plt.ylabel("Height of the pile h(t; L)")
plt.show()

#%% task 2b for L=16
"""This shows a distribution the numerically estimated cross-over times for L=16 only using histograms."""
plt.figure()

model2b = OsloModel(16)
result2ba = []

for _ in tqdm(range(1000)):  # 1000 runs
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
plt.figure()

result2b = []
system_size2b = [4, 8, 16, 32, 64, 128, 256]

for sys_size in tqdm(system_size2b):
    model = OsloModel(sys_size)
    individual_time = []
    for _ in range(10):  # 10 runs to compute the average
        prev_total_height = 0
        while sum(model.heights) == prev_total_height:
            model.run()
            total_height = sum(model.heights)
            prev_total_height += 1
        individual_time.append(model.time)
    result2b.append(np.mean(individual_time))

plt.plot(system_size2b, result2b, "o")
plt.xlabel("System size L")
plt.ylabel("Numerically estimated crossover time")
plt.show()

#%% task 2d
plt.figure()
M = 10  # M is the iterations to average over

for system_size in [4, 8, 16, 32]:
    average = []
    for _ in range(M):
        model = OsloModel(system_size)
        internal_result = []
        for t in tqdm(range(80000)):
            model.run()
            internal_result.append(model.heights[0])
        average.append(internal_result)

    plt.plot(np.arange(80000), result, label=f"L={system_size}")

plt.legend()
plt.xlabel("Time t")
plt.ylabel("Height of pile h(t; L)")
plt.show()

# %%
"""To estimate t_c for each L. This is taken from the script for 2a"""

plt.figure()
for system_size in [4, 8, 16, 32, 64, 128]:
    model = OsloModel(system_size)
    result = []
    for t in tqdm(range(25000)):
        model.run()
        result.append(model.heights[0])
    plt.plot(np.arange(25000), result, label=f"L={system_size}")

plt.legend()
plt.xlabel("Time t")
plt.ylabel("Height of the pile h(t; L)")
plt.show()


#%% task 2e, 2f
plt.figure()

T = 200000

# estimates of t_c for each system size. Needs len(t_c) == len(system_size2e)
t_c = [100, 200, 400, 1500, 5000, 18000, 70000]
system_size2e = [4, 8, 16, 32, 64, 128, 256]
average_heights = []
average_square_heights = []

# the 1 and 2 suffixes are for average_heights and average_square_heights respectively
for sys_size, crit_time in tqdm(zip(system_size2e, t_c), total=len(system_size2e)):
    model = OsloModel(sys_size)
    sum1 = 0
    sum2 = 0
    for _ in range(crit_time):
        model.run()
    for t in range(T):
        model.run()
        sum1 += model.heights[0]
        sum2 += model.heights[0] ** 2
    result1 = sum1/T
    result2 = sum2/T
    average_heights.append(result1)
    average_square_heights.append(result2)

average_heights = np.array(average_heights)
average_square_heights = np.array(average_square_heights)

std_dev = np.sqrt(average_square_heights - average_heights ** 2)

print(f"The system sizes are {system_size2e}")
print(f"The average heights are {average_heights}")
print(f"The standard deviations are {std_dev}")

plt.plot(system_size2e, std_dev, ".")
plt.xlabel("Time t")
plt.ylabel("$\sigma_{h}(L)$")
plt.show()

#%% task 2e continued.
plt.figure()

plt.plot(system_size2e, average_heights)
plt.show()
