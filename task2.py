from oslo import *
from logbin import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import scienceplots
from collections import Counter

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

#%% task 2a - up to L=256 only
plt.figure()

for system_size in L:
    model = OsloModel(system_size)
    result = []
    for t in tqdm(range(80000)):
        model.run()
        result.append(model.heights[0])
    plt.plot(np.arange(80000), result, label=f"L={system_size}")

plt.legend(title="System size L")
plt.xlabel("Time t")
plt.ylabel("Height of the pile $h(t; L)$")
plt.show()

#%% task 2b for L=16 - NOT ESSENTIAL
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

#%% task 2b for varying L
plt.figure()

result2b = []

for sys_size in tqdm(L):
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

"""Use polyfit on log-log plot to calculate exponent. Note that error propagation still needs to be done.
i.e. I need to know the error on each data point to begin this"""
# noinspection PyTupleAssignmentBalance
p, pcov = np.polyfit(np.log2(L), np.log2(result2b), deg=1, cov=True)

plt.plot(L, result2b, "o")
high_density = np.arange(max(L))
plt.plot(high_density, pow(high_density, p[0]), label=f"exponent {p[0]}")
plt.legend()
plt.xlabel("System size L")
plt.ylabel("Numerically estimated crossover time")
plt.show()

#%% task 2d
plt.figure()
M = 20  # M is the iterations to average over
result2d = []
for system_size in tqdm(L):
    average = []
    for _ in range(M):
        model = OsloModel(system_size)
        internal_timings = []
        for t in range(80000):
            model.run()
            internal_timings.append(model.heights[0])
        average.append(internal_timings)
    result2d.append(np.mean(average, axis=0))
    plt.plot(np.arange(80000), np.mean(average, axis=0), label=f"L={system_size}")

plt.legend(title="Iterations M=20")
plt.xlabel("Time t")
plt.ylabel("Smoothed height of pile $\\tilde{h}(t; L)$")
plt.show()

#%% To estimate t_c for each L. This is taken from the script for 2a
plt.figure()
for system_size in L:
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

#%% task 2d log-log plot collapsed
plt.figure()
result2d = np.array(result2d)
L = np.array(L)

for system_size, time_series in zip(L, result2d):
    plt.plot(np.arange(80000)/system_size ** 2, time_series/system_size, label=f"L={system_size}")

plt.legend()
plt.xlabel("Time $t/L^2$")
plt.ylabel("Height of the pile $h(t; L)/L$")
plt.show()

#%% task 2e, 2f, 3a, 3b - This cell block deals with results after t > t_c
plt.figure()

T = 200000  # time steps after t_c

# estimates of t_c for each system size. Needs len(t_c) == len(system_size2e)
t_c = [100, 200, 400, 1500, 5000, 18000, 70000]

average_heights = []
average_square_heights = []
S_1, S_2, S_3, S_4 = [], [], [], []

# the 1 and 2 suffixes are for average_heights and average_square_heights respectively
for sys_size, crit_time in tqdm(zip(L, t_c), total=len(L)):
    model = OsloModel(sys_size)
    N = 0
    sum1 = 0
    sum2 = 0
    avalanche_sizes = []

    s_1, s_2, s_3, s_4 = 0, 0, 0, 0

    for _ in range(crit_time):  # run until t > t_c
        model.run()
    for t in range(T):
        # This is for avalanches, task 3a
        avalanches = model.run()
        N += avalanches
        avalanche_sizes.append(avalanches)
        s_2 += avalanches ** 2
        s_3 += avalanches ** 3
        s_4 += avalanches ** 4

        # This is for calculating means and std, tasks 2e and 2f
        sum1 += model.heights[0]
        sum2 += model.heights[0] ** 2

    # print(Counter(avalanche_sizes))
    x, y = logbin(avalanche_sizes, scale=1.2, zeros=False)
    # print(debug)
    plt.loglog(x, y, "o--", ms=3, label=f"L={sys_size}")
    # plt.loglog(x/sys_size ** 2.25, x ** 1.556 * y, "o--", ms=3, label=f"L={sys_size}")
    S_1.append(N/T)
    S_2.append(s_2/T)
    S_3.append(s_3/T)
    S_4.append(s_4/T)
    average_heights.append(sum1/T)
    average_square_heights.append(sum2/T)

average_heights = np.array(average_heights)
average_square_heights = np.array(average_square_heights)

std_dev = np.sqrt(average_square_heights - average_heights ** 2)

plt.xlabel("Avalanche size s")
plt.ylabel("Normalised frequency $\\tilde P_{N}$(s; L)")
plt.legend(title="System size L")
plt.grid()
plt.show()

print(f"The system sizes are {L}")
print(f"The average heights are {average_heights}")
print(f"The standard deviations are {std_dev}")

print(f"S_1 = {S_1}")
print(f"S_2 = {S_2}")
print(f"S_3 = {S_3}")
print(f"S_4 = {S_4}")

# %%
plt.plot(L, std_dev, ".")
plt.xlabel("Time t")
plt.ylabel("$\sigma_{h}(L)$")
plt.show()

#%% task 2e continued.
plt.figure()

plt.plot(L, average_heights, ".")
plt.show()
