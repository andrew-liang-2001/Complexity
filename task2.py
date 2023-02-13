import matplotlib as mpl
import tikzplotlib

mpl.use("pgf")
# mpl.use("MacOSX")

from oslo import *
from logbin import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import scienceplots
from collections import Counter
import scipy as sp
from statsmodels.graphics.tsaplots import plot_acf
import pandas as pd

plt.style.use(["science", "grid"])
plt.rcParams.update({"font.size": 16,
                     "figure.figsize": (7.5, 4.5),
                     "axes.labelsize": 15,
                     "legend.fontsize": 12,
                     "xtick.labelsize": 13,
                     "ytick.labelsize": 13,
                     'axes.prop_cycle': plt.cycler(color=plt.cm.tab10.colors)})

# %% task 2a - log scaling for large L
"""In this Oslo model, heights for all sites are stored already, and hard-coded to be updated in the run method.
Hence, the optimisation that is recording changes in height is not required, because looking up model.heights[0] is O(1)
anyway. I am not using eq (3) to calculate the height at the leftmost site.
"""

plt.figure()
for system_size in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
    model = OsloModel(system_size)
    result = []
    for t in tqdm(range(1_200_000)):
        model.run()
        result.append(model.heights[0])
    plt.loglog(np.arange(1200000), result, label=f"L={system_size}")

plt.legend()
plt.xlabel("Time t")
plt.ylabel("Height of the pile $h(t; L)$")
# plt.show()
plt.savefig('Task2a1.pgf', format='pgf')
# tikzplotlib.clean_figure()
# tikzplotlib.save("Task2a.tex")

# %% task 2a - up to L=256 only
plt.figure()

for system_size in L:
    model = OsloModel(system_size)
    result = []
    for t in tqdm(range(80000)):
        model.run()
        result.append(model.heights[0])
    plt.plot(np.arange(80000), result, label=f"L={system_size}")

plt.legend()
plt.xlabel("Time t")
plt.ylabel("Height of the pile $h(t; L)$")

plt.savefig('Task2a.pgf', format='pgf')
# plt.show()

# %% task 2b for L=16 - NOT ESSENTIAL
"""This shows a distribution the numerically estimated cross-over times for L=16 only using histograms."""
model2b = OsloModel(16)
result2ba = []

for _ in tqdm(range(2000)):
    model2b.reset()
    prev_total_height = 0
    while sum(model2b.heights) == prev_total_height:
        model2b.run()
        total_height = sum(model2b.heights)
        prev_total_height += 1
    result2ba.append(model2b.time)

_, bins, _ = plt.hist(result2ba, bins=20, density=True, histtype='step')
x = np.linspace(min(result2ba), max(result2ba), 1000)
mu, sigma = sp.stats.norm.fit(result2ba)
best_fit_line = sp.stats.norm.pdf(x, mu, sigma)
plt.plot(x, best_fit_line, "r-", label="$\mu$ = %.1f, $\sigma$ = %.1f" % (mu, sigma))
plt.xlabel("Cross-over time $\\langle t_c(L) \\rangle$ for L=16")
plt.legend()
plt.ylabel("Density")
plt.show()

# %% task 2b for varying L
plt.figure()
result2b = []
std2b = []

for sys_size in tqdm(L_big):
    model = OsloModel(sys_size)
    individual_time = []
    for _ in range(10):  # 10 runs to compute the average
        model.reset()
        prev_total_height = 0
        while sum(model.heights) == prev_total_height:
            model.run()
            total_height = sum(model.heights)
            prev_total_height += 1
        individual_time.append(model.time)
    # print(individual_time)
    result2b.append(np.mean(individual_time))
    std2b.append(np.std(individual_time))

# noinspection PyTupleAssignmentBalance
p2b, pcov2b = np.polyfit(np.log(L), np.log(result2b), deg=1, cov=True)

# p2b, pcov2b = np.polyfit(np.log2(L), np.log2(result2b), deg=1, cov="unscaled", w=1/np.array(std2b))

plt.errorbar(L, result2b, yerr=std2b, fmt=".")
high_density = np.arange(max(L))

"We deliberately don't give error for the exponent because the fit applied is not weighted!"
plt.plot(high_density, np.exp(p2b[1])*pow(high_density, p2b[0]), "--", label=f"exponent {p2b[0]}")

plt.legend()
plt.xlabel("System size L")
plt.ylabel("Mean crossover time $\\langle t_c(L) \\rangle$")
plt.show()

# %% task 2d - this takes forever!
# TODO: Consider is it better to increase average number or time
plt.figure()
# M = [200_000, 100_000, 50_000, 20_000, 5000, 1000, 200, 100, 10]  # M is the iterations to average over
M = [10_000_000, 2_500_000, 750_000, 200_000, 40_000, 8000, 1000, 400, 20]  # run this overnight 8h+
# M = np.ones(len(L_big), dtype=int)

time2d = t_c * 2
result2d = []

for system_size, i, time_to_plot_until in zip(L_big, M, time2d):
    average = []
    for _ in tqdm(range(i)):
        model = OsloModel(system_size)
        internal_timings = [0]  # height is 0 at t=0 always
        for t in range(time_to_plot_until):
            model.run()
            internal_timings.append(model.heights[0])
        average.append(internal_timings)
    result2d.append(np.mean(average, axis=0))
    plt.plot(np.arange(time_to_plot_until+1), np.mean(average, axis=0), label=f"L={system_size}")

plt.legend()
plt.xlabel("Time t")
plt.ylabel("Smoothed height of pile $\\tilde{h}(t; L)$")
plt.show()

# %% task 2d plot collapsed
plt.figure()
result2d = np.array(result2d)

for system_size, time_series in zip(L_big, result2d):
    plt.plot(np.arange(2_500_000) / system_size ** 2, np.pad(time_series, (0, 2_500_000 - len(time_series)), 'constant') / system_size, label=f"L={system_size}")

# noinspection PyTupleAssignmentBalance
# p2d, pcov2d = np.polyfit(np.log(), np.log(result2d[:][-1].flatten()), deg=1, cov=True)

plt.plot(np.linspace(0, 1.5, 10_000), np.sqrt(np.linspace(0, 1.5, 10_000)), "--", label="$\\sqrt{t}$")
plt.legend()
plt.xlim(-0.05, 1.5)
plt.xlabel("$t/L^2$")
plt.ylabel("Smoothed height of the pile $\\tilde{h}(t; L)/L$")
# plt.savefig('Task2d_collapsed.pgf', format='pgf')
plt.show()

# %% task 2d log-log plot collapsed
plt.figure()
result2d = np.array(result2d)

for system_size, time_series in zip(L_big, result2d):
    plt.loglog(np.arange(2_500_000) / system_size ** 2, np.pad(time_series, (0, 2_500_000 - len(time_series)), 'constant') / system_size, label=f"L={system_size}")

# noinspection PyTupleAssignmentBalance
# p2d, pcov2d = np.polyfit(np.log(), np.log(result2d[:][-1].flatten()), deg=1, cov=True)

plt.loglog(np.linspace(0, 1.5, 10_000), np.sqrt(np.linspace(0, 1.5, 10_000)), "--", label="$\\sqrt{t}$")
plt.legend()
plt.xlabel("Time $t/L^2$")
plt.ylabel("Smoothed height of the pile $\\tilde{h}(t; L)/L$")
# plt.savefig('Task2d_collapsed.pgf', format='pgf')
plt.show()

# %% task 2e, 2f, 3a, 3b - This cell block deals with results after t > t_c
# T = 200_000  # time steps after t_c
#
# # estimates of t_c for each system size. Needs len(t_c) == len(system_size2e)
# t_c = [100, 200, 400, 1500, 5000, 18000, 70000, 220_000, 1_000_000]
#
# average_heights = []
# average_square_heights = []
#
# # the 1 and 2 suffixes are for average_heights and average_square_heights respectively
# for sys_size, crit_time in tqdm(zip(L_big, t_c), total=len(L_big)):
#     model = OsloModel(sys_size)
#     sum1 = 0
#     sum2 = 0
#
#     for _ in range(crit_time):  # run until t > t_c
#         model.run()
#     for t in range(T):
#
#         # This is for calculating means and std, tasks 2e and 2f
#         sum1 += model.heights[0]
#         sum2 += model.heights[0] ** 2
#         # temp_np_std.append(model.heights[0])
#
#     average_heights.append(sum1 / T)
#     average_square_heights.append(sum2 / T)
#
# average_heights = np.array(average_heights)
# average_square_heights = np.array(average_square_heights)
#
# std_dev = np.sqrt(average_square_heights - average_heights ** 2)
#
# print(f"The average heights are {average_heights}")
# print(f"The standard deviations are {std_dev}")


# %% Task 2e & 3b - Find moments, avg heights and std dev
T = 10_000_000  # time steps after t_c

S_matrix = []
average_heights = []
average_square_heights = []

# the 1 and 2 suffixes are for average_heights and average_square_heights respectively
for sys_size, crit_time in tqdm(zip(L_big, t_c), total=len(L_big)):
    model = OsloModel(sys_size)
    s_1, s_2, s_3, s_4, s_5, s_6, s_7, s_8, s_9, s_10 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    sum1 = 0
    sum2 = 0
    for _ in range(crit_time):  # run until t > t_c
        model.run()
    for t in range(T):
        # This is for avalanches, task 3a
        avalanches = model.run()

        s_1 += avalanches
        s_2 += avalanches ** 2
        s_3 += avalanches ** 3
        s_4 += avalanches ** 4
        s_5 += avalanches ** 5
        s_6 += avalanches ** 6
        s_7 += avalanches ** 7
        s_8 += avalanches ** 8
        s_9 += avalanches ** 9
        s_10 += avalanches ** 10

        sum1 += model.heights[0]
        sum2 += model.heights[0] ** 2

    S_matrix.append([s_1 / T, s_2 / T, s_3 / T, s_4 / T, s_5 / T, s_6 / T, s_7 / T, s_8 / T, s_9 / T, s_10 / T])

    average_heights.append(sum1 / T)
    average_square_heights.append(sum2 / T)

S_matrix = np.transpose(S_matrix)
average_heights = np.array(average_heights)
average_square_heights = np.array(average_square_heights)

std_dev = np.sqrt(average_square_heights - average_heights ** 2)

print(f"The average heights are {average_heights}")
print(f"The standard deviations are {std_dev}")

# %% task 2e - curve fit method
plt.figure()

# noinspection PyTupleAssignmentBalance
popt2e, pcov2e = sp.optimize.curve_fit(truncated_series, L_big, average_heights, sigma=std_dev/np.sqrt(T),
                                       absolute_sigma=True)

print("a_0 = %.6f ± %.6f" % (popt2e[0], np.sqrt(pcov2e[0, 0])))
print("a_1 = %.4f ± %.4f" % (popt2e[1], np.sqrt(pcov2e[1, 1])))
print("w_1 = %.4f ± %.4f" % (popt2e[2], np.sqrt(pcov2e[2, 2])))

# plt.plot(L_big, average_heights, ".")
plt.errorbar(L_big, average_heights, yerr=std_dev/np.sqrt(T), fmt=".")
plt.plot(L_big, truncated_series(L_big, *popt2e))
plt.xlabel("System size L")
plt.ylabel("Average height $\\langle h \\rangle$")
plt.show()

#%% task 2e - log method
plt.figure()
a_0_guess = 1.7346

plt.loglog(L_big, a_0_guess - average_heights/L_big)
plt.xlabel("System size L")
plt.ylabel(r"$a_0 - \langle h(t ; L)\rangle_t /L$")
plt.show()

# %% task 2f - log-log plot of sigma_h vs L
plt.figure()
plt.loglog(L_big, std_dev, ".")

# noinspection PyTupleAssignmentBalance
p2f, pcov2f = np.polyfit(np.log(L_big), np.log(std_dev), deg=1, cov=True)

high_density = np.arange(min(L_big), max(L_big))
plt.loglog(high_density, np.exp(p2f[1]) * pow(high_density, p2f[0]), label=f"exponent {p2f[0]}")

plt.xlabel("System size L")
plt.ylabel("$\sigma_{h}(L)$")
plt.legend()
plt.show()

# %% task 2f - check slope against L as L -> infinity
plt.figure()

plt.plot(L, average_heights / L, ".", label="Standard deviation from formula")
plt.xlabel("System size L")
plt.ylabel("$\\langle h(t; L) \\rangle_{t} / L$")
plt.show()

# %%
plt.figure()

plt.plot(L, std_dev, ".")
plt.show()

# %% task 2g
plt.figure()

T = 200000  # time steps after t_c

# estimates of t_c for each system size. Needs len(t_c) == len(system_size2e)
t_c = [100, 200, 400, 1500, 5000, 18000, 70000]

for i, (sys_size, crit_time) in tqdm(enumerate(zip(L, t_c)), total=len(L)):
    model = OsloModel(sys_size)
    observed_config = []
    for _ in range(crit_time):  # run until t > t_c
        model.run()
    for t in range(T):
        # This is for avalanches, task 3a
        model.run()
        observed_config.append(model.heights[0])  # make array hashable

    prob = Counter(observed_config)  # count the first index for each L
    prob = {k: v / T for k, v in prob.items()}  # normalise
    # plt.plot(list(prob.keys()), list(prob.values()), ".--", label=f"L={sys_size}")
    # plt.plot((np.fromiter(prob.keys(), dtype=np.double) - average_heights[i]) / std_dev[i],
    #          np.fromiter(prob.values(), dtype=np.double) * std_dev[i], ".", label=f"L={sys_size}")
    plt.hist(prob.keys(), weights=prob.values(), bins=100, label=f"L={sys_size}")

plt.xlabel("$h$")
plt.ylabel("Probability $P(h; L)$")
plt.legend()
plt.show()

# %% task 2g - collapsed
plt.figure()

T = 10_000_000  # time steps after t_c

# estimates of t_c for each system size. Needs len(t_c) == len(system_size2e)
t_c = [100, 200, 400, 1500, 5000, 18000, 70000, 220_000, 1_000_000]

for i, (sys_size, crit_time) in tqdm(enumerate(zip(L_big, t_c)), total=len(L_big)):
    model = OsloModel(sys_size)
    observed_config = []
    for _ in range(crit_time):  # run until t > t_c
        model.run()
    for t in range(T):
        # This is for avalanches, task 3a
        model.run()
        observed_config.append(model.heights[0])

    prob = Counter(observed_config)  # count the first index for each L
    prob = {k: v / T for k, v in prob.items()}  # normalise
    # plt.plot(list(prob.keys()), list(prob.values()), ".--", label=f"L={sys_size}")
    plt.plot((np.fromiter(prob.keys(), dtype=np.double) - average_heights[i]) / std_dev[i],
             np.fromiter(prob.values(), dtype=np.double) * std_dev[i], ".", label=f"L={sys_size}")

plt.xlabel(r"$\left(h-\langle h \rangle\right)/ \sigma_h$")
plt.ylabel(r"Probability $\sigma_h P(h; L)$")
plt.legend()
plt.show()

#%% Task 3b - plot moments
plt.figure()
p = plt.get_cmap("tab10")
result3b = []  # list of $D(1+k-\tau_s)$ for each system size

for i, moment in enumerate(S_matrix):
    plt.loglog(L_big, moment, ".", label=f"k={i + 1}", color=p(i / 10))
    # noinspection PyTupleAssignmentBalance
    p31, pcov3a = np.polyfit(np.log(L_big[-4:]), np.log(moment[-4:]), deg=1, cov=True)  # fit only to last 5 points
    print(p31[0])
    result3b.append(p31[0])
    plt.loglog(L_big, np.exp(p31[1]) * L_big ** p31[0], "--", color=p(i / 10))

plt.xlabel("System size L")
plt.ylabel("Moment $ \langle s^{k} \\rangle $")
plt.legend()
plt.show()

# %% Task 3b - polyfit to find $D(1+k-\tau_s)$
# plt.figure()
#
# p = plt.get_cmap("viridis")
# result3b = []  # list of $D(1+k-\tau_s)$ for each system size
# for j, i in enumerate(S_matrix):
#     plt.loglog(L, i, ".", label=f"S_{j + 1}", color=p(j / 10))
#     # noinspection PyTupleAssignmentBalance
#     p31, pcov3a = np.polyfit(np.log(L[-4:]), np.log(i[-4:]), deg=1, cov=True)  # fit only to last 5 points
#     print(p31[0])
#     result3b.append(p31[0])
#     plt.loglog(L, np.exp(p31[1]) * L ** p31[0], "--", color=p(j / 10))
#
# plt.legend()
# plt.show()

# %% Task 3b - Find both exponents
plt.figure()

k = np.arange(1, 11)
plt.plot(k, result3b, ".")
# noinspection PyTupleAssignmentBalance
p3b1, pcov3b1 = np.polyfit(k, result3b, deg=1, cov=True)
plt.plot(k, k * p3b1[0] + p3b1[1], "--")
print(f"D= {p3b1[0]}")
print(f"\\tau_s = {p3b1[1] + 1}")

plt.xlabel("$k$")
plt.ylabel("$D(1+k-\\tau_s)$")
plt.show()

# %% Task 3a Not collapsed
plt.figure()

T = 1_000_000  # time steps after t_c

t_c = [100, 200, 400, 1500, 5000, 18000, 70000, 220_000, 1_000_000]

for sys_size, crit_time in tqdm(zip(L_big, t_c), total=len(L_big)):
    model = OsloModel(sys_size)
    avalanche_sizes = []

    for _ in range(crit_time):  # run until t > t_c
        model.run()
    for t in range(T):
        # This is for avalanches, task 3a
        avalanches = model.run()
        avalanche_sizes.append(avalanches)

    x, y = logbin(avalanche_sizes, scale=1.3, zeros=False)
    plt.loglog(x, y, "o--", ms=3, label=f"L={sys_size}")

plt.xlabel("Avalanche size s")
plt.ylabel(r"Log-binned probability $\tilde P_{N}(s; L)$")
plt.legend()
plt.savefig('Task3a1.pgf', format='pgf')
# plt.show()

# %%

# %% Task 3a collapsed

plt.figure()

T = 1_000_000  # time steps after t_c

t_c = [100, 200, 400, 1500, 5000, 18000, 70000, 220_000, 1_000_000]

for sys_size, crit_time in tqdm(zip(L_big, t_c), total=len(L_big)):
    model = OsloModel(sys_size)
    avalanche_sizes = []

    for _ in range(crit_time):  # run until t > t_c
        model.run()
    for t in range(T):
        # This is for avalanches, task 3a
        avalanches = model.run()
        avalanche_sizes.append(avalanches)

    x, y = logbin(avalanche_sizes, scale=1.3, zeros=False)
    # TODO: Need to justify scaling exponents below
    plt.loglog(x / sys_size ** 2.25, x ** (14/9) * y, "-", ms=3, label=f"L={sys_size}")

# TODO: Rewrite the axes labels
plt.xlabel("$sL^{-D}$")
plt.ylabel("$s^{\\tau_s} \\tilde P_{N}(s; L)$")
plt.legend()
plt.savefig('Task3a2.pgf', format='pgf')
# plt.show()
