import matplotlib as mpl

mpl.use("MacOSX")

from oslo import *
from logbin import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import scienceplots
from collections import Counter
import scipy as sp
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm
import pandas as pd
import pickle
import uncertainties as unc
from uncertainties import umath

plt.style.use(["science"])
plt.rcParams.update({"font.size": 16,
                     "figure.figsize": (6.6942, 4.016538),
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
result2a = []

for system_size in L_big:
    model = OsloModel(system_size)
    result = []
    for t in tqdm(range(1_200_000)):
        model.run()
        result.append(model.heights[0])
    result2a.append(result)
    plt.loglog(np.arange(1200000), result, label=f"L={system_size}")

pd.to_pickle(result2a, "task2a.pkl")
plt.legend()
plt.xlabel("t")
plt.ylabel("$h(t; L)$")
plt.savefig("plots/Task2a1.pgf', format='pgf")

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
plt.xlabel("t")
plt.ylabel("$h(t; L)$")
plt.savefig("plots/Task2a.pgf', format='pgf")
# plt.show()

# %% task 2b for L=16 - NOT ESSENTIAL
"""This shows a distribution the numerically estimated cross-over times for L=16 only using histograms."""
model2b = OsloModel(16)
result2ba = []
fig, ax = plt.subplots(1, 2)

for _ in tqdm(range(100)):
    model2b.reset()
    prev_total_height = 0
    while sum(model2b.heights) == prev_total_height:
        model2b.run()
        total_height = sum(model2b.heights)
        prev_total_height += 1
    result2ba.append(model2b.time)

_, bins, _ = ax[0].hist(result2ba, density=True, histtype='step')
x = np.linspace(min(result2ba), max(result2ba), 1000)
mu, sigma = sp.stats.norm.fit(result2ba)
best_fit_line = sp.stats.norm.pdf(x, mu, sigma)
ax[0].plot(x, best_fit_line, "r-")
ax[0].set_xlabel("Cross-over time $\\langle t_c(L) \\rangle$ for L=16")
ax[0].legend()
ax[0].set_ylabel("Density")
plt.show()

# %% task 2b for varying L - this takes forever! Pickle this
result2b = []
std2b = []
M2b = 50

for sys_size in L_big:
    model = OsloModel(sys_size)
    individual_time = []
    for _ in tqdm(range(M2b)):  # 10 runs to compute the average
        model.reset()
        prev_total_height = 0
        while sum(model.heights) == prev_total_height:
            model.run()
            total_height = sum(model.heights)
            prev_total_height += 1
        individual_time.append(model.time)
    result2b.append(np.mean(individual_time))
    std2b.append(np.std(individual_time))

pd.to_pickle(result2b, "data/task2b.pkl")
pd.to_pickle(std2b, "data/task2b_std.pkl")

# %%
M2b = 50
result2b = pd.read_pickle("data/task2b.pkl")
std2b = pd.read_pickle("data/task2b_std.pkl")

# %% task 2b - plot
fig, ax = plt.subplots(1, 2, figsize=(6.6942, 4.016538))

# noinspection PyTupleAssignmentBalance
p2b, pcov2b = np.polyfit(np.log(L_big)[-4:], np.log(result2b)[-4:], deg=1, cov=True, w=(L_big/std2b)[-4:])
ax[0].loglog(L_big, result2b, ".")
high_density = np.arange(max(L_big))

exponent2b = unc.ufloat(p2b[0], np.sqrt(pcov2b[0, 0]))
log_b = unc.ufloat(p2b[1], np.sqrt(pcov2b[1, 1]))
b = umath.exp(log_b)
ax[0].plot(high_density, np.exp(p2b[1])*pow(high_density, p2b[0]), "--", label=r"$\beta$ = %.3f $\pm$ %.3f" % (exponent2b.n, exponent2b.s))

ax[0].legend(fontsize=12)
ax[0].set_xlabel("L")
ax[0].set_ylabel(r"$\langle t_c(L) \rangle$")
ax[1].plot(L_big, result2b / L_big ** 2, ".--")
ax[1].set_xlabel("L")
ax[1].set_ylabel(r"$\langle t_c(L) \rangle/L^{2}$")
fig.tight_layout()
plt.savefig("plots/Task2b.pgf", format="pgf")
plt.show()

# %%
fig = plt.figure()
# noinspection PyTupleAssignmentBalance
plt.errorbar(L_big, result2b / L_big ** 2, fmt=".--")

plt.xlabel("L")
plt.ylabel(r"$\langle t_c(L) \rangle/L^{2}$")
fig.tight_layout()
plt.show()

# %% task 2d - this takes forever!
# plt.figure()
# M = [2_000_000, 500_000, 100_000, 20_000, 5000, 1000, 200, 50, 10]  # M is the iterations to average over
M = [10_000_000, 2_500_000, 750_000, 200_000, 40_000, 8000, 1000, 400, 20]  # run this overnight 8h+
# M = np.ones(len(L_big), dtype=int)

time2d = t_c * 3
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

pd.to_pickle(result2d, "data/task2d.pkl")

# %% task 2d - load from pickle
result2d = pd.read_pickle("data/task2d.pkl")

# %% task 2d plot
fig = plt.figure()

result2d = np.array(result2d)

for system_size, time_series in zip(L_big, result2d):
    plt.plot(np.arange(4_000_000) / system_size ** 2,
             np.pad(time_series, (0, 4_000_000 - len(time_series)), 'constant') / system_size,
             label=f"L={system_size}")

transient_cutoff_estimate = np.where(result2d[-1] / L_big[-1] > 1.730)[0][0]
a_0_fit = np.mean((result2d[-1] / L_big[-1])[transient_cutoff_estimate:])
a_0_fit_std = np.std((result2d[-1] / L_big[-1])[transient_cutoff_estimate:])

plt.axhline(a_0_fit, linestyle="--", color="b")
plt.axvline(np.arange(4_000_000)[transient_cutoff_estimate] / L_big[-1] ** 2, linestyle="--", color="r")
plt.plot()
plt.legend()
plt.xlim(-0.05, 1.5)
plt.xlabel("$t/L^2$")
plt.ylabel(r"$\tilde{h}(t; L)/L$")
fig.tight_layout()
plt.savefig("plots/Task2d_collapsed.pgf", format="pgf")
# plt.show()

# %% Task 2e, 2f, 3a & 3b - Find moments, avg heights and std dev
T = 10_000_000  # time steps after t_c

result2g = []
S_matrix = []
average_heights = []
average_square_heights = []
result3a = []

# the 1 and 2 suffixes are for average_heights and average_square_heights respectively
for sys_size, crit_time in zip(L_big, t_c):
    model = OsloModel(sys_size)
    s_1, s_2, s_3, s_4, s_5, s_6, s_7, s_8, s_9, s_10 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    sum1 = 0
    sum2 = 0
    avalanche_sizes = []
    observed_config = []
    for _ in range(crit_time):  # run until t > t_c
        model.run()
    for t in tqdm(range(T)):
        # task 3a
        avalanches = model.run()
        avalanche_sizes.append(avalanches)

        # task 2g
        observed_config.append(model.heights[0])

        # task 3b
        # TODO: merge this with task 3a using vectorised arrays
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

    # task 3a
    result3a.append(logbin(avalanche_sizes, scale=1.3, zeros=False))

    # task 2g
    prob = Counter(observed_config)  # count the first index for each L
    prob = {k: v / T for k, v in prob.items()}  # normalise
    prob = dict(sorted(prob.items()))   # sort by key. Shouldn't take that long
    result2g.append(prob)

S_matrix = np.transpose(S_matrix)
average_heights = np.array(average_heights)
average_square_heights = np.array(average_square_heights)

std_dev = np.sqrt(average_square_heights - average_heights ** 2)

pd.to_pickle(S_matrix, "data/S_matrix.pkl")
pd.to_pickle(average_heights, "data/average_heights.pkl")
pd.to_pickle(std_dev, "data/std_dev.pkl")
pd.to_pickle(result2g, "data/task2g.pkl")
pd.to_pickle(result3a, "data/task3a.pkl")

# %% Load the data
S_matrix = pd.read_pickle("data/S_matrix.pkl")
average_heights = pd.read_pickle("data/average_heights.pkl")
std_dev = pd.read_pickle("data/std_dev.pkl")
result2g = pd.read_pickle("data/task2g.pkl")
result3a = pd.read_pickle("data/task3a.pkl")

# %% task 2e - curve fit method
plt.figure()

# noinspection PyTupleAssignmentBalance
# popt2e, pcov2e = sp.optimize.curve_fit(truncated_series, L_big, average_heights, sigma=std_dev/np.sqrt(T),
#                                        absolute_sigma=True)

# noinspection PyTupleAssignmentBalance
popt2e, pcov2e = sp.optimize.curve_fit(truncated_series, L_big, average_heights)

a_0 = unc.ufloat(popt2e[0], np.sqrt(pcov2e[0, 0]))
a_1 = unc.ufloat(popt2e[1], np.sqrt(pcov2e[1, 1]))
w_1 = unc.ufloat(popt2e[2], np.sqrt(pcov2e[2, 2]))

print(a_0)
print(a_1)
print(w_1)

# plt.plot(L_big, average_heights, ".")
plt.errorbar(L_big, average_heights, yerr=std_dev/np.sqrt(T), fmt=".")
plt.plot(L_big, truncated_series(L_big, *popt2e))
plt.xlabel("System size L")
plt.ylabel("Average height $\\langle h \\rangle$")
plt.show()

#%% task 2e - log method
plt.figure()
a_0_guess = 1.7345

plt.loglog(L_big, a_0_guess - average_heights/L_big, ".--")
# popt2e1, pcov2e1 = np.polyfit(np.log(L_big), np.log(a_0_guess - average_heights/L_big), deg=1, cov=True)
plt.xlabel("System size L")
plt.ylabel(r"$a_0 - \langle h(t ; L)\rangle_t /L$")
plt.show()

# %% task 2f - plot of sigma_h vs L
fig, ax = plt.subplots(1, 2)
ax[0].plot(L_big, std_dev, ".")

# noinspection PyTupleAssignmentBalance
p2f, pcov2f = np.polyfit(np.log(L_big), np.log(std_dev), deg=1, cov=True)

high_density = np.arange(min(L_big), max(L_big))
ax[0].loglog(high_density, np.exp(p2f[1]) * pow(high_density, p2f[0]), label="exponent = %.3f ± %.3f" % (p2f[0], np.sqrt(pcov2f[0, 0])))
ax[0].set_xlabel("L", fontsize=12)
ax[0].set_ylabel("$\sigma_{h}(L)$", fontsize=12)
ax[0].tick_params(axis='both', which='major', labelsize=10)
ax[0].legend(fontsize=10)

# task 2f - check slope against L as L -> infinity
ax[1].plot(L_big, average_heights / L_big, ".")
ax[1].set_xlabel("L", fontsize=12)
ax[1].set_ylabel("$\\langle h(t; L) \\rangle_{t} / L$", fontsize=12)
ax[1].tick_params(axis='both', which='major', labelsize=10)
ax[1].axhline(y=max(average_heights / L_big), color="r", linestyle="--", label=r"$\min {a_0}$ = %.3f" % max(average_heights / L_big))
ax[1].legend(fontsize=10)

print(max(average_heights / L_big))
# plt.savefig("task2f.pgf", format="pgf")
plt.show()

# %% task 2g - plot not collapsed
plt.figure()

for i, data in enumerate(result2g):
    plt.plot(list(data.keys()), list(data.values()), "--", label=f"L = {2 ** (i+2)}")
    plt.bar(list(data.keys()), list(data.values()))

plt.xscale("log")
# plt.plot(std_dev, (np.sqrt(2 * np.pi) * std_dev) ** -1, label="$\sigma_h$")
plt.xlabel("$h$")
plt.ylabel("$P(h; L)$")
plt.legend()

plt.show()

# %%
T = 10_000_000
unpack = [{k: round(v * T) for k, v in prob.items()} for prob in result2g]  # recover the counts from probabilities
# standardise the keys using key - avg_height[i]/std_dev[i]
unpack = [{(k - average_heights[i]) / std_dev[i]: v for k, v in prob.items()} for i, prob in enumerate(unpack)]
# flatten the list of dictionaries
unpack = [item for sublist in unpack for item in sublist]

# %% task 2g - plot collapsed
plt.figure()

for i, data in enumerate(result2g):
    plt.plot((np.fromiter(data.keys(), dtype=np.double) - average_heights[i]) / std_dev[i],
             np.fromiter(data.values(), dtype=np.double) * std_dev[i], ".", label=f"L = {2 ** (i+2)}")

x = np.linspace(-6, 6, 1000)
plt.plot(x, sp.stats.norm.pdf(x, 0, 1), "--", label="$\mathcal{H}(x)$")
plt.xlim(-6, 6)
plt.xlabel(r"$\left(h-\langle h \rangle\right)/ \sigma_h$")
plt.ylabel(r"$\sigma_h P(h; L)$")
plt.legend()
plt.show()

# %% task 2g - Q-Q plot
for i, data in enumerate(result2g):
    fig = sm.qqplot(np.fromiter(data.values(), dtype=np.double) * std_dev[i], line='45')
plt.show()

#%% Task 3b - plot moments
plt.figure()
p = plt.get_cmap("tab10")
result3b = []  # list of $D(1+k-\tau_s)$ for each system size
error3b = []

for i, moment in enumerate(S_matrix):
    plt.loglog(L_big, moment, ".", label=f"k={i + 1}", color=p(i / 10))
    # noinspection PyTupleAssignmentBalance
    p31, pcov3b = np.polyfit(np.log(L_big[-4:]), np.log(moment[-4:]), deg=1, cov=True)  # fit only to last 5 points
    print(p31[0])
    result3b.append(p31[0])
    error3b.append(np.sqrt(pcov3b[0, 0]))
    plt.loglog(L_big, np.exp(p31[1]) * L_big ** p31[0], "--", color=p(i / 10))

error3b = np.array(error3b)
plt.xlabel("L")
plt.ylabel(r"$ \langle s^{k} \rangle $")
plt.legend(fontsize=11)
# plt.savefig("task3b1.pgf", format="pgf")
plt.show()

# %%
plt.figure()
p = plt.get_cmap("tab10")
# ratio = S_matrix[-1][0] / S_matrix[1][0]

for i, moment in enumerate(S_matrix):
    # noinspection PyTupleAssignmentBalance
    p31, pcov3b = np.polyfit(np.log(L_big[-4:]), np.log(moment[-4:]), deg=1, cov=True)  # fit only to last 5 points
    plt.loglog(L_big, moment/L_big ** p31[0], label=f"k={i + 1}", color=p(i / 10))
    # plt.loglog(L_big, np.exp(p31[1]) * L_big ** p31[0], "--", color=p(i / 10))

error3b = np.array(error3b)
plt.xlabel("L")
plt.ylabel(r"$ \langle s^{k} \rangle $")
plt.legend(fontsize=11)
# plt.savefig("task3b1.pgf", format="pgf")
plt.show()

# %% Task 3b - Find both exponents
plt.figure()

k = np.arange(1, 11)
plt.plot(k, result3b, ".")
# noinspection PyTupleAssignmentBalance
p3b1, pcov3b1 = np.polyfit(k, result3b, deg=1, cov=True, w=1 / error3b)
plt.plot(np.arange(0, 12), np.arange(0, 12) * p3b1[0] + p3b1[1], "--")
polynomial = np.poly1d(p3b1)

grad1 = unc.ufloat(p3b1[0], np.sqrt(pcov3b1[0, 0]))
intercept1 = unc.ufloat(p3b1[1], np.sqrt(pcov3b1[1, 1]))

tau_s = 1 + (-intercept1 / grad1)  # relative errors add in quadrature using uncertainties package
print(f"tau_s = {tau_s}")
print(f"D={unc.ufloat(p3b1[0], np.sqrt(pcov3b1[0, 0]))}")

plt.xlim(0, 11)
plt.ylim(0, 23)
plt.xlabel("$k$")
plt.ylabel("$D(1+k-\\tau_s)$")
plt.savefig("task3b2.pgf", format="pgf")
plt.show()

# %% task 3a - plot not collapsed
plt.figure()

idx = np.where((result3a[-1][0] > 10) & (result3a[-1][0] < 10_000))[0]
# noinspection PyTupleAssignmentBalance
p3a1, pcov3a1 = np.polyfit(np.log(result3a[-1][0][idx]), np.log(result3a[-1][1][idx]), deg=1, cov=True)

for i, data in enumerate(result3a):
    plt.loglog(*data, "o--", ms=3, label=f"L={2 ** (i + 2)}")

tau_s_3a = unc.ufloat(p3a1[0], np.sqrt(pcov3a1[0, 0]))
print(f"tau_s = {tau_s_3a}")
plt.xlabel("s")
plt.ylabel(r"$\tilde P_{N}(s; L)$")
plt.legend()
plt.show()

# %% Task 3a - plot collapsed
plt.figure()

for i, (x, y) in enumerate(result3a):
    plt.loglog(x / (2 ** (i + 2)) ** 2.25, x ** (14 / 9) * y, "-", ms=3, label=f"L={2 ** (i + 2)}")

plt.xlabel("$sL^{-D}$")
plt.ylabel("$s^{\\tau_s} \\tilde P_{N}(s; L)$")
plt.legend()

# %%
model2g = OsloModel(2048)
for _ in tqdm(range(4_000_000)):
    model2g.run()

data = model2g.z
plot_acf(x=data, lags=10)
plt.show()

# %%
plt.plot(np.arange(2048), data, ".")
plt.show()
