from oslo import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.constants as const
from collections import Counter

plt.style.use(["science"])
plt.rcParams.update({"font.size": 16,
                     "figure.figsize": (7.5, 4.5),
                     "axes.labelsize": 15,
                     "legend.fontsize": 12,
                     "xtick.labelsize": 13,
                     "ytick.labelsize": 13,
                     'axes.prop_cycle': plt.cycler(color=plt.cm.tab10.colors)})

#%% test1 - L=16
model = OsloModel(16)
for _ in range(400):  # run the model for a long time to reach the steady state
    model.run()

result = []
for i in tqdm(range(3_000_000)):  # take average over t=5000
    model.run()
    result.append(model.heights[0])

print(np.mean(result), np.std(result) / np.sqrt(3_000_000))

# %% test1 - L=32
model = OsloModel(32)
for _ in range(1500):  # run the model for a long time to reach the steady state
    model.run()

result = []
for i in tqdm(range(3_000_000)):  # take average over t=5000
    model.run()
    result.append(model.heights[0])

print(np.mean(result), np.std(result) / np.sqrt(3_000_000))

#%% test2 - transient and recurrent configurations
# TODO: seed this before submission to ensure N_R matches


def recurrent_configurations(L: int):
    """
    Return the number of recurrent configurations.
    :param L: system size
    :return: number of recurrent configurations
    """
    return 1/np.sqrt(5) * (const.golden * (1+const.golden) ** L + (1/(const.golden * (1+const.golden) ** L)))


t_c = [200, 200, 200, 200]
T = [10_000_000, 10_000_000, 10_000_000, 10_000_000]  # time steps after t_c

for i, (sys_size, crit_time, time_after_tc) in enumerate(zip([2, 3, 4, 5], t_c, T)):
    model = OsloModel(sys_size)
    observed_config = set()
    for _ in range(crit_time):  # run until t > t_c
        model.run()
    for t in tqdm(range(time_after_tc), disable=time_after_tc < 500_000):
        model.run()
        observed_config.add(tuple(model.heights))
    if sys_size == 1:
        print(observed_config)
    print(f"System size: {sys_size}, N_R: {len(observed_config)}")
    print(f"N_R is analytically: {recurrent_configurations(sys_size)}")  # compare with equation

# %% BTW model

model = OsloModel(8, z_threshold_max=1)  # z_threshold_max=1 is equivalent to only allowing z_th = 1 i.e. BTW model
TestCondition = True
true_result = np.tril(np.ones((8, 8), dtype=int), -1) + np.identity(8, dtype=int)

for _ in range(1000):  # run the model for a long time to reach the steady state
    model.run()
for _ in range(1000):
    if np.all(model.heights_matrix() != true_result):
        raise ValueError("BTW model is not working")

print("BTW test passed")
