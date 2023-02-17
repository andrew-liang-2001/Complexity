from oslo import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import scienceplots
import pandas as pd


# %% Task
results = []

for system_size in L_big:
    model = OsloModel(system_size)
    result = []
    for t in tqdm(range(1000000)):
        model.run()
        result.append(model.heights[0])
    results.append(result)

df = pd.DataFrame(results).T.astype(np.ushort)  # optimise storage
df.columns = [f"{L}" for L in L_big]
df.index += 1
df.index.name = "t"
df.to_parquet(engine="pyarrow", path="task2a.parquet")

# %%
df = pd.read_parquet(engine="pyarrow", path="task2a.parquet")

# %%
df.tail()

# %%
df.plot(logy=True, logx=True)
plt.show()
