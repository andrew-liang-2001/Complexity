# import numpy as np
# import matplotlib.pyplot as plt
#
# L = [100, 200, 400]
# res = [25000, 141420, 800000]
#
# plt.plot(L, res, ".")
# plt.plot(np.arange(400), np.arange(400) ** 2.5)
# plt.show()
#

# %%

import oslo
from tqdm import tqdm

# %%

model1 = oslo.OsloModel(256)

#%%
for _ in tqdm(range(300000)):
    model1.run()

#%%
for i in range(1, 0):
    print("hi")
