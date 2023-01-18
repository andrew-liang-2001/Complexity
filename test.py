from oslo import *
from tqdm import tqdm

#%% test1
model = OsloModel(16)
for _ in range(5000):  # run the model for a long time to reach the steady state
    model.run()

result = []
for i in tqdm(range(500000)):  # take average over t=5000
    model.run()
    result.append(model.heights[0])

print(np.mean(result))

#%% test2