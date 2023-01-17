from oslo import *
import matplotlib.pyplot as plt

#%% test1
model = OsloModel(16)
model.run(1000)  # run the model for a long time to reach the steady state

result = []
for i in tqdm(range(5000)):
    model.run()
    result.append(model.heights[0])

print(np.mean(result))
