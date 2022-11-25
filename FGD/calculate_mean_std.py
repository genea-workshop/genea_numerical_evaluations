'''
calculate mean and std of the natural (*NA) condition files
'''

import glob
import numpy as np

tier = 'upperbody'  # fullbody, upperbody
code = 'F' if tier == 'fullbody' else 'U'
files = glob.glob(f'../data/{tier}/{code}NA/*.npy')

all_data = []
for file in files:
    data = np.load(file)
    all_data.append(data)

all_data = np.vstack(all_data)

print(all_data.shape)

mean = np.mean(all_data, axis=0)
std = np.std(all_data, axis=0)
print(mean.shape)
print(std.shape)

print(*mean, sep=',')
print(*std, sep=',')
