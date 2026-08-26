#!/usr/bin/env python
"""
Make handy plots for presentation.
Just for preprocessing, the normalization, standardization, minmax, robust scaler
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
#rng = np.random.default_rng(seed=42)
example_data = pd.read_csv('C:/Users/Bipo1/Downloads/PPPL Project Materials/SURGENEW/runs/generated_datasets/normal_2d.csv')

plt.figure()
plt.plot(example_data['x1'], example_data['x2'], marker='o', linewidth=0)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Random Gaussian 2D Data')
plt.savefig('normal_2d_plot.png')
plt.show()

#robust scaled
r_scaled = RobustScaler().fit_transform(example_data[['x1', 'x2']])
plt.figure()
plt.plot(r_scaled[:, 0], r_scaled[:, 1], marker='o', linewidth=0)
plt.xlabel('Robust x1')
plt.ylabel('Robust x2')
plt.title('Robust Scaled 2D Data')
plt.savefig('robust_scaled_2d_plot.png')
plt.show()
#minmax scaled
m_scaled = MinMaxScaler().fit_transform(example_data[['x1', 'x2']])
plt.figure()
plt.plot(m_scaled[:, 0], m_scaled[:, 1], marker='o', linewidth=0)
plt.xlabel('MinMax x1')
plt.ylabel('MinMaxx2')
plt.title('MinMax Scaled 2D Data')
plt.savefig('minmax_scaled_2d_plot.png')
plt.show()#standard scaled
s_scaled = StandardScaler().fit_transform(example_data[['x1', 'x2']])
plt.figure()
plt.plot(s_scaled[:, 0], s_scaled[:, 1], marker='o', linewidth=0)
plt.xlabel('Standard x1')
plt.ylabel('Standard x2')
plt.title('Standard Scaled 2D Data')
plt.savefig('standard_scaled_2d_plot.png')
plt.show()