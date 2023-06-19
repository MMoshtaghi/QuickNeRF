import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import ipdb

baseline_file = "./run-baseline_orchids_test-tag-psnr.csv"
our_file = "./run-hash_orchids_test-tag-psnr.csv"

baseline_df = pd.read_csv(baseline_file)
our_df = pd.read_csv(our_file)

baseline_df = baseline_df[:50000]
our_df = our_df[:10000]

x1 = baseline_df["Step"].tolist()
y1 = baseline_df["Value"].tolist()

x2 = our_df["Step"].tolist()
y2 = our_df["Value"].tolist()

# Smoothing interpolation
interpolation_num = 100
x1_smooth = np.linspace(min(x1), max(x1), interpolation_num)  # Adjust the number of points as needed
y1_smooth = make_interp_spline(x1, y1)(x1_smooth)
x2_smooth = np.linspace(min(x2), max(x2), interpolation_num)  # Adjust the number of points as needed
y2_smooth = make_interp_spline(x2, y2)(x2_smooth)

# Plot the curve
plt.plot(x1_smooth, y1_smooth, label='NeRF')
plt.plot(x2_smooth, y2_smooth, label='HashNeRF')

# Customize the plot
plt.xlabel('Steps')
plt.ylabel('PSNR')
plt.title('Convergence Comparison')
plt.legend()

# Display the plot
plt.show()
