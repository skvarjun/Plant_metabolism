import numpy as np
import matplotlib.pyplot as plt

# Your numpy array
data = np.array([[-0.30638104,  0.55909002,  0.68125166,  0.00471407],
                 [ 0.1104443,   0.0970974,   0.59365567, -0.59402825],
                 [ 0.01330823,  0.54159542, -0.11939512,  0.60560456],
                 [ 0.50412199,  0.40178521,  0.28428661,  0.39563653],
                 [ 0.3639548,   0.64765151,  0.78415386,  0.60562628]])

# Get number of groups (which is the number of rows)
num_groups = data.shape[0]

# Generate x values for each group
x = np.arange(data.shape[1])

# Plot each group of bars
fig, ax = plt.subplots()
bar_width = 0.15  # Width of each bar group
opacity = 0.8

for i in range(num_groups):
    ax.bar(x + i * bar_width, data[i], bar_width,
           alpha=opacity,
           label=f'Group {i+1}')

ax.set_xlabel('Variables')
ax.set_ylabel('Values')
ax.set_title('Grouped Bar Chart')
ax.set_xticks(x + bar_width * (num_groups - 1) / 2)
ax.set_xticklabels(['Var1', 'Var2', 'Var3', 'Var4'])  # Replace with your actual variable names
ax.legend()

plt.tight_layout()
plt.show()
