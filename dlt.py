import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Your data
A = ["ant", 'cat', 'dog', 'mouse']
B = [[1, 4, 2, 7, 4, 1], [4, 7, 2, 3, 4, 1], [1, 8, 9, 3, 4, 7], [8, 2, 7, 6, 3, 7]]

# Create DataFrame
df = pd.DataFrame(B, index=A, columns=[f'Feature_{i+1}' for i in range(len(B[0]))])

# Plot heatmap
plt.figure(figsize=(10, 6))
ax = sns.heatmap(df, annot=True, cmap='YlGnBu', linewidths=0.5)

# Set y-axis labels to horizontal
plt.yticks(rotation=0)

plt.title('Heatmap of Data')
plt.show()
