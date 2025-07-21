import pandas as pd
import matplotlib.pyplot as plt

# Load the file
df = pd.read_csv("lse_accuracy_vs_iterations.csv")  

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(df['MAX_ITER'], df['Accuracy'], marker='o')
plt.xlabel('Max Iterations')
plt.ylabel('Accuracy')
plt.title('Hamming Kernel - LSE Algorithm Accuracy')
plt.grid(True)
plt.tight_layout()
plt.show()
