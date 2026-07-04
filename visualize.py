import matplotlib.pyplot as plt
import numpy as np

#Comment out the stylistic tweaks if it isn't to your liking
plt.style.use('dark_background')
plt.rcParams.update({
    'axes.facecolor': '#181C14',
    'figure.facecolor': 'none',
    'savefig.facecolor': 'none',
    'axes.edgecolor': '#697565',
    'axes.labelcolor': '#CFC4B3',
    'xtick.color': '#CFC4B3',
    'ytick.color': '#CFC4B3',
    'text.color': '#ECDFCC',
    'axes.titleweight': 'bold',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'font.family': 'serif'
})

def learning_curve(X, y):
    
    plt.plot(X, y, color="#D5D0C3")

    plt.xlabel("Iteration")
    plt.ylabel("Binary Cross-Entropy Cost")
    plt.title("Learning Curve")

    plt.grid(True, alpha=0.2, linestyle= "--", linewidth="0.7")

    plt.tight_layout()
    plt.show()
