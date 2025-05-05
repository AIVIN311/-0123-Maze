import pandas as pd
import matplotlib.pyplot as plt

def plot_hs_curve(df, save_path="hs_curve_many.png"):
    fig, ax1 = plt.subplots()
    ax1.plot(df['episode'], df['entropy'], 'b-', label='Entropy (H)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Entropy', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()
    ax2.plot(df['episode'], df['wisdom_density'], 'r-', label='Wisdom Density (S)')
    ax2.set_ylabel('Wisdom Density', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    fig.tight_layout()
    fig.legend(loc='upper center')
    plt.title('0-1-2-3 H-S Curve')
    plt.savefig(save_path)
    plt.close()
