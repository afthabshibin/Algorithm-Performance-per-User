
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="white", context="paper")

summary_data = pd.read_csv("C:/Users/aftha/OneDrive/Desktop/Final graphs/Final_summary_results_proportional.csv")

plt.figure(figsize=(12, 6))

ax1 = sns.barplot(x="Algorithm", y="Proportion of Users with Score = 1", gap=.15, hue="Dataset", palette="colorblind",
                  data=summary_data)

for container in ax1.containers:
    ax1.bar_label(container, fontsize=7, fmt='%.3f', padding=3)

ax1.set_ylabel('Proportion of Users With NDCG = 1', fontsize=12)
ax1.set_xlabel('Algorithm', fontsize=12)


plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("C:/Users/aftha/OneDrive/Desktop/Final Graphs/Proportion of Users With an Ndcg Score of 1.jpg",
            bbox_inches='tight',
            dpi=300)

plt.show()
