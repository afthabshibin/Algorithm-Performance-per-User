
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="white", context="paper")

summary_data = pd.read_csv("path to saved CSV file obtained from 'Table_proportion_of_users.py'")
plt.figure(figsize=(12, 6))

ax1 = sns.barplot(x="Algorithm", y="Proportion of Users with Score = 0", gap=.15, hue="Dataset", palette="colorblind",
                  data=summary_data)

for container in ax1.containers:
    ax1.bar_label(container, fontsize=7, fmt='%.3f', padding=3)

ax1.set_ylabel('Proportion of Users With NDCG = 0', fontsize=12)
ax1.set_xlabel('Algorithm', fontsize=12)


plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("path to save the figure",
            bbox_inches='tight',
            dpi=300)

plt.show()
