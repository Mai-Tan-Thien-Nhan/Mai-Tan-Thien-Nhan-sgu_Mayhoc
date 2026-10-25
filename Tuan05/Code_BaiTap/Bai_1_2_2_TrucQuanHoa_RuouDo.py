# Bài tập 1.2.2 – Trực quan hóa dữ liệu Red Wine Quality
# Chạy trong Jupyter (copy từng khối) hoặc: python Bai_1_2_2_TrucQuanHoa_RuouDo.py  (dữ liệu nằm trong thư mục data/)



import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: f'{x:,.3f}')
sns.set_theme(style='whitegrid', palette='deep')
plt.rcParams['figure.dpi'] = 100
os.makedirs('figures', exist_ok=True)
os.makedirs('reports', exist_ok=True)
try:
    from IPython.display import display
except ImportError:
    display = print



wine = pd.read_csv('data/winequality-red.csv')
print('Kích thước:', wine.shape)
display(wine.head())


wine['quality_group'] = pd.cut(wine['quality'], bins=[2, 4, 6, 8], labels=['low', 'medium', 'high'])
display(wine.groupby('quality_group', observed=True)[['alcohol', 'volatile acidity', 'sulphates', 'citric acid', 'density', 'pH']].agg(['mean', 'median']).round(3))


features = wine.columns.drop(['quality', 'quality_group'])
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
for ax, col in zip(axes.flat, features):
    ax.hist(wine[col], bins=30, color='#4C72B0', edgecolor='white')
    ax.axvline(wine[col].mean(), color='red', ls='--', lw=1.2, label='mean')
    ax.axvline(wine[col].median(), color='green', ls='-', lw=1.2, label='median')
    ax.set_title(col, fontsize=12)
axes.flat[0].legend()
axes.flat[-1].axis('off')
plt.suptitle('Phân bố các đặc trưng hóa lý của rượu vang đỏ', fontsize=16)
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(1, 2, figsize=(14, 5))
counts = wine['quality'].value_counts().sort_index()
ax[0].bar(counts.index.astype(str), counts.values, color='#DD8452')
for i, v in enumerate(counts.values):
    ax[0].text(i, v + 10, str(v), ha='center')
ax[0].set_title('Số mẫu theo quality (Matplotlib)')
ax[0].set_xlabel('quality'); ax[0].set_ylabel('Số mẫu')
sns.countplot(data=wine, x='quality_group', ax=ax[1], palette='Set2', order=['low', 'medium', 'high'])
ax[1].set_title('Số mẫu theo nhóm chất lượng (Seaborn)')
plt.tight_layout()
plt.show()


key = ['alcohol', 'volatile acidity', 'sulphates', 'citric acid']
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
for ax, col in zip(axes, key):
    sns.boxplot(data=wine, x='quality', y=col, ax=ax, palette='coolwarm')
    ax.set_title(f'{col} theo quality')
plt.tight_layout()
plt.show()


corr = wine.drop(columns='quality_group').corr()
plt.figure(figsize=(12, 9))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0, linewidths=.5)
plt.title('Ma trận tương quan Pearson – Red Wine', fontsize=15)
plt.show()


display(corr['quality'].drop('quality').sort_values(ascending=False).to_frame('corr_with_quality'))


fig, ax = plt.subplots(1, 2, figsize=(16, 6))
sns.scatterplot(data=wine, x='alcohol', y='volatile acidity', hue='quality_group',
                palette={'low': 'red', 'medium': 'gray', 'high': 'green'}, alpha=.7, ax=ax[0])
ax[0].set_title('alcohol vs volatile acidity theo nhóm chất lượng')
sns.violinplot(data=wine, x='quality_group', y='alcohol', palette='Set2', ax=ax[1], order=['low', 'medium', 'high'])
ax[1].set_title('Phân bố alcohol theo nhóm chất lượng (violin)')
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize=(8, 5))
sns.regplot(data=wine, x='alcohol', y='quality', scatter_kws={'alpha': .3}, line_kws={'color': 'red'}, ax=ax)
ax.set_title('Hồi quy tuyến tính alcohol – quality')
fig.savefig('figures/wine_alcohol_quality.png', dpi=200, bbox_inches='tight')
fig.savefig('figures/wine_alcohol_quality.pdf', bbox_inches='tight')
plt.close(fig)
print('Đã lưu figures/wine_alcohol_quality.png và .pdf')
