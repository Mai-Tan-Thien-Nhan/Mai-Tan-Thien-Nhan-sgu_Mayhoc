# Bài tập 1.1.3 – Thống kê mô tả dữ liệu Red Wine Quality
# Chạy trong Jupyter (copy từng khối) hoặc: python Bai_1_1_3_ThongKeMoTa_RuouDo.py  (dữ liệu nằm trong thư mục data/)



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



def descriptive_stats(df):
    """Tính đầy đủ các chỉ số thống kê mô tả cho các cột số (numpy + scipy.stats)."""
    rows = {}
    for col in df.select_dtypes(include='number').columns:
        x = df[col].dropna().values
        rows[col] = {
            'mean': np.mean(x),
            'median': np.median(x),
            'mode': stats.mode(x, keepdims=False).mode,
            'variance': np.var(x, ddof=1),
            'std': np.std(x, ddof=1),
            'min': np.min(x),
            'max': np.max(x),
            'range': np.ptp(x),
            'P60': np.percentile(x, 60),
            'Q1': np.quantile(x, 0.25),
            'Q3': np.quantile(x, 0.75),
            'IQR': stats.iqr(x),
            'skewness': stats.skew(x),
            'kurtosis': stats.kurtosis(x),
        }
    return pd.DataFrame(rows).T


def iqr_outliers(df):
    """Đếm số giá trị ngoại lai theo quy tắc 1.5*IQR cho từng cột số."""
    res = {}
    for col in df.select_dtypes(include='number').columns:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n = ((df[col] < lo) | (df[col] > hi)).sum()
        res[col] = {'lower_fence': lo, 'upper_fence': hi, 'n_outliers': n, 'pct_outliers': 100 * n / len(df)}
    return pd.DataFrame(res).T


wine = pd.read_csv('data/winequality-red.csv')
print('Kích thước:', wine.shape)
display(wine.head())


wine.info()


print('Số giá trị thiếu mỗi cột:')
print(wine.isnull().sum())
print('\nSố dòng trùng lặp:', wine.duplicated().sum())


wine_nodup = wine.drop_duplicates()
print('Sau khi loại trùng:', wine_nodup.shape)


display(wine.describe().T)


wine_stats = descriptive_stats(wine)
display(wine_stats)


x = wine['alcohol']
print(f"Mean            : {np.mean(x):.3f}")
print(f"Median          : {np.median(x):.3f}")
print(f"Mode            : {stats.mode(x, keepdims=False).mode:.3f}")
print(f"Variance        : {np.var(x):.3f}")
print(f"Std             : {np.std(x):.3f}")
print(f"Range           : {np.max(x) - np.min(x):.3f}  (min={np.min(x)}, max={np.max(x)})")
print(f"Percentile 60   : {np.percentile(x, 60):.3f}")
print(f"Quartile Q3     : {np.quantile(x, 0.75):.3f}")
print(f"IQR             : {stats.iqr(x):.3f}")


q = wine['quality'].value_counts().sort_index().to_frame('count')
q['percent'] = 100 * q['count'] / q['count'].sum()
display(q)


wine_out = iqr_outliers(wine)
display(wine_out.sort_values('n_outliers', ascending=False))


wine['quality_group'] = pd.cut(wine['quality'], bins=[2, 4, 6, 8], labels=['low', 'medium', 'high'])
display(wine.groupby('quality_group', observed=True)[['alcohol', 'volatile acidity', 'sulphates', 'citric acid', 'density', 'pH']].agg(['mean', 'median']).round(3))
