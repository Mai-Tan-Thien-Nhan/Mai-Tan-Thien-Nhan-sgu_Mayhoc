# Bài tập 1.1.4 – Thống kê mô tả dữ liệu bệnh tiểu đường (Pima)
# Chạy trong Jupyter (copy từng khối) hoặc: python Bai_1_1_4_ThongKeMoTa_TieuDuong.py  (dữ liệu nằm trong thư mục data/)



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


pima = pd.read_csv('data/pima_diabetes.csv')
print('Kích thước:', pima.shape)
display(pima.head())


pima.info()
print('\nGiá trị thiếu (NaN):', pima.isnull().sum().sum(), '| Dòng trùng:', pima.duplicated().sum())


display(pima.describe().T)


zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
zeros = pd.DataFrame({'n_zero': (pima[zero_cols] == 0).sum(),
                      'pct_zero': 100 * (pima[zero_cols] == 0).mean()})
display(zeros)


pima_clean = pima.copy()
pima_clean[zero_cols] = pima_clean[zero_cols].replace(0, np.nan)
print('NaN sau khi thay 0:')
print(pima_clean.isnull().sum())
for c in zero_cols:
    pima_clean[c] = pima_clean[c].fillna(pima_clean.groupby('Outcome')[c].transform('median'))
print('\nNaN sau khi điền trung vị theo Outcome:', pima_clean.isnull().sum().sum())


pima_stats = descriptive_stats(pima_clean)
display(pima_stats)


o = pima_clean['Outcome'].value_counts().to_frame('count')
o['percent'] = 100 * o['count'] / len(pima_clean)
display(o)


display(pima_clean.groupby('Outcome').agg(['mean', 'median']).T.round(2))


display(iqr_outliers(pima_clean.drop(columns='Outcome')).sort_values('n_outliers', ascending=False))
