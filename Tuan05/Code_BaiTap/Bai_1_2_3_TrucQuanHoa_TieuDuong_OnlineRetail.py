# Bài tập 1.2.3 – Trực quan hóa dữ liệu tiểu đường + EDA Online Retail
# Chạy trong Jupyter (copy từng khối) hoặc: python Bai_1_2_3_TrucQuanHoa_TieuDuong_OnlineRetail.py  (dữ liệu nằm trong thư mục data/)



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



pima = pd.read_csv('data/pima_diabetes.csv')
print('Kích thước:', pima.shape)
display(pima.head())


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


fig, ax = plt.subplots(1, 2, figsize=(12, 5))
vc = pima_clean['Outcome'].value_counts()
ax[0].pie(vc, labels=['Không mắc (0)', 'Mắc bệnh (1)'], autopct='%1.1f%%', colors=['#55A868', '#C44E52'],
          startangle=90, explode=[0, .05])
ax[0].set_title('Tỷ lệ mắc tiểu đường')
sns.countplot(data=pima_clean, x='Outcome', palette=['#55A868', '#C44E52'], ax=ax[1])
ax[1].set_title('Số bệnh nhân theo Outcome')
plt.tight_layout()
plt.show()


cols = pima_clean.columns.drop('Outcome')
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for ax, col in zip(axes.flat, cols):
    sns.histplot(data=pima_clean, x=col, hue='Outcome', kde=True, element='step', stat='density',
                 common_norm=False, palette=['#55A868', '#C44E52'], ax=ax)
    ax.set_title(col)
plt.suptitle('Phân bố các đặc trưng theo Outcome (sau khi xử lý giá trị 0)', fontsize=15)
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for ax, col in zip(axes.flat, cols):
    sns.boxplot(data=pima_clean, x='Outcome', y=col, palette=['#55A868', '#C44E52'], ax=ax)
    ax.set_title(col)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 8))
sns.heatmap(pima_clean.corr(), annot=True, fmt='.2f', cmap='YlGnBu')
plt.title('Ma trận tương quan – Pima Diabetes', fontsize=14)
plt.show()


g = sns.pairplot(pima_clean[['Glucose', 'BMI', 'Age', 'Insulin', 'Outcome']], hue='Outcome',
                 palette=['#55A868', '#C44E52'], diag_kind='kde', plot_kws={'alpha': .5, 's': 18})
g.fig.suptitle('Pairplot các đặc trưng chính', y=1.02)
plt.show()


retail = pd.read_excel('data/Online_Retail.xlsx', dtype={'InvoiceNo': str, 'StockCode': str})
print('Kích thước:', retail.shape)
display(retail.head())


retail.info()


print('Giá trị thiếu:')
print(retail.isnull().sum())
print('\nDòng trùng lặp:', retail.duplicated().sum())
print('Hóa đơn bị hủy (InvoiceNo bắt đầu bằng C):', retail['InvoiceNo'].str.startswith('C').sum())
print('Quantity <= 0:', (retail['Quantity'] <= 0).sum(), '| UnitPrice <= 0:', (retail['UnitPrice'] <= 0).sum())


display(retail[['Quantity', 'UnitPrice']].describe().T)


rt = retail.drop_duplicates()
rt = rt[~rt['InvoiceNo'].str.startswith('C')]
rt = rt[(rt['Quantity'] > 0) & (rt['UnitPrice'] > 0)]
rt = rt.dropna(subset=['CustomerID'])
rt['CustomerID'] = rt['CustomerID'].astype(int)
rt['TotalPrice'] = rt['Quantity'] * rt['UnitPrice']
rt['YearMonth'] = rt['InvoiceDate'].dt.to_period('M')
rt['DayOfWeek'] = rt['InvoiceDate'].dt.day_name()
rt['Hour'] = rt['InvoiceDate'].dt.hour
print('Sau làm sạch:', rt.shape)
print(f"Số hóa đơn: {rt['InvoiceNo'].nunique():,} | Số khách hàng: {rt['CustomerID'].nunique():,} | "
      f"Số sản phẩm: {rt['StockCode'].nunique():,} | Số quốc gia: {rt['Country'].nunique()}")
print(f"Tổng doanh thu: £{rt['TotalPrice'].sum():,.0f}")


monthly = rt.groupby('YearMonth').agg(Revenue=('TotalPrice', 'sum'), Orders=('InvoiceNo', 'nunique'))
fig, ax1 = plt.subplots(figsize=(14, 5))
ax1.bar(monthly.index.astype(str), monthly['Revenue'] / 1e3, color='#4C72B0', label='Doanh thu (nghìn £)')
ax1.set_ylabel('Doanh thu (nghìn £)')
ax2 = ax1.twinx()
ax2.plot(monthly.index.astype(str), monthly['Orders'], color='#C44E52', marker='o', label='Số hóa đơn')
ax2.set_ylabel('Số hóa đơn'); ax2.grid(False)
ax1.set_title('Doanh thu và số hóa đơn theo tháng (12/2010 – 12/2011)', fontsize=14)
ax1.tick_params(axis='x', rotation=45)
fig.legend(loc='upper left', bbox_to_anchor=(0.07, 0.9))
plt.tight_layout()
plt.show()
display(monthly)


country = rt.groupby('Country')['TotalPrice'].sum().sort_values(ascending=False)
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
country.head(10).plot.barh(ax=ax[0], color='#55A868'); ax[0].invert_yaxis()
ax[0].set_title('Top 10 quốc gia theo doanh thu'); ax[0].set_xlabel('£')
country.drop('United Kingdom').head(10).plot.barh(ax=ax[1], color='#8172B2'); ax[1].invert_yaxis()
ax[1].set_title('Top 10 quốc gia (không tính UK)'); ax[1].set_xlabel('£')
plt.tight_layout()
plt.show()
print(f"UK chiếm {100 * country['United Kingdom'] / country.sum():.1f}% doanh thu")
display((country.head(8) / 1e3).round(1).to_frame('Doanh thu (nghìn £)'))


top_qty = rt.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
top_rev = rt.groupby('Description')['TotalPrice'].sum().sort_values(ascending=False).head(10)
fig, ax = plt.subplots(1, 2, figsize=(18, 6))
sns.barplot(x=top_qty.values, y=top_qty.index, ax=ax[0], palette='Blues_r'); ax[0].set_title('Top 10 sản phẩm theo số lượng bán')
sns.barplot(x=top_rev.values, y=top_rev.index, ax=ax[1], palette='Greens_r'); ax[1].set_title('Top 10 sản phẩm theo doanh thu (£)')
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(1, 2, figsize=(16, 5))
order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dow = rt.groupby('DayOfWeek')['InvoiceNo'].nunique().reindex(order).fillna(0).astype(int)
ax[0].bar(dow.index, dow.values, color='#DD8452'); ax[0].set_title('Số hóa đơn theo ngày trong tuần')
ax[0].tick_params(axis='x', rotation=30)
hour = rt.groupby('Hour')['InvoiceNo'].nunique()
ax[1].plot(hour.index, hour.values, marker='o', color='#4C72B0'); ax[1].set_title('Số hóa đơn theo giờ trong ngày')
ax[1].set_xlabel('Giờ')
plt.tight_layout()
plt.show()
print(dow)
print('\nSố hóa đơn theo giờ:', hour.to_dict())


cust = rt.groupby('CustomerID').agg(Orders=('InvoiceNo', 'nunique'), Revenue=('TotalPrice', 'sum'))
fig, ax = plt.subplots(1, 2, figsize=(16, 5))
sns.histplot(np.log10(cust['Revenue']), bins=40, ax=ax[0], color='#4C72B0')
ax[0].set_title('Phân bố doanh thu theo khách hàng (log10 £)'); ax[0].set_xlabel('log10(Revenue)')
sns.boxplot(x=rt.loc[rt['UnitPrice'] < 20, 'UnitPrice'], ax=ax[1], color='#DD8452')
ax[1].set_title('Boxplot đơn giá (UnitPrice < 20£)')
plt.tight_layout()
plt.show()
display(cust.describe().T)


top20 = cust['Revenue'].sort_values(ascending=False)
share = top20.head(int(len(top20) * 0.2)).sum() / top20.sum()
print(f'20% khách hàng lớn nhất đóng góp {100 * share:.1f}% doanh thu (nguyên lý Pareto)')
