# Bài tập 1.3.4 – AutoViz trên dữ liệu Marketing Campaign
# Chạy trong Jupyter (copy từng khối) hoặc: python Bai_1_3_4_AutoViz_Marketing.py  (dữ liệu nằm trong thư mục data/)



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



marketing = pd.read_csv('data/marketing_campaign.csv', sep='\t')
print(marketing.shape)
display(marketing.head())


mk = marketing.copy()
mk['Dt_Customer'] = pd.to_datetime(mk['Dt_Customer'], format='%d-%m-%Y')
mk['Age'] = 2014 - mk['Year_Birth']
mk['Children'] = mk['Kidhome'] + mk['Teenhome']
mk['TotalSpent'] = mk[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].sum(axis=1)
mk = mk.drop(columns=['Z_CostContact', 'Z_Revenue'])
print('Giá trị thiếu:', mk.isnull().sum()[mk.isnull().sum() > 0].to_dict())
display(mk[['Age', 'Income', 'Children', 'TotalSpent', 'Response']].describe().T)


from autoviz.AutoViz_Class import AutoViz_Class

AV = AutoViz_Class()
mk_av = mk.drop(columns=['ID', 'Dt_Customer'])
# AutoViz nhận diện biến phân loại ở dạng object -> chuyển các cột chuỗi về object (cần với pandas >= 3)
for c in ['Education', 'Marital_Status']:
    mk_av[c] = mk_av[c].astype(object)
dft = AV.AutoViz(filename='', dfte=mk_av, depVar='Response', verbose=2, chart_format='png',
                 max_rows_analyzed=5000, max_cols_analyzed=30, save_plot_dir='reports/autoviz')


from IPython.display import Image, display
import glob
for f in sorted(glob.glob('reports/autoviz/Response/*.png')):
    print(os.path.basename(f))
    display(Image(filename=f))
