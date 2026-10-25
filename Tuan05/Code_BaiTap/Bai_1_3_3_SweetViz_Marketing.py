# Bài tập 1.3.3 – SweetViz trên dữ liệu Marketing Campaign
# Chạy trong Jupyter (copy từng khối) hoặc: python Bai_1_3_3_SweetViz_Marketing.py  (dữ liệu nằm trong thư mục data/)



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


import sweetviz as sv

cfg = sv.FeatureConfig(skip=['ID'], force_num=['Response'])
report = sv.analyze(mk, target_feat='Response', feat_cfg=cfg)
report.show_html('reports/sweetviz_marketing.html', open_browser=False, layout='widescreen')


# So sánh nội bộ: khách hàng đã phản hồi chiến dịch (Response=1) với khách chưa phản hồi
compare = sv.compare_intra(mk, mk['Response'] == 1, ['Responded', 'Not responded'], feat_cfg=sv.FeatureConfig(skip=['ID']))
compare.show_html('reports/sweetviz_compare_response.html', open_browser=False, layout='widescreen')


display(mk.groupby('Response')[['Income', 'Recency', 'TotalSpent', 'NumWebPurchases', 'NumCatalogPurchases', 'Children', 'Age']].mean().round(1))
