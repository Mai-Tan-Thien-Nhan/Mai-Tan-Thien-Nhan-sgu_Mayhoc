# Bài tập 2.3.3 – Naive Bayes dự đoán hành vi khách hàng
# Chạy trong Jupyter (copy từng khối) hoặc: python Bai_2_3_3_NaiveBayes_KhachHang.py  (dữ liệu nằm trong thư mục data/)



import warnings
warnings.filterwarnings('ignore')
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix, classification_report, ConfusionMatrixDisplay, RocCurveDisplay)

pd.set_option('display.max_columns', 40)
pd.set_option('display.width', 200)
sns.set_theme(style='whitegrid')
RS = 24  # random_state dùng xuyên suốt như bài mẫu
try:
    from IPython.display import display
except ImportError:
    display = print



def evaluate(model, X_test, y_test, name, pos_label=1, average='binary'):
    """Tính các chỉ số đánh giá trên tập test và trả về dict."""
    y_pred = model.predict(X_test)
    res = {'model': name,
           'accuracy': accuracy_score(y_test, y_pred),
           'precision': precision_score(y_test, y_pred, pos_label=pos_label, average=average, zero_division=0),
           'recall': recall_score(y_test, y_pred, pos_label=pos_label, average=average, zero_division=0),
           'f1': f1_score(y_test, y_pred, pos_label=pos_label, average=average, zero_division=0)}
    if hasattr(model, 'predict_proba') and average == 'binary':
        proba = model.predict_proba(X_test)[:, list(model.classes_).index(pos_label)]
        res['roc_auc'] = roc_auc_score((np.asarray(y_test) == pos_label).astype(int), proba)
    return res


def plot_cm(models, X_test, y_test, labels, title):
    """Vẽ confusion matrix cho danh sách (tên, model)."""
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.2))
    axes = np.atleast_1d(axes)
    for ax, (name, m) in zip(axes, models):
        ConfusionMatrixDisplay.from_estimator(m, X_test, y_test, display_labels=labels, cmap='Blues', ax=ax, colorbar=False)
        ax.set_title(name)
        ax.grid(False)
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_cv_curve(cv, param, xlabel, title, n_folds):
    """Vẽ mean ± 1 SE của train/test score theo một siêu tham số (như bài mẫu)."""
    r = pd.DataFrame(cv.cv_results_)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(r[f'param_{param}'].astype(float), r['mean_train_score'], yerr=r['std_train_score'] / np.sqrt(n_folds),
                label='Mean $\\pm$ 1 SE training scores', marker='o')
    ax.errorbar(r[f'param_{param}'].astype(float), r['mean_test_score'], yerr=r['std_test_score'] / np.sqrt(n_folds),
                label='Mean $\\pm$ 1 SE testing scores', marker='s')
    ax.set_xlabel(xlabel); ax.set_ylabel('ROC AUC'); ax.set_title(title); ax.legend()
    plt.show()
    return r[[f'param_{param}', 'mean_train_score', 'mean_test_score', 'std_test_score']].round(4)


ads = pd.read_csv('data/Social_Network_Ads.csv')
print(ads.shape)
display(ads.head())
print(ads.isnull().sum().sum(), 'giá trị thiếu')
display(ads.describe().round(1))
display(ads.groupby('Purchased')[['Age', 'EstimatedSalary']].mean().round(1))


fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
sns.countplot(data=ads, x='Purchased', ax=ax[0], palette='Set2'); ax[0].set_title('Phân bố Purchased')
sns.histplot(data=ads, x='Age', hue='Purchased', kde=True, ax=ax[1], palette='Set1'); ax[1].set_title('Age theo Purchased')
sns.scatterplot(data=ads, x='Age', y='EstimatedSalary', hue='Purchased', ax=ax[2], palette='Set1'); ax[2].set_title('Age – EstimatedSalary')
plt.tight_layout(); plt.show()


Xs = ads.drop(columns=['User ID', 'Purchased']).copy()
Xs['Gender'] = (Xs['Gender'] == 'Male').astype(int)
ys = ads['Purchased'].values
Xs_train, Xs_test, ys_train, ys_test = train_test_split(Xs, ys, test_size=0.25, random_state=0, stratify=ys)
sc = StandardScaler().fit(Xs_train[['Age', 'EstimatedSalary']])
def scale(X):
    X = X.copy(); X[['Age', 'EstimatedSalary']] = sc.transform(X[['Age', 'EstimatedSalary']]); return X
Xs_train_s, Xs_test_s = scale(Xs_train), scale(Xs_test)
print(Xs_train_s.shape, Xs_test_s.shape)


gnb = GaussianNB().fit(Xs_train_s, ys_train)
print('Prior P(y):', gnb.class_prior_.round(3))
print('Trung bình theo lớp (theta_):\n', pd.DataFrame(gnb.theta_, columns=Xs.columns, index=['No', 'Yes']).round(3))
gnb2 = GaussianNB().fit(Xs_train_s[['Age', 'EstimatedSalary']], ys_train)
res_ads = pd.DataFrame([evaluate(gnb, Xs_test_s, ys_test, 'GaussianNB (Gender, Age, Salary)'),
                        evaluate(gnb2, Xs_test_s[['Age', 'EstimatedSalary']], ys_test, 'GaussianNB (Age, Salary)')]).set_index('model').round(4)
display(res_ads)


cv_nb = GridSearchCV(GaussianNB(), {'var_smoothing': np.logspace(-12, 0, 13)}, scoring='accuracy',
                     cv=StratifiedKFold(10, shuffle=True, random_state=0))
cv_nb.fit(Xs_train_s[['Age', 'EstimatedSalary']], ys_train)
print('var_smoothing tốt nhất:', cv_nb.best_params_, '| Accuracy CV:', round(cv_nb.best_score_, 4))
acc10 = cross_val_score(GaussianNB(), Xs_train_s[['Age', 'EstimatedSalary']], ys_train, cv=StratifiedKFold(10, shuffle=True, random_state=0))
print(f'10-fold CV accuracy (mặc định): {acc10.mean():.4f} ± {acc10.std():.4f}')


plot_cm([('GaussianNB (Age, Salary)', gnb2)], Xs_test_s[['Age', 'EstimatedSalary']], ys_test, ['Not purchased', 'Purchased'],
        'Confusion matrix – Social Network Ads')
print(classification_report(ys_test, gnb2.predict(Xs_test_s[['Age', 'EstimatedSalary']]), target_names=['Not purchased', 'Purchased']))


fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
for ax, (Xp, yp, t) in zip(axes, [(Xs_train_s, ys_train, 'Tập train'), (Xs_test_s, ys_test, 'Tập test')]):
    x1, x2 = np.meshgrid(np.linspace(-2.5, 2.5, 400), np.linspace(-2.5, 2.8, 400))
    zz = gnb2.predict_proba(pd.DataFrame(np.c_[x1.ravel(), x2.ravel()], columns=['Age', 'EstimatedSalary']))[:, 1].reshape(x1.shape)
    ax.contourf(x1, x2, zz, levels=[0, .5, 1], colors=['#F4CCCC', '#CFE2F3'], alpha=.8)
    ax.contour(x1, x2, zz, levels=[.5], colors='k', linewidths=1)
    ax.scatter(Xp['Age'], Xp['EstimatedSalary'], c=np.where(yp == 1, '#1f77b4', '#d62728'), s=18, edgecolor='k', lw=.3)
    ax.set_xlabel('Age (chuẩn hóa)'); ax.set_ylabel('EstimatedSalary (chuẩn hóa)'); ax.set_title(f'Gaussian NB – {t}')
plt.tight_layout(); plt.show()
