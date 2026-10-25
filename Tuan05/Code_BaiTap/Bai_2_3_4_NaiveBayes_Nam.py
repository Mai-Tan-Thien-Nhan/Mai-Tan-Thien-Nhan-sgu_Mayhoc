# Bài tập 2.3.4 – Naive Bayes trên dữ liệu nấm (Mushroom)
# Chạy trong Jupyter (copy từng khối) hoặc: python Bai_2_3_4_NaiveBayes_Nam.py  (dữ liệu nằm trong thư mục data/)



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


mush = pd.read_csv('data/mushrooms.csv')
print(mush.shape)
display(mush.head())
print('Giá trị thiếu:', mush.isnull().sum().sum(), '| Dòng trùng:', mush.duplicated().sum())
print(mush['class'].value_counts())
print("Giá trị '?' trong stalk-root:", (mush['stalk-root'] == '?').sum())
display(mush.nunique().sort_values().to_frame('n_unique').T)


fig, ax = plt.subplots(1, 2, figsize=(15, 4.5))
sns.countplot(data=mush, x='odor', hue='class', ax=ax[0], palette={'e': '#55A868', 'p': '#C44E52'}); ax[0].set_title('Mùi (odor) theo lớp')
sns.countplot(data=mush, x='spore-print-color', hue='class', ax=ax[1], palette={'e': '#55A868', 'p': '#C44E52'}); ax[1].set_title('Màu bào tử theo lớp')
plt.tight_layout(); plt.show()


mu = mush.drop(columns=['veil-type'])
Xm = mu.drop(columns='class'); ym = (mu['class'] == 'p').astype(int).values  # 1 = độc
Xm_train, Xm_test, ym_train, ym_test = train_test_split(Xm, ym, test_size=0.2, random_state=RS, stratify=ym)
oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1).fit(Xm_train)
ohe = OneHotEncoder(handle_unknown='ignore').fit(Xm_train)
Xm_tr_o, Xm_te_o = oe.transform(Xm_train).astype(int), oe.transform(Xm_test).astype(int)
Xm_tr_h, Xm_te_h = ohe.transform(Xm_train), ohe.transform(Xm_test)
print('Ordinal:', Xm_tr_o.shape, '| One-hot:', Xm_tr_h.shape)


nb_models = {
    'CategoricalNB (ordinal)': (CategoricalNB(min_categories=[len(c) for c in oe.categories_]), Xm_tr_o, Xm_te_o),
    'MultinomialNB (one-hot)': (MultinomialNB(), Xm_tr_h, Xm_te_h),
    'BernoulliNB (one-hot)': (BernoulliNB(), Xm_tr_h, Xm_te_h),
    'GaussianNB (ordinal, mốc)': (GaussianNB(), Xm_tr_o, Xm_te_o),
}
rows = []
for name, (m, Xtr, Xte) in nb_models.items():
    m.fit(Xtr, ym_train)
    rows.append({**evaluate(m, Xte, ym_test, name), 'train_acc': m.score(Xtr, ym_train)})
res_m = pd.DataFrame(rows).set_index('model').round(4)
display(res_m)


cv_cnb = GridSearchCV(CategoricalNB(min_categories=[len(c) for c in oe.categories_]),
                      {'alpha': [1e-3, 1e-2, 0.1, 0.5, 1, 2, 5]}, scoring='accuracy', cv=StratifiedKFold(5, shuffle=True, random_state=RS))
cv_cnb.fit(Xm_tr_o, ym_train)
print('alpha tốt nhất:', cv_cnb.best_params_, '| Accuracy CV:', round(cv_cnb.best_score_, 4))
display(pd.DataFrame(cv_cnb.cv_results_)[['param_alpha', 'mean_test_score']].round(4).T)


best_cnb = cv_cnb.best_estimator_
display(pd.DataFrame([evaluate(best_cnb, Xm_te_o, ym_test, f"CategoricalNB alpha={cv_cnb.best_params_['alpha']}")]).set_index('model').round(4))
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, (name, m, Xte) in zip(axes, [('CategoricalNB tối ưu', best_cnb, Xm_te_o),
                                     ('MultinomialNB', nb_models['MultinomialNB (one-hot)'][0], Xm_te_h),
                                     ('GaussianNB (mốc)', nb_models['GaussianNB (ordinal, mốc)'][0], Xm_te_o)]):
    ConfusionMatrixDisplay.from_estimator(m, Xte, ym_test, display_labels=['edible', 'poisonous'], cmap='Blues', ax=ax, colorbar=False)
    ax.set_title(name); ax.grid(False)
plt.suptitle('Confusion matrix – Mushroom (tập test)'); plt.tight_layout(); plt.show()
print(classification_report(ym_test, best_cnb.predict(Xm_te_o), target_names=['edible', 'poisonous'], digits=4))


cols_m = list(Xm.columns)
score = {}
for j, col in enumerate(cols_m):
    lp = best_cnb.feature_log_prob_[j]  # shape (2 lớp, số giá trị)
    score[col] = np.abs(lp[1] - lp[0]).max()
imp = pd.Series(score).sort_values()
imp.plot.barh(figsize=(8, 7), color='#8172B2')
plt.title('Mức độ phân biệt của thuộc tính (CategoricalNB)'); plt.xlabel('max |log P(x|p) − log P(x|e)|')
plt.show()
print(pd.crosstab(mush['odor'], mush['class']))
