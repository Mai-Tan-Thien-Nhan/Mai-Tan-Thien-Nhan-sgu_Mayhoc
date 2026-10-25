# Bài tập 2.2.4 – SVM trên dữ liệu tình trạng động vật
# Chạy trong Jupyter (copy từng khối) hoặc: python Bai_2_2_4_SVM_DongVat.py  (dữ liệu nằm trong thư mục data/)



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


animal = pd.read_csv('data/animal_data.csv')
print(animal.shape)
display(animal.head())
print(animal.isnull().sum())
print(animal['Dangerous'].value_counts(dropna=False))
print('Số loài:', animal['AnimalName'].nunique(), '| Số dòng trùng:', animal.duplicated().sum())


symptoms = pd.concat([animal[f'symptoms{i}'] for i in range(1, 6)]).str.strip().str.lower()
print('Số triệu chứng khác nhau:', symptoms.nunique())
display(symptoms.value_counts().head(10))


an = animal.dropna(subset=['Dangerous']).copy()
for c in ['AnimalName'] + [f'symptoms{i}' for i in range(1, 6)]:
    an[c] = an[c].str.strip().str.lower()
sym_cols = [f'symptoms{i}' for i in range(1, 6)]
multi_hot = pd.get_dummies(an[sym_cols].stack()).groupby(level=0).max().astype(int).add_prefix('sym_')
X_an = pd.concat([pd.get_dummies(an['AnimalName'], prefix='animal', dtype=int), multi_hot], axis=1)
y_an = (an['Dangerous'] == 'Yes').astype(int).values
print('Ma trận đặc trưng:', X_an.shape)
Xa_train, Xa_test, ya_train, ya_test = train_test_split(X_an.values, y_an, test_size=0.2, random_state=RS, stratify=y_an)
print('Train:', Xa_train.shape, np.bincount(ya_train), '| Test:', Xa_test.shape, np.bincount(ya_test))


from sklearn.metrics import balanced_accuracy_score
rows = []
for k in ['linear', 'poly', 'rbf', 'sigmoid']:
    for cw in [None, 'balanced']:
        m = SVC(kernel=k, class_weight=cw).fit(Xa_train, ya_train)
        p = m.predict(Xa_test)
        rows.append({'kernel': k, 'class_weight': cw or 'none', 'train_acc': m.score(Xa_train, ya_train),
                     'test_acc': accuracy_score(ya_test, p), 'balanced_acc': balanced_accuracy_score(ya_test, p),
                     'f1_macro': f1_score(ya_test, p, average='macro')})
display(pd.DataFrame(rows).round(4))


an_grid = [{'kernel': ['linear'], 'C': [0.01, 0.1, 1, 10]},
           {'kernel': ['rbf'], 'C': [0.1, 1, 10, 100], 'gamma': ['scale', 0.01, 0.1, 1]}]
cv_svm_a = GridSearchCV(SVC(class_weight='balanced', probability=True, random_state=RS), an_grid, scoring='f1_macro',
                        cv=StratifiedKFold(5, shuffle=True, random_state=RS), n_jobs=-1)
cv_svm_a.fit(Xa_train, ya_train)
print('Tham số tốt nhất:', cv_svm_a.best_params_, '| F1-macro CV:', round(cv_svm_a.best_score_, 4))
display(pd.DataFrame(cv_svm_a.cv_results_)[['params', 'mean_test_score', 'std_test_score']].sort_values('mean_test_score', ascending=False).head(6).round(4))


svm_a = cv_svm_a.best_estimator_
svm_a_plain = SVC(kernel='rbf').fit(Xa_train, ya_train)
res_a = pd.DataFrame([{**evaluate(svm_a_plain, Xa_test, ya_test, 'SVM RBF mặc định', average='macro'),
                       'balanced_acc': balanced_accuracy_score(ya_test, svm_a_plain.predict(Xa_test))},
                      {**evaluate(svm_a, Xa_test, ya_test, f'SVM tối ưu {cv_svm_a.best_params_}', average='macro'),
                       'balanced_acc': balanced_accuracy_score(ya_test, svm_a.predict(Xa_test))}]).set_index('model').round(4)
display(res_a)
plot_cm([('SVM RBF mặc định', svm_a_plain), ('SVM tối ưu (balanced)', svm_a)], Xa_test, ya_test, ['No', 'Yes'],
        'Confusion matrix – Animal condition (tập test)')
print(classification_report(ya_test, svm_a.predict(Xa_test), target_names=['No', 'Yes']))


skf = StratifiedKFold(5, shuffle=True, random_state=RS)
best_cfg = {k: v for k, v in cv_svm_a.best_params_.items()}
for name, m in [('SVM RBF mặc định', SVC(kernel='rbf')), ('SVM tối ưu + balanced', SVC(**best_cfg, class_weight='balanced'))]:
    s = cross_val_score(m, X_an.values, y_an, cv=skf, scoring='f1_macro')
    b = cross_val_score(m, X_an.values, y_an, cv=skf, scoring='balanced_accuracy')
    print(f'{name:25s}: F1-macro = {s.mean():.4f} ± {s.std():.4f} | balanced acc = {b.mean():.4f} ± {b.std():.4f}')
