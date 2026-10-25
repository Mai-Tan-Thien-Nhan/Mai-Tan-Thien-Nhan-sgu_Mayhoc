# Bài tập 2.2.3 – SVM trên dữ liệu tiểu đường
# Chạy trong Jupyter (copy từng khối) hoặc: python Bai_2_2_3_SVM_TieuDuong.py  (dữ liệu nằm trong thư mục data/)



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


diab = pd.read_csv('data/diabetes_prediction_dataset.csv')
print(diab.shape)
display(diab.head())
print(diab.isnull().sum().sum(), 'giá trị thiếu |', diab.duplicated().sum(), 'dòng trùng')
print(diab['diabetes'].value_counts(normalize=True).round(4))


d = diab.drop_duplicates()
d = d[d['gender'] != 'Other'].copy()
d['gender'] = (d['gender'] == 'Male').astype(int)
d['smoking_history'] = d['smoking_history'].replace({'ever': 'former', 'not current': 'former'})
d = pd.get_dummies(d, columns=['smoking_history'], prefix='smoke', dtype=int)
features_d = [c for c in d.columns if c != 'diabetes']
print(d.shape, features_d)
print('Tỷ lệ mắc bệnh sau làm sạch:', round(d['diabetes'].mean(), 4))
Xd_train, Xd_test, yd_train, yd_test = train_test_split(d[features_d].values, d['diabetes'].values,
                                                        test_size=0.2, random_state=RS, stratify=d['diabetes'])
print(Xd_train.shape, Xd_test.shape)


X_sub, _, y_sub, _ = train_test_split(Xd_train, yd_train, train_size=8000, stratify=yd_train, random_state=RS)
X_fin, _, y_fin, _ = train_test_split(Xd_train, yd_train, train_size=20000, stratify=yd_train, random_state=RS)
print('Mẫu tìm tham số:', X_sub.shape, '| Mẫu huấn luyện cuối:', X_fin.shape, '| Test:', Xd_test.shape)


kernels = ['linear', 'poly', 'rbf', 'sigmoid']
rows = []
for k in kernels:
    t0 = time.time()
    pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel=k))])
    pipe.fit(X_sub, y_sub)
    rows.append({'kernel': k, 'train_acc': pipe.score(X_sub, y_sub), 'test_acc': pipe.score(Xd_test, yd_test),
                 'test_f1': f1_score(yd_test, pipe.predict(Xd_test)), 'n_support_vectors': pipe['svc'].n_support_.sum(),
                 'fit_time_s': time.time() - t0})
kern_df = pd.DataFrame(rows).set_index('kernel').round(4)
display(kern_df)


svm_grid = [{'svc__kernel': ['rbf'], 'svc__C': [0.1, 1, 10, 100], 'svc__gamma': ['scale', 0.01, 0.1, 1]},
            {'svc__kernel': ['linear'], 'svc__C': [0.1, 1, 10]}]
pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])
cv_svm_d = GridSearchCV(pipe, svm_grid, scoring='f1', cv=StratifiedKFold(3, shuffle=True, random_state=RS), n_jobs=-1)
cv_svm_d.fit(X_sub, y_sub)
print('Tham số tốt nhất:', cv_svm_d.best_params_, '| F1 CV:', round(cv_svm_d.best_score_, 4))
g = pd.DataFrame(cv_svm_d.cv_results_)
g_rbf = g[g['param_svc__kernel'] == 'rbf'].pivot_table(index='param_svc__C', columns='param_svc__gamma', values='mean_test_score', aggfunc='first')
g_rbf.columns = g_rbf.columns.astype(str)
plt.figure(figsize=(7, 4.5))
sns.heatmap(g_rbf, annot=True, fmt='.3f', cmap='viridis')
plt.title('F1 (CV) của SVM RBF theo C và gamma'); plt.xlabel('gamma'); plt.ylabel('C')
plt.show()


bp = {k.replace('svc__', ''): v for k, v in cv_svm_d.best_params_.items()}
t0 = time.time()
svm_d = Pipeline([('scaler', StandardScaler()), ('svc', SVC(**bp, probability=True, random_state=RS))]).fit(X_fin, y_fin)
print(f'Thời gian huấn luyện trên 20.000 mẫu: {time.time() - t0:.1f}s')
svm_d_bal = Pipeline([('scaler', StandardScaler()), ('svc', SVC(**bp, class_weight='balanced', probability=True, random_state=RS))]).fit(X_fin, y_fin)
res_svm_d = pd.DataFrame([evaluate(svm_d, Xd_test, yd_test, f'SVM {bp}'),
                          evaluate(svm_d_bal, Xd_test, yd_test, 'SVM + class_weight=balanced')]).set_index('model').round(4)
display(res_svm_d)
plot_cm([('SVM', svm_d), ('SVM + balanced', svm_d_bal)], Xd_test, yd_test, ['No', 'Diabetes'], 'Confusion matrix – SVM (Diabetes)')
print(classification_report(yd_test, svm_d.predict(Xd_test), target_names=['No diabetes', 'Diabetes']))


dt_best = tree.DecisionTreeClassifier(max_depth=10, random_state=RS).fit(Xd_train, yd_train)
rf_best = RandomForestClassifier(n_estimators=100, max_depth=14, min_samples_leaf=5, random_state=4, n_jobs=-1).fit(Xd_train, yd_train)
fig, ax = plt.subplots(figsize=(7, 6))
for name, m in [('Decision Tree', dt_best), ('Random Forest', rf_best), ('SVM', svm_d)]:
    RocCurveDisplay.from_estimator(m, Xd_test, yd_test, name=name, ax=ax)
ax.plot([0, 1], [0, 1], 'k--', lw=1); ax.set_title('ROC curve – Diabetes (tập test)')
plt.show()
