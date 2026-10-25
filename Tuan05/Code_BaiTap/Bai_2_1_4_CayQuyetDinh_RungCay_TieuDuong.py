# Bài tập 2.1.4 – Cây quyết định & rừng cây trên dữ liệu tiểu đường
# Chạy trong Jupyter (copy từng khối) hoặc: python Bai_2_1_4_CayQuyetDinh_RungCay_TieuDuong.py  (dữ liệu nằm trong thư mục data/)



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


display(diab.describe().T.round(2))


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


dt2_d = tree.DecisionTreeClassifier(max_depth=2, random_state=RS).fit(Xd_train, yd_train)
plt.figure(figsize=(16, 7))
tree.plot_tree(dt2_d, feature_names=features_d, class_names=['No diabetes', 'Diabetes'], filled=True, rounded=True,
               proportion=True, fontsize=12)
plt.title('Cây quyết định max_depth = 2 – Diabetes', fontsize=15)
plt.show()
print(tree.export_text(dt2_d, feature_names=features_d))


cv_dt_d = GridSearchCV(tree.DecisionTreeClassifier(random_state=RS), {'max_depth': [1, 2, 4, 6, 8, 10, 12, 16, 20]},
                       scoring='roc_auc', cv=4, return_train_score=True, n_jobs=-1)
cv_dt_d.fit(Xd_train, yd_train)
print('Tham số tốt nhất:', cv_dt_d.best_params_, '| ROC AUC CV:', round(cv_dt_d.best_score_, 4))
display(plot_cv_curve(cv_dt_d, 'max_depth', 'max_depth', 'Decision Tree – ROC AUC theo max_depth (Diabetes)', 4))


cv_rf_d = GridSearchCV(RandomForestClassifier(max_depth=3, max_features='sqrt', random_state=4),
                       {'n_estimators': list(range(10, 110, 10))}, scoring='roc_auc', cv=4, return_train_score=True, n_jobs=-1)
cv_rf_d.fit(Xd_train, yd_train)
r = pd.DataFrame(cv_rf_d.cv_results_)
fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
axs[0].plot(r['param_n_estimators'], r['mean_fit_time'], '-o'); axs[0].set_xlabel('Number of trees'); axs[0].set_ylabel('Mean fit time (s)')
axs[1].errorbar(r['param_n_estimators'], r['mean_test_score'], yerr=r['std_test_score'] / np.sqrt(4), marker='o')
axs[1].set_xlabel('Number of trees'); axs[1].set_ylabel('Mean testing ROC AUC $\\pm$ 1 SE')
plt.suptitle('Random Forest (max_depth=3) – Diabetes'); plt.tight_layout(); plt.show()
print('Số cây tốt nhất:', cv_rf_d.best_params_, '| ROC AUC CV:', round(cv_rf_d.best_score_, 4))


cv_rf2_d = GridSearchCV(RandomForestClassifier(n_estimators=100, random_state=4, n_jobs=-1),
                        {'max_depth': [6, 10, 14, None], 'min_samples_leaf': [1, 5, 20]},
                        scoring='roc_auc', cv=4, n_jobs=-1)
cv_rf2_d.fit(Xd_train, yd_train)
print('Tham số tốt nhất:', cv_rf2_d.best_params_, '| ROC AUC CV:', round(cv_rf2_d.best_score_, 4))
display(pd.DataFrame(cv_rf2_d.cv_results_)[['param_max_depth', 'param_min_samples_leaf', 'mean_test_score']].sort_values('mean_test_score', ascending=False).head(6).round(4))


best_rf_d = cv_rf2_d.best_estimator_
fi_d = pd.DataFrame({'Importance': best_rf_d.feature_importances_}, index=features_d).sort_values('Importance')
fi_d.plot.barh(figsize=(8, 5), legend=False, color='#55A868')
plt.title('Feature importance – Random Forest (Diabetes)'); plt.xlabel('Mean decrease in impurity')
plt.show()
display(fi_d.sort_values('Importance', ascending=False).round(4))


rf_bal_d = RandomForestClassifier(**{**best_rf_d.get_params(), 'class_weight': 'balanced'}).fit(Xd_train, yd_train)
res_d = pd.DataFrame([evaluate(dt2_d, Xd_test, yd_test, 'DT (depth=2)'),
                      evaluate(cv_dt_d.best_estimator_, Xd_test, yd_test, f"DT tốt nhất (depth={cv_dt_d.best_params_['max_depth']})"),
                      evaluate(cv_rf_d.best_estimator_, Xd_test, yd_test, f"RF depth=3 ({cv_rf_d.best_params_['n_estimators']} cây)"),
                      evaluate(best_rf_d, Xd_test, yd_test, 'RF tối ưu'),
                      evaluate(rf_bal_d, Xd_test, yd_test, 'RF tối ưu + class_weight=balanced')]).set_index('model').round(4)
display(res_d)
plot_cm([('DT tốt nhất', cv_dt_d.best_estimator_), ('RF tối ưu', best_rf_d), ('RF + balanced', rf_bal_d)],
        Xd_test, yd_test, ['No', 'Diabetes'], 'Confusion matrix – Diabetes (tập test)')
print(classification_report(yd_test, best_rf_d.predict(Xd_test), target_names=['No diabetes', 'Diabetes']))
