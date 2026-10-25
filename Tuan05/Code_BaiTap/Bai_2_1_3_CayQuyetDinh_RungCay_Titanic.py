# Bài tập 2.1.3 – Cây quyết định & rừng cây trên dữ liệu Titanic
# Chạy trong Jupyter (copy từng khối) hoặc: python Bai_2_1_3_CayQuyetDinh_RungCay_Titanic.py  (dữ liệu nằm trong thư mục data/)



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


titanic = pd.read_csv('data/titanic.csv')
print(titanic.shape)
display(titanic.head())


print(titanic.isnull().sum())
print('\nTỷ lệ sống sót:', round(titanic['Survived'].mean(), 3))
display(titanic.groupby(['Sex', 'Pclass'])['Survived'].mean().unstack().round(3))


df = titanic.copy()
df['Title'] = df['Name'].str.extract(r',\s*([^\.]+)\.')[0].str.strip()
df['Title'] = df['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
df.loc[~df['Title'].isin(['Mr', 'Mrs', 'Miss', 'Master']), 'Title'] = 'Rare'
df['Age'] = df['Age'].fillna(df.groupby('Title')['Age'].transform('median'))
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['HasCabin'] = df['Cabin'].notnull().astype(int)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
df = pd.get_dummies(df, columns=['Embarked', 'Title'], drop_first=False, dtype=int)
features_t = [c for c in df.columns if c != 'Survived']
print('Số đặc trưng:', len(features_t)); print(features_t)
print('Còn giá trị thiếu:', df.isnull().sum().sum())


X_train, X_test, y_train, y_test = train_test_split(df[features_t].values, df['Survived'].values,
                                                    test_size=0.2, random_state=RS, stratify=df['Survived'])
print(X_train.shape, X_test.shape)


dt2 = tree.DecisionTreeClassifier(max_depth=2, random_state=RS)
dt2.fit(X_train, y_train)
plt.figure(figsize=(16, 7))
tree.plot_tree(dt2, feature_names=features_t, class_names=['Not survived', 'Survived'],
               filled=True, rounded=True, proportion=True, fontsize=12)
plt.title('Cây quyết định max_depth = 2 – Titanic', fontsize=15)
plt.show()
print(tree.export_text(dt2, feature_names=features_t))
print('Accuracy train/test:', round(dt2.score(X_train, y_train), 4), '/', round(dt2.score(X_test, y_test), 4))


params = {'max_depth': [1, 2, 3, 4, 5, 6, 8, 10, 12]}
cv_dt = GridSearchCV(tree.DecisionTreeClassifier(random_state=RS), param_grid=params, scoring='roc_auc',
                     cv=4, refit=True, return_train_score=True, n_jobs=-1)
cv_dt.fit(X_train, y_train)
print('Tham số tốt nhất:', cv_dt.best_params_, '| ROC AUC CV:', round(cv_dt.best_score_, 4))
display(plot_cv_curve(cv_dt, 'max_depth', 'max_depth', 'Decision Tree – ROC AUC theo max_depth (Titanic)', 4))


rf = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=3, max_features='sqrt',
                            bootstrap=True, random_state=4)
rf_params = {'n_estimators': list(range(10, 110, 10))}
cv_rf = GridSearchCV(rf, param_grid=rf_params, scoring='roc_auc', cv=4, refit=True, return_train_score=True, n_jobs=-1)
cv_rf.fit(X_train, y_train)
r = pd.DataFrame(cv_rf.cv_results_)
fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
axs[0].plot(r['param_n_estimators'], r['mean_fit_time'], '-o'); axs[0].set_xlabel('Number of trees'); axs[0].set_ylabel('Mean fit time (s)')
axs[1].errorbar(r['param_n_estimators'], r['mean_test_score'], yerr=r['std_test_score'] / np.sqrt(4), marker='o')
axs[1].set_xlabel('Number of trees'); axs[1].set_ylabel('Mean testing ROC AUC $\\pm$ 1 SE')
plt.suptitle('Random Forest (max_depth=3) – Titanic'); plt.tight_layout(); plt.show()
print('Số cây tốt nhất:', cv_rf.best_params_, '| ROC AUC CV:', round(cv_rf.best_score_, 4))


rf_grid = {'n_estimators': [100, 200], 'max_depth': [3, 5, 7, 9, None],
           'min_samples_leaf': [1, 3, 5], 'max_features': ['sqrt', 0.5]}
cv_rf2 = GridSearchCV(RandomForestClassifier(random_state=4, n_jobs=-1), rf_grid, scoring='roc_auc', cv=4, n_jobs=-1)
cv_rf2.fit(X_train, y_train)
print('Tham số tốt nhất:', cv_rf2.best_params_, '| ROC AUC CV:', round(cv_rf2.best_score_, 4))


best_rf_t = cv_rf2.best_estimator_
fi = pd.DataFrame({'Importance': best_rf_t.feature_importances_}, index=features_t).sort_values('Importance')
fi.plot.barh(figsize=(8, 6), legend=False, color='#4C72B0')
plt.title('Feature importance – Random Forest tốt nhất (Titanic)'); plt.xlabel('Mean decrease in impurity')
plt.show()
display(fi.sort_values('Importance', ascending=False).head(8).round(4))


best_dt_t = cv_dt.best_estimator_
res_t = pd.DataFrame([evaluate(dt2, X_test, y_test, 'DT (depth=2)'),
                      evaluate(best_dt_t, X_test, y_test, f"DT tốt nhất (depth={cv_dt.best_params_['max_depth']})"),
                      evaluate(cv_rf.best_estimator_, X_test, y_test, f"RF depth=3 ({cv_rf.best_params_['n_estimators']} cây)"),
                      evaluate(best_rf_t, X_test, y_test, 'RF tối ưu nhiều tham số')]).set_index('model').round(4)
display(res_t)
plot_cm([('Decision Tree tốt nhất', best_dt_t), ('Random Forest tối ưu', best_rf_t)], X_test, y_test,
        ['Not survived', 'Survived'], 'Confusion matrix – Titanic (tập test)')
print(classification_report(y_test, best_rf_t.predict(X_test), target_names=['Not survived', 'Survived']))
