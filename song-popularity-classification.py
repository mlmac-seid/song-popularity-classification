# imports
import numpy as np
import matplotlib.pyplot as plt
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
import seaborn as sns
import pandas as pd
import sklearn as sk

# styling additions
from IPython.display import HTML
style = '''
    <style>
        div.info{
            padding: 15px;
            border: 1px solid transparent;
            border-left: 5px solid #dfb5b4;
            border-color: transparent;
            margin-bottom: 10px;
            border-radius: 4px;
            background-color: #fcf8e3;
            border-color: #faebcc;
        }
        hr{
            border: 1px solid;
            border-radius: 5px;
        }
    </style>'''
HTML(style)

df = pd.read_csv('song_data_orig.csv')
df.columns

df['song_popularity'].hist();

df['is_pop'] = (df['song_popularity'] >= 80).values.astype('int')

df.query('is_pop == 1')

df = df.drop('song_popularity',axis=1)
df.columns

sns.pairplot(df, vars=df.columns[1:15], hue='is_pop');

song_df = df[['song_duration_ms', 'instrumentalness', 'is_pop']]
song_df

X = np.array(song_df[['song_duration_ms', 'instrumentalness']])
X
y = np.array(song_df['is_pop'])
y

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=123,
                                                    shuffle=True,
                                                    stratify=y)

sns.scatterplot(x="song_duration_ms", y="instrumentalness", data=song_df, hue="is_pop")

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(penalty='none')
log_reg.fit(X_train,y_train)

log_reg.coef_, log_reg.intercept_

def sig_curve(data):
  e_array = math.e*(np.ones_like(len(data)))
  return 1/(1+np.power(e_array, -(log_reg.intercept_ + (log_reg.coef_[0][0])*data[0:len(data), 0]+(log_reg.coef_[0][1]*data[0:len(data), 1]))))

import math
x_pl = np.linspace(30,60,100)
y_pl = np.linspace(5,22,100)
X_pl,Y_pl = np.meshgrid(x_pl,y_pl)
XY_pl = np.vstack((X_pl.ravel(),Y_pl.ravel())).T
XY_pl.shape
Z = sig_curve(XY_pl).reshape(100,100)

pop_X = song_df.loc[song_df["is_pop"]==1]
pop_X =  pop_X[['song_duration_ms', 'instrumentalness']].values
not_pop_X = song_df.loc[song_df["is_pop"]==0]
not_pop_X =  not_pop_X[['song_duration_ms', 'instrumentalness']].values

errors = log_reg.predict(X_test) != y_test
sum_errors = 0
for i in range(len(errors)):
  if(errors[i] == True):
    sum_errors +=1
print("Sum Errors:",sum_errors)
percent = sum_errors / len(X_test)
percent *=100
percent = 100 - percent
print("Accuracy:", str(percent) +"%")

sns.scatterplot(x="song_duration_ms", y="instrumentalness", data=song_df, hue="is_pop")

from sklearn.neighbors import KNeighborsClassifier

n_neighbors = 10

knn_model = KNeighborsClassifier(n_neighbors)
knn_model.fit(X_train, y_train)

print(f'Accuracy: {knn_model.score(X, y)*100:.2f}%')

feature_names = ['song_duration_ms', 'instrumentalness']

# decision boundary plotting function
from matplotlib.colors import ListedColormap
def plot_decision_boundary(model, X, y, scale=1):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    h=0.5
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h*scale))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    cmap = sns.palettes.color_palette('muted',as_cmap=True)
    cmap_light = ListedColormap(cmap[:2])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=cmap_light,alpha=0.5);
    # Plot also the training points
    ax = sns.scatterplot(
        x=X[:, 0],
        y=X[:, 1],
        hue=y,
        alpha=1.0,
        edgecolor="black",
        palette='muted'
    )
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(f'KNN: K={n_neighbors} and Accuracy: {model.score(X, y)*100:.2f}%');
    plt.xlabel(feature_names[0]);
    plt.ylabel(feature_names[1]);
    handles,labels = ax.get_legend_handles_labels()
    ax.legend(handles, ['pop','not_pop'])

plot_decision_boundary(knn_model,X_train,y_train)

errors2 = knn_model.predict(X_test) != y_test
sum_errors2 = 0
for i in range(len(errors2)):
  if(errors2[i] == True):
    sum_errors2 +=1
print("Sum Errors:",sum_errors2)
percent2 = sum_errors2 / len(X_test)
percent2 *=100
percent2 = 100 - percent2
print("Accuracy:", str(percent2) +"%")

knn_model = KNeighborsClassifier(10)
knn_model.fit(X_test, y_test)
print(f'Accuracy: {knn_model.score(X_test, y_test)*100:.2f}%')

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, RocCurveDisplay, ConfusionMatrixDisplay, auc
from sklearn.linear_model import LogisticRegression
from sklearn import svm
classifier2 = LogisticRegression()

classifier2.fit(X_test, y_test)

cm2 = confusion_matrix(y_test, classifier2.predict(X_test))
cm2

# display it on the subplot figure
disp = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=['Popular', 'Not Popular'])
disp.plot();

RocCurveDisplay.from_estimator(classifier2,X,y,name='ROC',lw=1);

cv2 = StratifiedKFold(n_splits=2)
cv2.split(X,y)

for fold_num, (train_idx, test_idx) in enumerate(cv2.split(X,y)):
    print('=====================================================')
    print(f"CV fold #{fold_num+1}")
    print('=====================================================')
    print(f'train idices: {train_idx}')
    print()
    print(f'test idices: {test_idx}')

################################################
#                Setup figures
################################################
# NOTE: my default code has this working for 6 folds giving a 2x3 figure
fig_cm, ax_cm = plt.subplots(2,3,figsize=(10,8))

# setup ROC figure
fig, ax = plt.subplots(figsize=(10,8))
ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    title="Receiver Operating Characteristic",
)

# setup mean curve variables
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

# plot "no-skill"/"guessing" classifier
ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

################################################
#               Cross-Validation
################################################
cv2 = StratifiedKFold(n_splits=6)

# actually perform the CV and loop through each fold
for fold_num, (train_idx, test_idx) in enumerate(cv2.split(X,y)):
    print('=====================================================')
    print(f"CV fold #{fold_num+1}")
    print('=====================================================')

    # fit classifier **on this folds data**
    classifier2.fit(X[train_idx],y[train_idx])

    # predict on this folds "testing" set
    y_pred = classifier2.predict(X[test_idx])

    # get this fold's confusion matrix
    cm2 = confusion_matrix(y[test_idx], classifier2.predict(X[test_idx]))

    # display it on the subplot figure
    disp = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=['Popular', 'Not Popular'])
    disp.plot(ax=ax_cm.flatten()[fold_num])
    ax_cm.flatten()[fold_num].set(title=f'CM for fold {fold_num+1}')

    # print this folds classification report
    print('Classification report:')
    print(classification_report(y[test_idx], y_pred, target_names=['Popular', 'Not Popular']))

    # build and display ROC curve for this fold
    viz = RocCurveDisplay.from_estimator(
        classifier2,
        X[test_idx],
        y[test_idx],
        name=f"ROC fold {fold_num+1}",
        alpha=0.4,
        lw=1,
        ax=ax,
    )

    # store information for the mean curve
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

# now actually plot MEAN ROC curve
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=f"Mean ROC (AUC = {mean_auc:0.2f})",
    lw=2,
    alpha=0.8,
)

ax.legend(loc="lower right");

from sklearn.neighbors import KNeighborsClassifier
classifier3 = KNeighborsClassifier(n_neighbors=1)

classifier3.fit(X,y)

cm3 = confusion_matrix(y, classifier3.predict(X))
cm3

# display it on the subplot figure
disp = ConfusionMatrixDisplay(confusion_matrix=cm3, display_labels=['Popular', 'Not Popular'])
disp.plot();

RocCurveDisplay.from_estimator(classifier3,X,y,name='ROC',lw=1);

cv3 = StratifiedKFold(n_splits=2)
cv3.split(X,y)

for fold_num, (train_idx, test_idx) in enumerate(cv3.split(X,y)):
    print('=====================================================')
    print(f"CV fold #{fold_num+1}")
    print('=====================================================')
    print(f'train idices: {train_idx}')
    print()
    print(f'test idices: {test_idx}')

################################################
#                Setup figures
################################################
# NOTE: my default code has this working for 6 folds giving a 2x3 figure
fig_cm, ax_cm = plt.subplots(2,3,figsize=(10,8))

# setup ROC figure
fig, ax = plt.subplots(figsize=(10,8))
ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    title="Receiver Operating Characteristic",
)

# setup mean curve variables
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

# plot "no-skill"/"guessing" classifier
ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

################################################
#               Cross-Validation
################################################
cv3 = StratifiedKFold(n_splits=6)

# actually perform the CV and loop through each fold
for fold_num, (train_idx, test_idx) in enumerate(cv3.split(X,y)):
    print('=====================================================')
    print(f"CV fold #{fold_num+1}")
    print('=====================================================')

    # fit classifier **on this folds data**
    classifier3.fit(X[train_idx],y[train_idx])

    # predict on this folds "testing" set
    y_pred = classifier3.predict(X[test_idx])

    # get this fold's confusion matrix
    cm3 = confusion_matrix(y[test_idx], classifier3.predict(X[test_idx]))

    # display it on the subplot figure
    disp = ConfusionMatrixDisplay(confusion_matrix=cm3, display_labels=['Popular', 'Not Popular'])
    disp.plot(ax=ax_cm.flatten()[fold_num])
    ax_cm.flatten()[fold_num].set(title=f'CM for fold {fold_num+1}')

    # print this folds classification report
    print('Classification report:')
    print(classification_report(y[test_idx], y_pred, target_names=['Popular', 'Not Popular']))

    # build and display ROC curve for this fold
    viz = RocCurveDisplay.from_estimator(
        classifier3,
        X[test_idx],
        y[test_idx],
        name=f"ROC fold {fold_num+1}",
        alpha=0.4,
        lw=1,
        ax=ax,
    )

    # store information for the mean curve
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

# now actually plot MEAN ROC curve
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=f"Mean ROC (AUC = {mean_auc:0.2f})",
    lw=2,
    alpha=0.8,
)

ax.legend(loc="lower right");