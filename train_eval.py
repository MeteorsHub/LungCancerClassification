import argparse
import os
import pickle
import glob
import numpy as np
import seaborn as sns
from sklearn import svm, metrics, naive_bayes
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.manifold import TSNE
from sklearn.model_selection import GridSearchCV
from scipy.interpolate import Rbf

import matplotlib.pyplot as plt

from data_loader import MarkerExpressionDataset
from utils import load_ymal, save_yaml, maybe_create_path, double_print

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, required=False, default='default',
                    help='config name. default: \'default\'')
parser.add_argument('-s', '--sub_setting', type=str, default='default',
                    help='sub setting name for current method config')
parser.add_argument('-r', '--retrain', action='store_true',
                    help='if set, override the existing saved model')
parser.add_argument('-o', '--overwrite_config', action='store_true',
                    help='if set, ignore existing config file in exp path and use that in config folder')
args = parser.parse_args()


def evaluate(y_true, y_pred, labels=(1,), average=None):
    precision = 100*metrics.precision_score(y_true, y_pred, labels=labels, average=average)
    recall = 100*metrics.recall_score(y_true, y_pred, labels=labels, average=average)

    jaccard = 100*metrics.jaccard_score(y_true, y_pred, labels=labels, average=average)
    conf_mat = metrics.confusion_matrix(y_true, y_pred, labels=labels)

    tps = np.diagonal(conf_mat)
    ps = np.sum(conf_mat, 1)
    sensitivity = 100*tps / ps
    specificity = np.zeros_like(sensitivity)
    for i in range(len(specificity)):
        all_negative_tps = 0
        all_negative_ps = 0
        for j in range(len(specificity)):
            if i != j:
                all_negative_tps += tps[j]
                all_negative_ps += ps[j]
        specificity[i] = all_negative_tps / all_negative_ps
    specificity *= 100
    f1 = 100*metrics.f1_score(y_true, y_pred, average='macro')
    return precision, recall, sensitivity, specificity, jaccard, conf_mat, f1


def train_eval(config, exp_path):
    dataset = MarkerExpressionDataset(config)
    if config['model'] == 'svm':
        model_class = svm.SVC
    elif config['model'] == 'rdf':
        model_class = RandomForestClassifier
    elif config['model'] == 'adaboost':
        model_class = AdaBoostClassifier
    elif config['model'] == 'gaussion_bayes':
        model_class = naive_bayes.GaussianNB
    elif config['model'] == 'mlp':
        model_class = MLPClassifier
    else:
        raise AttributeError('unrecognized model %s' % config['model'])

    fig, ax = plt.subplots(6, len(dataset.markers), squeeze=False, figsize=(6*len(dataset.markers), 30))
    # fig.suptitle(config['model'])
    metrics_file = open(os.path.join(exp_path, 'metrics.txt'), 'w')
    metrics_fig_filename = os.path.join(exp_path, 'conf_mat.png')
    best_params = dict()
    all_marker_train_f1s = []
    all_marker_test_f1s = []
    for i, marker in enumerate(dataset.markers):
        print('parameter search for marker %s...' % marker)
        all_x, all_y, cv_index = dataset.get_all_data(marker)
        best_model = GridSearchCV(model_class(),
                                  param_grid=config['model_kwargs'],
                                  n_jobs=4,
                                  cv=cv_index,
                                  scoring='recall_macro')
        best_model.fit(all_x, all_y)
        best_params[marker] = best_model.best_params_
        print('search done')

        train_xs = []
        train_ys = []
        pred_train_ys = []
        test_xs = []
        test_ys = []
        pred_test_ys = []
        for fold_i, (train_x, train_y, test_x, test_y) in enumerate(dataset.get_split_data(marker)):
            model = model_class()
            model.set_params(**best_params[marker])
            model.fit(train_x, train_y)
            train_xs += train_x
            train_ys += train_y
            test_xs += test_x
            test_ys += test_y
            pred_train_y = model.predict(train_x).tolist()
            pred_train_ys += pred_train_y
            pred_test_y = model.predict(test_x).tolist()
            pred_test_ys += pred_test_y
            model_filename = os.path.join(exp_path, 'model', '%s_%s_fold_%d.pkl'
                                          % (config['model'], marker, fold_i))
            maybe_create_path(os.path.dirname(model_filename))
            with open(model_filename, 'wb') as f:
                pickle.dump(model, f)
        train_precision, train_recall, train_sensitivity, train_specificity, train_jaccard, train_conf_mat, train_f1 = \
            evaluate(train_ys, pred_train_ys, dataset.classes, None)
        test_precision, test_recall, test_sensitivity, test_specificity, test_jaccard, test_conf_mat, test_f1 = \
            evaluate(test_ys, pred_test_ys, dataset.classes, None)
        all_marker_train_f1s.append(train_f1)
        all_marker_test_f1s.append(test_f1)

        double_print('marker: %s' % marker, metrics_file)
        double_print('metrics on training set:', metrics_file)
        for j, class_j in enumerate(dataset.classes):
            double_print('class: %s: precision: %1.1f. recall: %1.1f. sensitivity: %1.1f. specificity: %1.1f. acc: %1.1f.'
                         % (class_j, train_precision[j], train_recall[j], train_sensitivity[j], train_specificity[j], train_jaccard[j]),
                         metrics_file)
        double_print('avg f1: %1.1f' % train_f1)

        current_ax = ax[0, i]
        if len(train_xs[0]) == 2:
            embedded_train_xs = train_xs
        else:
            embedded_train_xs = TSNE(n_components=2, init='pca', random_state=1, n_jobs=4).fit_transform(train_xs)
        x = [item[0] for item in embedded_train_xs]
        y = [item[1] for item in embedded_train_xs]
        l_x, u_x = np.percentile(x, (25, 75), interpolation='midpoint')
        lower_bound_x = max(l_x - 3.0*(u_x - l_x), min(x))
        upper_bound_x = min(u_x + 3.0*(u_x - l_x), max(x))
        l_y, u_y = np.percentile(y, (25, 75), interpolation='midpoint')
        lower_bound_y = max(l_y - 3.0 * (u_y - l_y), min(y))
        upper_bound_y = min(u_y + 3.0 * (u_y - l_y), max(y))
        if len(train_xs[0]) == 2:
            xx, yy = np.meshgrid(np.arange(lower_bound_x, upper_bound_x, (upper_bound_x - lower_bound_x) / 1000),
                                 np.arange(lower_bound_y, upper_bound_y, (upper_bound_y - lower_bound_y) / 1000))
            Z = best_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
            current_ax.contourf(xx, yy, Z, alpha=0.2)
        sns.scatterplot(x=x, y=y,
                        style=[str(item) for item in train_ys],
                        style_order=[str(item) for item in dataset.classes],
                        hue=[str(item) for item in train_ys],
                        hue_order=[str(item) for item in dataset.classes],
                        ax=current_ax)
        current_ax.set_title('%s trained on whole set' % marker)
        current_ax.legend(loc="best")
        current_ax.set_xlim(lower_bound_x, upper_bound_x)
        current_ax.set_ylim(lower_bound_y, upper_bound_y)

        current_ax = ax[1, i]
        metrics.ConfusionMatrixDisplay(train_conf_mat, display_labels=dataset.classes).plot(ax=current_ax)
        current_ax.set_title('%s on train set of all folds' % marker)
        current_ax = ax[2, i]
        current_ax.axis('off')
        table_vals = [[str(class_i) for class_i in dataset.classes],
                      ['%1.1f' % train_p for train_p in train_precision],
                      ['%1.1f' % train_r for train_r in train_recall],
                      ['%1.1f' % train_ss for train_ss in train_sensitivity],
                      ['%1.1f' % train_sp for train_sp in train_specificity],
                      ['%1.1f' % train_j for train_j in train_jaccard]]
        row_labels = ['cls', 'pre', 'rec', 'sen', 'spe', 'jac']
        current_ax.table(cellText=table_vals, rowLabels=row_labels, cellLoc='center', loc='upper center')
        current_ax.text(0, 0.5, 'avg f1: %1.1f' % train_f1)
        current_ax.text(0, 0.3, best_model.best_params_, wrap=True)
        double_print('metrics on test set:', metrics_file)
        for j, class_j in enumerate(dataset.classes):
            double_print('class: %s: precision: %1.1f. recall: %1.1f. sensitivity: %1.1f. specificity: %1.1f. acc: %1.1f.'
                         % (class_j, test_precision[j], test_recall[j], test_sensitivity[j], test_specificity[j], test_jaccard[j]),
                         metrics_file)
        double_print('avg f1: %1.1f' % test_f1)
        current_ax = ax[3, i]
        if len(train_xs[0]) == 2:
            embedded_test_xs = test_x
        else:
            embedded_test_xs = TSNE(n_components=2, init='pca', random_state=1, n_jobs=4).fit_transform(test_x)
        x = [item[0] for item in embedded_test_xs]
        y = [item[1] for item in embedded_test_xs]

        l_x, u_x = np.percentile(x, (25, 75), interpolation='midpoint')
        lower_bound_x = max(l_x - 3.0 * (u_x - l_x), min(x))
        upper_bound_x = min(u_x + 3.0 * (u_x - l_x), max(x))
        l_y, u_y = np.percentile(y, (25, 75), interpolation='midpoint')
        lower_bound_y = max(l_y - 3.0 * (u_y - l_y), min(y))
        upper_bound_y = min(u_y + 3.0 * (u_y - l_y), max(y))
        if len(train_xs[0]) == 2:
            xx, yy = np.meshgrid(np.arange(lower_bound_x, upper_bound_x, (upper_bound_x - lower_bound_x) / 1000),
                                 np.arange(lower_bound_y, upper_bound_y, (upper_bound_y - lower_bound_y) / 1000))
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
            current_ax.contourf(xx, yy, Z, alpha=0.2)
        sns.scatterplot(x=x, y=y,
                        style=[str(item) for item in test_y],
                        style_order=[str(item) for item in dataset.classes],
                        hue=[str(item) for item in test_y],
                        hue_order=[str(item) for item in dataset.classes],
                        ax=current_ax)
        current_ax.set_title('%s on test set of the last fold' % marker)
        current_ax.legend(loc="best")
        current_ax.set_xlim(lower_bound_x, upper_bound_x)
        current_ax.set_ylim(lower_bound_y, upper_bound_y)

        current_ax = ax[4, i]
        metrics.ConfusionMatrixDisplay(test_conf_mat, display_labels=dataset.classes).plot(ax=current_ax)
        current_ax.set_title('%s on test set of all folds' % marker)
        current_ax = ax[5, i]
        current_ax.axis('off')
        table_vals = [[str(class_i) for class_i in dataset.classes],
                      ['%1.1f' % test_p for test_p in test_precision],
                      ['%1.1f' % test_r for test_r in test_recall],
                      ['%1.1f' % test_ss for test_ss in test_sensitivity],
                      ['%1.1f' % test_sp for test_sp in test_specificity],
                      ['%1.1f' % test_j for test_j in test_jaccard]]
        row_labels = ['cls', 'pre', 'rec', 'sen', 'spe', 'jac']
        current_ax.table(cellText=table_vals, rowLabels=row_labels, cellLoc='center', loc='upper center')
        current_ax.text(0, 0.2, 'avg f1: %1.1f' % test_f1)
    double_print('marker ave_trian_f1: %1.1f' % (sum(all_marker_train_f1s) / len(all_marker_train_f1s)))
    double_print('marker ave_test_f1: %1.1f' % (sum(all_marker_test_f1s) / len(all_marker_test_f1s)))
    metrics_file.close()
    save_yaml(os.path.join(exp_path, 'best_params.yaml'), best_params)
    fig.savefig(metrics_fig_filename, bbox_inches='tight', pad_inches=1)


if __name__ == '__main__':
    exp_path = os.path.join('exp', args.config, args.sub_setting)
    if not args.overwrite_config and os.path.exists(os.path.join(exp_path, 'config.yaml')):
        config = load_ymal(os.path.join(exp_path, 'config.yaml'))
    else:
        config = load_ymal(os.path.join('config', args.config + '.yaml'))
        save_yaml(os.path.join(exp_path, 'config.yaml'), config)

    if not args.retrain:
        if os.path.exists(os.path.join(exp_path, 'model')) \
                or len(glob.glob(os.path.join(exp_path, 'model', '*.pkl'))) != 0:
            raise FileExistsError('there are already models saved in %s.' % exp_path)
    maybe_create_path(exp_path)
    train_eval(config, exp_path)
