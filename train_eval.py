import argparse
import glob
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, metrics, naive_bayes
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

from data_loader import MarkerExpressionDataset
from utils import load_ymal, save_yaml, maybe_create_path, double_print
from visualization import plot_table, plot_feature_distribution

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


def evaluate(y_true, y_pred, labels=(1,), average=None, num_fold=None):
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
    f1 = 100 * metrics.f1_score(y_true, y_pred, average='macro')

    if num_fold is None:
        return precision, recall, sensitivity, specificity, jaccard, conf_mat, f1
    all_fold_f1s = []
    for fold_i in range(num_fold):
        start_i = fold_i * len(y_true) // num_fold
        end_i = (fold_i + 1) * len(y_true) // num_fold
        all_fold_f1s.append(100 * metrics.f1_score(y_true[start_i:end_i], y_pred[start_i:end_i], average='macro'))
    f1_std = np.std(all_fold_f1s)
    return precision, recall, sensitivity, specificity, jaccard, conf_mat, f1, f1_std


def train_eval(config, exp_path):
    dataset = MarkerExpressionDataset(config)
    if dataset.data_clean is not None:
        with open(os.path.join(exp_path, 'dirty_data.txt'), 'w') as f:
            f.write('---data clean method: %s---\n' % dataset.data_clean)
            for marker, item in dataset.outlier_samples.items():
                f.write('marker %s:\n' % marker)
                for class_id in dataset.classes:
                    f.write('class %s:\n' % class_id)
                    for sample_id in item.keys():
                        if item[sample_id]['class'] == class_id:
                            f.write('\t%s\n' % sample_id)

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

    fig, ax = plt.subplots(7, len(dataset.markers), squeeze=False, figsize=(6*len(dataset.markers), 30))
    metrics_file = open(os.path.join(exp_path, 'metrics.txt'), 'w')
    metrics_fig_filename = os.path.join(exp_path, 'conf_mat.png')
    best_params = dict()
    all_marker_train_f1s = []
    all_marker_train_f1_stds = []
    all_marker_test_f1s = []
    all_marker_test_f1_stds = []
    for i, marker in enumerate(dataset.markers):
        # parameter search
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

        # run train and test
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
        train_precision, train_recall, train_sensitivity, train_specificity, \
        train_jaccard, train_conf_mat, train_f1, train_f1_std = \
            evaluate(train_ys, pred_train_ys, dataset.classes, None, num_fold=dataset.num_fold)
        test_precision, test_recall, test_sensitivity, test_specificity, \
        test_jaccard, test_conf_mat, test_f1, test_f1_std = \
            evaluate(test_ys, pred_test_ys, dataset.classes, None, num_fold=dataset.num_fold)
        all_marker_train_f1s.append(train_f1)
        all_marker_train_f1_stds.append(train_f1_std)
        all_marker_test_f1s.append(test_f1)
        all_marker_test_f1_stds.append(test_f1_std)

        # print metrics to console and file
        double_print('marker: %s' % marker, metrics_file)
        double_print('metrics on training set:', metrics_file)
        for j, class_j in enumerate(dataset.classes):
            double_print(
                'class: %s: precision: %1.1f. recall: %1.1f. sensitivity: %1.1f. specificity: %1.1f. acc: %1.1f.'
                % (class_j, train_precision[j], train_recall[j], train_sensitivity[j], train_specificity[j],
                   train_jaccard[j]),
                metrics_file)
        double_print('avg f1: %1.1f' % train_f1, metrics_file)
        double_print('f1 std: %1.1f' % train_f1_std, metrics_file)
        double_print('metrics on test set:', metrics_file)
        for j, class_j in enumerate(dataset.classes):
            double_print(
                'class: %s: precision: %1.1f. recall: %1.1f. sensitivity: %1.1f. specificity: %1.1f. acc: %1.1f.'
                % (
                    class_j, test_precision[j], test_recall[j], test_sensitivity[j], test_specificity[j],
                    test_jaccard[j]),
                metrics_file)
        double_print('avg f1: %1.1f' % test_f1, metrics_file)
        double_print('f1 std: %1.1f' % test_f1_std, metrics_file)

        # generate figure
        current_ax = ax[0, i]
        dataset.plot_data_clean_distribution(current_ax, marker)
        current_ax.set_title('data cleaning on marker %s' % marker)

        current_ax = ax[1, i]
        contour_flag = len(train_xs[0]) == 2
        # dup_reduced = list(tuple(tuple([train_xs[j] + [train_ys[j]] for j in range(len(train_xs))])))
        # dup_reduced_train_xs = [item[:-1] for item in dup_reduced]
        # dup_reduced_train_ys = [item[-1] for item in dup_reduced]
        # dup_reduced_train_ys_str = [str(item) for item in dup_reduced_train_ys]
        dup_reduced_train_xs = train_x + test_x
        dup_reduced_train_ys = train_y + test_y
        dup_reduced_train_ys_str = [str(item) for item in dup_reduced_train_ys]
        classes_str = [str(item) for item in dataset.classes]
        plot_feature_distribution(dup_reduced_train_xs, ax=current_ax, t_sne=True,
                                  hue=dup_reduced_train_ys_str, hue_order=classes_str,
                                  style=dup_reduced_train_ys_str, style_order=classes_str,
                                  x_lim='min_max_extend', y_lim='min_max_extend',
                                  contour=contour_flag, z_generator=best_model.predict)
        current_ax.set_title('%s trained on whole set' % marker)

        current_ax = ax[2, i]
        metrics.ConfusionMatrixDisplay(train_conf_mat, display_labels=dataset.classes).plot(ax=current_ax)
        current_ax.set_title('%s on train set of all folds' % marker)

        current_ax = ax[3, i]
        table_val_list = [dataset.classes,
                          train_precision, train_recall, train_sensitivity, train_specificity, train_jaccard]
        row_labels = ['cls', 'pre', 'rec', 'sen', 'spe', 'jac']
        additional_text = ['avg f1: %1.1f' % train_f1, 'f1 std: %1.1f' % train_f1_std, best_model.best_params_, ]
        plot_table(table_val_list, row_labels, ax=current_ax, additional_text=additional_text)

        current_ax = ax[4, i]
        contour_flag = len(train_xs[0]) == 2
        test_y_str = [str(item) for item in test_y]
        classes_str = [str(item) for item in dataset.classes]
        plot_feature_distribution(test_x, ax=current_ax, t_sne=True,
                                  hue=test_y_str, hue_order=classes_str,
                                  style=test_y_str, style_order=classes_str,
                                  x_lim='min_max_extend', y_lim='min_max_extend',
                                  contour=contour_flag, z_generator=model.predict)
        current_ax.set_title('%s on test set of the last fold' % marker)

        current_ax = ax[5, i]
        metrics.ConfusionMatrixDisplay(test_conf_mat, display_labels=dataset.classes).plot(ax=current_ax)
        current_ax.set_title('%s on test set of all folds' % marker)

        current_ax = ax[6, i]
        table_val_list = [dataset.classes,
                          test_precision, test_recall, test_sensitivity, test_specificity, test_jaccard]
        row_labels = ['cls', 'pre', 'rec', 'sen', 'spe', 'jac']
        additional_text = ['avg f1: %1.1f' % test_f1, 'f1 std: %1.1f' % test_f1_std]
        plot_table(table_val_list, row_labels, ax=current_ax, additional_text=additional_text)

    double_print('marker ave_trian_f1: %1.1f' % (sum(all_marker_train_f1s) / len(all_marker_train_f1s)), metrics_file)
    double_print('marker ave_trian_f1_std: %1.1f' % (sum(all_marker_train_f1_stds) / len(all_marker_train_f1_stds)),
                 metrics_file)
    double_print('marker ave_test_f1: %1.1f' % (sum(all_marker_test_f1s) / len(all_marker_test_f1s)), metrics_file)
    double_print('marker ave_test_f1_std: %1.1f' % (sum(all_marker_test_f1_stds) / len(all_marker_test_f1_stds)),
                 metrics_file)
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
