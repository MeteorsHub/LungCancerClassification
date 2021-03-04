import argparse
import glob
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, metrics, naive_bayes
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize
from sklearn.tree import DecisionTreeClassifier

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


def evaluate(y_true, y_score, labels, average='macro', threshold=0.5, num_fold=None):
    """

    :param y_true: [n_samples]
    :param y_score: [n_samples, n_classes]
    :param labels: the y_score order of y_true label
    :param average:
    :param threshold: only used in binary classification
    :param num_fold:
    :return:
    """
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    all_results = dict()
    # threshold average
    if len(labels) == 2:
        binary = True
    else:
        binary = False

    roc_curves = []
    roc_auc_scores = []
    y_true_binary_all = label_binarize(y_true, classes=labels)
    for i, class_i in enumerate(labels):
        y_true_binary = y_true_binary_all[:, i]
        y_score_binary = y_score[:, i]
        roc_curves.append(metrics.roc_curve(y_true_binary, y_score_binary))
        roc_auc_scores.append(100 * metrics.roc_auc_score(y_true_binary, y_score_binary))
    all_results['roc_curve'] = roc_curves
    all_results['roc_auc_score'] = roc_auc_scores
    all_results['roc_auc_score_avg'] = 100 * metrics.roc_auc_score(
        y_true, y_score, average=average, multi_class='ovr', labels=labels)

    # threshold specific
    if binary:
        y_pred = np.where(y_score[:, -1] > threshold,
                          labels[-1] * np.ones_like(y_true),
                          labels[0] * np.ones_like(y_true))
    else:
        y_pred = np.argmax(y_score, 1)
        for sample_i in range(len(y_pred)):
            y_pred[sample_i] = labels[y_pred[sample_i]]
    all_results['conf_mat'] = metrics.confusion_matrix(y_true, y_pred, labels=labels)

    tps = np.diagonal(all_results['conf_mat'])
    ps = np.sum(all_results['conf_mat'], 1)
    sensitivity = 100 * tps / ps
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
    all_results['sensitivity'] = sensitivity
    all_results['specificity'] = specificity
    all_results['f1_avg'] = 100 * metrics.f1_score(y_true, y_pred, average=average)
    all_results['f1'] = 100 * metrics.f1_score(y_true, y_pred, average=None)

    if num_fold is not None:
        all_fold_roc_auc_scores = []
        all_fold_f1s = []
        for fold_i in range(num_fold):
            start_i = fold_i * len(y_true) // num_fold
            end_i = (fold_i + 1) * len(y_true) // num_fold
            all_fold_f1s.append(100 * metrics.f1_score(y_true[start_i:end_i], y_pred[start_i:end_i], average=average))
            all_fold_roc_auc_scores.append(100 * metrics.roc_auc_score(
                y_true[start_i:end_i], y_score[start_i:end_i], average=average, multi_class='ovr', labels=labels))
        all_results['f1_avg_std'] = np.std(all_fold_f1s)
        all_results['roc_auc_score_avg_std'] = np.std(all_fold_roc_auc_scores)
    return all_results


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
    if dataset.feature_selection is not None:
        with open(os.path.join(exp_path, 'feature_selection.txt'), 'w') as f:
            f.write('---feature selection method: %s---\n' % dataset.feature_selection['method'])
            f.write('---feature selection kwargs: %s---\n' % str(dataset.feature_selection['kwargs']))
            for marker in dataset.markers:
                f.write('marker %s:\n' % marker)
                features = dataset.features
                feature_index = 0
                for flag in dataset.feature_selector[marker].get_support():
                    f.write('%s:\t%s\n' % (features[feature_index], flag))
                    feature_index = (feature_index + 1) % len(features)

    if config['model'] == 'svm':
        model_class = svm.SVC
    elif config['model'] == 'rdf':
        model_class = RandomForestClassifier
    elif config['model'] == 'adaboost':
        model_class = AdaBoostClassifier
    elif config['model'] == 'gradient_boost':
        model_class = GradientBoostingClassifier
    elif config['model'] == 'gaussion_bayes':
        model_class = naive_bayes.GaussianNB
    elif config['model'] == 'mlp':
        model_class = MLPClassifier
    elif config['model'] == 'k_neighbors':
        model_class = KNeighborsClassifier
    elif config['model'] == 'decision_tree':
        model_class = DecisionTreeClassifier
    else:
        raise AttributeError('unrecognized model %s' % config['model'])

    threshold = config.get('threshold', 0.5)
    metrics_names = ['sensitivity', 'specificity', 'f1', 'roc_auc_score']
    metrics_avg_names = ['f1_avg', 'f1_avg_std', 'roc_auc_score_avg', 'roc_auc_score_avg_std']

    fig, ax = plt.subplots(8, len(dataset.markers), squeeze=False, figsize=(6 * len(dataset.markers), 30))
    metrics_file = open(os.path.join(exp_path, 'metrics.txt'), 'w')
    metrics_fig_filename = os.path.join(exp_path, 'conf_mat.png')
    best_params = dict()
    all_marker_train_metrics = []
    all_marker_test_metrics = []
    for i, marker in enumerate(dataset.markers):
        # parameter search
        print('parameter search for marker %s...' % marker)
        all_x, all_y, cv_index = dataset.get_all_data(marker)
        best_model = GridSearchCV(model_class(),
                                  param_grid=config['model_kwargs'],
                                  cv=cv_index,
                                  scoring='roc_auc_ovr')
        best_model.fit(all_x, all_y)
        best_params[marker] = best_model.best_params_
        print('search done')

        # run train and test
        train_xs = []
        train_ys = []
        train_ys_score = []
        test_xs = []
        test_ys = []
        test_ys_score = []
        for fold_i, (train_x, train_y, test_x, test_y) in enumerate(dataset.get_split_data(marker)):
            model = model_class()
            model.set_params(**best_params[marker])
            model.classes_ = dataset.classes
            model.fit(train_x, train_y)
            train_xs += train_x
            train_ys += train_y
            test_xs += test_x
            test_ys += test_y
            train_y_score = model.predict_proba(train_x).tolist()
            train_ys_score += train_y_score
            test_y_score = model.predict_proba(test_x).tolist()
            test_ys_score += test_y_score
            model_filename = os.path.join(exp_path, 'model', '%s_%s_fold_%d.pkl'
                                          % (config['model'], marker, fold_i))
            maybe_create_path(os.path.dirname(model_filename))
            with open(model_filename, 'wb') as f:
                pickle.dump(model, f)

        train_metrics = evaluate(train_ys, train_ys_score,
                                 labels=dataset.classes, average='macro',
                                 threshold=threshold, num_fold=dataset.num_fold)
        test_metrics = evaluate(test_ys, test_ys_score,
                                labels=dataset.classes, average='macro',
                                threshold=threshold, num_fold=dataset.num_fold)
        all_marker_train_metrics.append(train_metrics)
        all_marker_test_metrics.append(test_metrics)

        # print metrics to console and file
        double_print('marker: %s' % marker, metrics_file)
        double_print('metrics on training set:', metrics_file)
        for j, class_j in enumerate(dataset.classes):
            log_str = '[class: %s] ' % class_j
            for metrics_name in metrics_names:
                log_str += '%s: %1.1f. ' % (metrics_name, train_metrics[metrics_name][j])
            double_print(log_str, metrics_file)
        for metrics_name in metrics_avg_names:
            double_print('%s: %1.1f' % (metrics_name, train_metrics[metrics_name]), metrics_file)
        double_print('metrics on test set:', metrics_file)
        for j, class_j in enumerate(dataset.classes):
            log_str = '[class: %s] ' % class_j
            for metrics_name in metrics_names:
                log_str += '%s: %1.1f. ' % (metrics_name, test_metrics[metrics_name][j])
            double_print(log_str, metrics_file)
        for metrics_name in metrics_avg_names:
            double_print('%s: %1.1f' % (metrics_name, test_metrics[metrics_name]), metrics_file)
        double_print('metrics on test set:', metrics_file)

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
        metrics.ConfusionMatrixDisplay(train_metrics['conf_mat'], display_labels=dataset.classes).plot(ax=current_ax)
        current_ax.set_title('%s on train set of all folds' % marker)

        current_ax = ax[3, i]
        for roc_curve, roc_auc_score in zip(train_metrics['roc_curve'], train_metrics['roc_auc_score']):
            metrics.RocCurveDisplay(fpr=roc_curve[0], tpr=roc_curve[1], roc_auc=roc_auc_score).plot(ax=current_ax)

        current_ax = ax[4, i]
        table_val_list = [dataset.classes]
        row_labels = ['cls']
        for metrics_name in metrics_names:
            table_val_list.append(train_metrics[metrics_name])
            row_labels.append(metrics_name[:min(3, len(metrics_name))])
        additional_text = []
        for metrics_name in metrics_avg_names:
            additional_text.append('%s: %1.1f' % (metrics_name, train_metrics[metrics_name]))
        additional_text.append(best_model.best_params_)
        plot_table(table_val_list, row_labels, ax=current_ax, additional_text=additional_text)

        current_ax = ax[5, i]
        contour_flag = len(train_xs[0]) == 2
        test_y_str = [str(item) for item in test_y]
        classes_str = [str(item) for item in dataset.classes]
        plot_feature_distribution(test_x, ax=current_ax, t_sne=True,
                                  hue=test_y_str, hue_order=classes_str,
                                  style=test_y_str, style_order=classes_str,
                                  x_lim='min_max_extend', y_lim='min_max_extend',
                                  contour=contour_flag, z_generator=model.predict)
        current_ax.set_title('%s on test set of the last fold' % marker)

        current_ax = ax[6, i]
        metrics.ConfusionMatrixDisplay(test_metrics['conf_mat'], display_labels=dataset.classes).plot(ax=current_ax)
        current_ax.set_title('%s on test set of all folds' % marker)

        current_ax = ax[7, i]
        table_val_list = [dataset.classes]
        row_labels = ['cls']
        for metrics_name in metrics_names:
            table_val_list.append(test_metrics[metrics_name])
            row_labels.append(metrics_name[:min(3, len(metrics_name))])
        additional_text = []
        for metrics_name in metrics_avg_names:
            additional_text.append('%s: %1.1f' % (metrics_name, test_metrics[metrics_name]))
        plot_table(table_val_list, row_labels, ax=current_ax, additional_text=additional_text)

    for metrics_name in metrics_avg_names:
        all_marker_values = [item[metrics_name] for item in all_marker_train_metrics]
        double_print('marker train %s: %1.1f' % (metrics_name, sum(all_marker_values) / len(all_marker_values)),
                     metrics_file)
    for metrics_name in metrics_avg_names:
        all_marker_values = [item[metrics_name] for item in all_marker_test_metrics]
        double_print('marker test %s: %1.1f' % (metrics_name, sum(all_marker_values) / len(all_marker_values)),
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
