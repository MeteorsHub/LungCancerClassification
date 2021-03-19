import argparse
import glob
import os
import pickle

import matplotlib.pyplot as plt
from sklearn import metrics, base
from sklearn.model_selection import GridSearchCV

from data_loader import MarkerExpressionDataset
from utils import load_ymal, save_yaml, maybe_create_path, double_print, eval_results, get_model
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

    if dataset.feature_selection is not None or dataset.feature_transformation is not None:
        with open(os.path.join(exp_path, 'feature_selection_and_transformation.txt'), 'w') as f:
            if dataset.feature_selection is not None:
                f.write('---feature selection method: %s---\n' % dataset.feature_selection['method'])
                if 'kwargs' in dataset.feature_selection:
                    f.write('---feature selection kwargs: %s---\n' % str(dataset.feature_selection['kwargs']))
            if dataset.feature_transformation is not None:
                f.write('---feature transformation method: %s---\n' % dataset.feature_transformation['method'])
                if 'kwargs' in dataset.feature_transformation:
                    f.write('---feature transformation kwargs: %s---\n' % str(dataset.feature_transformation['kwargs']))

            for marker in dataset.markers:
                f.write('marker %s:\n' % marker)
                if dataset.fs_metric_params is not None:
                    f.write('---feature selection and transformation kwargs: %s---\n'
                            % str(dataset.fs_metric_params[marker]))

                features = dataset.features
                feature_index = 0
                f.write('---selected features---\n')
                for flag in dataset.feature_selector[marker].get_support():
                    f.write('%s:\t%s\n' % (features[feature_index], flag))
                    feature_index = (feature_index + 1) % len(features)

                components = dataset.feature_transformer[marker].components_
                f.write('---feature transformation components---:\n%s' % components)

    threshold = config.get('threshold', 'roc_optimal')
    metrics_names = ['sensitivity', 'specificity', 'roc_auc_score']
    metrics_avg_names = ['roc_auc_score_avg', 'roc_auc_score_avg_std']

    fig, ax = plt.subplots(9, len(dataset.markers), squeeze=False, figsize=(6 * len(dataset.markers), 40))
    metrics_file = open(os.path.join(exp_path, 'metrics.txt'), 'w')
    metrics_fig_filename = os.path.join(exp_path, 'conf_mat.png')
    best_params = dict()
    all_marker_train_metrics = []
    all_marker_test_metrics = []
    for i, marker in enumerate(dataset.markers):
        model = get_model(config)
        if config.get('parameter_search', True):
            # parameter search
            print('parameter search for marker %s...' % marker)
            all_x, all_y, cv_index = dataset.get_all_data(marker)
            best_model = GridSearchCV(model,
                                      param_grid=config['model_kwargs_search'],
                                      cv=cv_index,
                                      scoring='roc_auc_ovr')
            best_model.fit(all_x, all_y)
            best_params[marker] = best_model.best_params_
            print('search done')
        else:
            best_model = model
            best_params[marker] = config['model_kwargs']

        # run train and test
        train_xs = []
        train_ys = []
        train_ys_score = []
        test_xs = []
        test_ys = []
        test_ys_score = []
        for fold_i, (train_x, train_y, test_x, test_y) in enumerate(dataset.get_split_data(marker)):
            if config.get('parameter_search', True):
                model = base.clone(model)
            model.set_params(**best_params[marker])
            model.fit(train_x, train_y)
            # model.classes_ = dataset.classes
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

        train_metrics = eval_results(train_ys, train_ys_score,
                                     labels=dataset.classes, average='macro',
                                     threshold=threshold, num_fold=dataset.num_fold)
        test_metrics = eval_results(test_ys, test_ys_score,
                                    labels=dataset.classes, average='macro',
                                    threshold=train_metrics['used_threshold'], num_fold=dataset.num_fold)
        all_marker_train_metrics.append(train_metrics)
        all_marker_test_metrics.append(test_metrics)

        # print metrics to console and file
        double_print('marker: %s' % marker, metrics_file)
        double_print('metrics on training set:', metrics_file)
        for j, class_j in enumerate(dataset.classes):
            log_str = '[class: %s. threshold: %1.1f] ' % (class_j, 100 * train_metrics['used_threshold'][j])
            for metrics_name in metrics_names:
                log_str += '%s: %1.1f. ' % (metrics_name, train_metrics[metrics_name][j])
            double_print(log_str, metrics_file)
        for metrics_name in metrics_avg_names:
            double_print('%s: %1.1f' % (metrics_name, train_metrics[metrics_name]), metrics_file)
        double_print('metrics on test set:', metrics_file)
        for j, class_j in enumerate(dataset.classes):
            log_str = '[class: %s. threshold: %1.1f] ' % (class_j, 100 * test_metrics['used_threshold'][j])
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
        for j in range(len(dataset.classes)):
            roc_curve = train_metrics['roc_curve'][j]
            roc_auc_score = train_metrics['roc_auc_score'][j]
            class_id = dataset.classes[j]
            sen = train_metrics['sensitivity'][j] / 100
            spe = train_metrics['specificity'][j] / 100
            metrics.RocCurveDisplay(fpr=roc_curve[0], tpr=roc_curve[1],
                                    roc_auc=roc_auc_score, estimator_name='class %s' % class_id).plot(ax=current_ax)
            current_ax.scatter(1 - spe, sen)

        current_ax = ax[4, i]
        table_val_list = [dataset.classes, [100 * item for item in train_metrics['used_threshold']]]
        row_labels = ['cls', 'thr']
        for metrics_name in metrics_names:
            table_val_list.append(train_metrics[metrics_name])
            row_labels.append(metrics_name[:min(3, len(metrics_name))])
        additional_text = []
        for metrics_name in metrics_avg_names:
            additional_text.append('%s: %1.1f' % (metrics_name, train_metrics[metrics_name]))
        additional_text.append(best_params[marker])
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
        for j in range(len(dataset.classes)):
            roc_curve = test_metrics['roc_curve'][j]
            roc_auc_score = test_metrics['roc_auc_score'][j]
            class_id = dataset.classes[j]
            sen = test_metrics['sensitivity'][j] / 100
            spe = test_metrics['specificity'][j] / 100
            metrics.RocCurveDisplay(fpr=roc_curve[0], tpr=roc_curve[1],
                                    roc_auc=roc_auc_score, estimator_name='class %s' % class_id).plot(ax=current_ax)
            current_ax.scatter(1 - spe, sen)

        current_ax = ax[8, i]
        table_val_list = [dataset.classes, [100 * item for item in test_metrics['used_threshold']]]
        row_labels = ['cls', 'thr']
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
