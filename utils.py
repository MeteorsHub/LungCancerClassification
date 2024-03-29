import csv
import os

import numpy as np
import yaml
from metric_learn import LFDA, LMNN, MMC_Supervised
from sklearn import svm, metrics, naive_bayes
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, \
    VotingClassifier, StackingClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize
from sklearn.tree import DecisionTreeClassifier


def read_dict_csv(filename: str, return_fieldnames=False) -> (list, list):
    with open(filename, encoding='utf8') as f:
        f_csv = csv.DictReader(f)
        data = list(f_csv)
        field_names = f_csv.fieldnames
    if return_fieldnames:
        return data, field_names
    else:
        return data


def write_dict_csv(filename: str, fieldnames: list, data: list):
    with open(filename, 'w', newline='', encoding='utf8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for item in data:
            writer.writerow(item)


def load_ymal(config_filename) -> dict:
    with open(config_filename, 'r', encoding='utf8') as f:
        data = yaml.safe_load(f)
    return data


def save_yaml(config_filename, data):
    maybe_create_path(os.path.dirname(config_filename))
    with open(config_filename, 'w', encoding='utf8') as f:
        yaml.safe_dump(data, f, sort_keys=False)


def maybe_create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def double_print(s, log_file=None, **kwargs):
    print(s, **kwargs)
    if log_file is not None:
        print(s, file=log_file)


def get_bounds(feature, method='min_max', extend_factor=0.1):
    assert method in ['min_max', 'min_max_extend', 'box']

    if method == 'min_max':
        lower_bound = min(feature)
        upper_bound = max(feature)
    elif method == 'min_max_extend':
        min_val = min(feature)
        max_val = max(feature)
        lower_bound = min_val - extend_factor*(max_val - min_val)
        upper_bound = max_val + extend_factor*(max_val - min_val)
    elif method == 'box':
        l, u = np.percentile(feature, (25, 75), interpolation='midpoint')
        lower_bound = l - 1.5 * (u - l)
        upper_bound = u + 1.5 * (u - l)
    else:
        return
    if lower_bound == upper_bound:
        lower_bound -= 0.25
        upper_bound += 1
    return lower_bound, upper_bound


def unstack(a, axis=0):
    return [np.squeeze(e, axis) for e in np.split(a, a.shape[axis], axis=axis)]


def find_roc_optimal_cutoff_point(fprs, tprs, thresholds, epsilon=1e-3):
    assert len(fprs) == len(tprs) == len(thresholds)
    cutoff_fprs = [fprs[0]]
    cutoff_tprs = [tprs[0]]
    cutoff_thresholds = [thresholds[0]]
    objective = (1 - cutoff_fprs[0]) + cutoff_tprs[0]
    for fpr_, tpr_, threshold_ in zip(fprs, tprs, thresholds):
        new_objective = (1 - fpr_) + tpr_
        if new_objective > objective + epsilon:
            cutoff_fprs = [fpr_]
            cutoff_tprs = [tpr_]
            cutoff_thresholds = [threshold_]
            objective = new_objective
        elif abs(new_objective - objective) < epsilon:
            cutoff_fprs.append(fpr_)
            cutoff_tprs.append(tpr_)
            cutoff_thresholds.append(threshold_)
    index = len(cutoff_tprs) // 2
    return cutoff_fprs[index], cutoff_tprs[index], cutoff_thresholds[index]


def eval_results(y_true, y_score, labels, average='macro', threshold='roc_optimal', num_fold=None):
    """
    Note that in multi-class problems, all metrics are multiple binary-class metrics.
    :param y_true: [n_samples]
    :param y_score: [n_samples, n_classes]
    :param labels: the y_score order of y_true label
    :param average:
    :param threshold: list of float (size labels), single float or 'roc_optimal'. only used in binary-class metrics
    :param num_fold: if not None, compute std metrics
    :return:
    """
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    if isinstance(threshold, list) or isinstance(threshold, tuple):
        assert len(threshold) == len(labels)
    else:
        threshold = [threshold for _ in labels]
    all_results = dict()
    # threshold average
    if len(labels) == 2:
        binary = True
    else:
        binary = False

    roc_curves = []
    roc_auc_scores = []
    used_thresholds = []
    sensitivity = []
    specificity = []
    y_true_binary_all = label_binarize(y_true, classes=labels)
    if binary:
        y_true_binary_all = np.concatenate([1 - y_true_binary_all, y_true_binary_all], -1)
    for i, class_i in enumerate(labels):
        y_true_binary = y_true_binary_all[:, i]
        y_score_binary = y_score[:, i]
        roc_curve = metrics.roc_curve(y_true_binary, y_score_binary)
        if threshold[i] == 'roc_optimal':
            _, _, used_threshold = find_roc_optimal_cutoff_point(roc_curve[0], roc_curve[1], roc_curve[2])
        else:
            used_threshold = threshold[i]
        used_thresholds.append(used_threshold)
        roc_curves.append(roc_curve)
        y_label_binary = (y_score_binary > used_threshold).astype(np.int32)
        roc_auc_scores.append(100 * metrics.roc_auc_score(y_true_binary, y_score_binary))
        sensitivity.append(100 * metrics.recall_score(y_true_binary, y_label_binary))
        specificity.append(100 * metrics.recall_score(1 - y_true_binary, 1 - y_label_binary))
    all_results['roc_curve'] = roc_curves
    all_results['roc_auc_score'] = roc_auc_scores
    all_results['used_threshold'] = used_thresholds
    all_results['sensitivity'] = sensitivity
    all_results['specificity'] = specificity
    if binary:
        all_results['roc_auc_score_avg'] = 100 * metrics.roc_auc_score(
            y_true, y_score[:, -1], average=average, labels=labels)
    else:
        all_results['roc_auc_score_avg'] = 100 * metrics.roc_auc_score(
            y_true, y_score, average=average, multi_class='ovr', labels=labels)

    # threshold specific
    if binary:
        y_pred = np.where(y_score[:, -1] > used_thresholds[-1],
                          labels[-1] * np.ones_like(y_true),
                          labels[0] * np.ones_like(y_true))
    else:
        y_pred = np.argmax(y_score, 1)
        for sample_i in range(len(y_pred)):
            y_pred[sample_i] = labels[y_pred[sample_i]]
    all_results['conf_mat'] = metrics.confusion_matrix(y_true, y_pred, labels=labels)

    # all_results['f1_avg'] = 100 * metrics.f1_score(y_true, y_pred, average=average)
    # all_results['f1'] = 100 * metrics.f1_score(y_true, y_pred, average=None)

    if num_fold is not None:
        all_fold_roc_auc_scores = []
        # all_fold_f1s = []
        for fold_i in range(num_fold):
            start_i = fold_i * len(y_true) // num_fold
            end_i = (fold_i + 1) * len(y_true) // num_fold
            # all_fold_f1s.append(100 * metrics.f1_score(y_true[start_i:end_i], y_pred[start_i:end_i], average=average))
            if binary:
                all_fold_roc_auc_scores.append(100 * metrics.roc_auc_score(
                    y_true[start_i:end_i], y_score[start_i:end_i, -1], average=average, labels=labels))
            else:
                all_fold_roc_auc_scores.append(100 * metrics.roc_auc_score(
                    y_true[start_i:end_i], y_score[start_i:end_i], average=average, multi_class='ovr', labels=labels))
        # all_results['f1_avg_std'] = np.std(all_fold_f1s)
        all_results['roc_auc_score_avg_std'] = np.std(all_fold_roc_auc_scores)
    return all_results


def get_model(config):
    if config is None:
        return None
    if 'model_kwargs' not in config:
        model_kwargs = dict()
    else:
        model_kwargs = config['model_kwargs']
    if config['model'] == 'svm':
        model = svm.SVC(**model_kwargs)
    elif config['model'] == 'rdf':
        model = RandomForestClassifier(**model_kwargs)
    elif config['model'] == 'adaboost':
        if 'sub_model' in config:
            base_estimator = get_model(config['sub_model'])
        else:
            base_estimator = None
        model = AdaBoostClassifier(base_estimator=base_estimator, **model_kwargs)
    elif config['model'] == 'gradient_boost':
        model = GradientBoostingClassifier(**model_kwargs)
    elif config['model'] == 'gaussion_bayes':
        model = naive_bayes.GaussianNB(**model_kwargs)
    elif config['model'] == 'mlp':
        model = MLPClassifier(**model_kwargs)
    elif config['model'] == 'k_neighbors':
        model = KNeighborsClassifier(**model_kwargs)
    elif config['model'] == 'decision_tree':
        model = DecisionTreeClassifier(**model_kwargs)
    elif config['model'] == 'voting':
        sub_models = []
        for sub_model in config['sub_model']:
            sub_models.append([sub_model['model_name'], get_model(sub_model)])
        model = VotingClassifier(sub_models, **model_kwargs)
    elif config['model'] == 'stacking':
        final_estimator = get_model(config.get('final_model', None))
        sub_models = []
        for sub_model in config['sub_model']:
            sub_models.append((sub_model['model_name'], get_model(sub_model)))
        model = StackingClassifier(estimators=sub_models, final_estimator=final_estimator, **model_kwargs)
    elif config['model'] == 'bagging':
        base_estimator = get_model(config.get('sub_model', None))
        model = BaggingClassifier(base_estimator=base_estimator, **model_kwargs)
    elif config['model'] in ['lfda', 'lmnn', 'mmc']:
        if config['model'] == 'lfda':
            metric_learner = LFDA(**model_kwargs)
        elif config['model'] == 'lmnn':
            metric_learner = LMNN(**model_kwargs)
        elif config['model'] == 'mmc':
            metric_learner = MMC_Supervised(**model_kwargs)
        else:
            raise AttributeError
        if 'final_model' in config:
            final_model = get_model(config['final_model'])
        else:
            final_model = KNeighborsClassifier()
        model = Pipeline([('metric', metric_learner), ('final', final_model)])
    else:
        raise AttributeError('unrecognized model %s' % config['model'])
    return model
