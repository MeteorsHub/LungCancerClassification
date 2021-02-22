import itertools
import os
import random

from scipy.special import comb
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from utils import read_dict_csv


class MarkerExpressionDataset:
    def __init__(self, config):
        self.data_root = config['data_root']
        self.marker_mapping = config['marker_mapping']
        self.features = config['features']
        self.data_clean = config.get('data_clean', None)
        self.num_split_fold = int(config['num_split_fold'])
        self.random_seed = float(config['split_random_seed'])
        self.class_mapping = config['class_mapping']
        self.default_missing_value = config['default_missing_value']
        self.num_samples_in_each_bag = int(config.get('num_samples_in_each_bag', 1))
        self.num_bags_limitation = int(config.get('num_bags_limitation', 10000))

        # one bag for the mode contains multiple samples with the same class in original data
        self.sample_data = dict()  # {marker: [fold_0: {id: {name, class, feature}}, fold_1, fold_2...]}.
        self.outlier_samples = dict()  # {marker: {id: {name, class, feature}}}
        self.num_samples = dict()  # {marker: [num_fold_0, num_fold_1...]}
        self.bag_data = dict()  # {marker: [fold_0: [[case_0_sample_ids], [case_1_sample_ids]...], fold_1...]}
        self.k_fold_splitter = None
        self.classes = []
        self.markers = []

        if not os.path.exists(self.data_root):
            raise FileNotFoundError('cannot locate data root dir %s' % self.data_root)
        for marker in self.marker_mapping.keys():
            if not os.path.exists(os.path.join(self.data_root, marker + '.csv')):
                raise FileNotFoundError('cannot find %s' % os.path.join(self.data_root, marker + '.csv'))
        if self.num_split_fold > 10 or self.num_split_fold < 2:
            raise ValueError('num_split_fold must be within 2 and 10')
        assert 0 < self.random_seed < 1, 'random_seed must be 0-1'
        if self.default_missing_value is not None:
            self.default_missing_value = float(self.default_missing_value)
        class_mapping = dict()
        for k, v in self.class_mapping.items():
            class_mapping[int(k)] = int(v)
        self.class_mapping = class_mapping
        self.k_fold_splitter = StratifiedKFold(n_splits=self.num_split_fold)
        self.classes = list(set(list(self.class_mapping.values())))
        self.classes.sort(key=list(self.class_mapping.values()).index)
        self.markers = list(set(list(self.marker_mapping.values())))
        self.markers.sort(key=list(self.marker_mapping.values()).index)
        for marker in self.markers:
            self.sample_data[marker] = dict()
        assert self.num_samples_in_each_bag > 0
        random.seed(self.random_seed)
        self.load()

    def load(self):
        # load data from file
        src_data = dict()  # {id: {name, class, feature: {src_marker}}}
        for src_marker in self.marker_mapping.keys():
            src_marker_data = read_dict_csv(os.path.join(self.data_root, src_marker + '.csv'))
            for item in src_marker_data:
                # skip ignored classes
                if int(item['class']) not in self.class_mapping:
                    continue
                item_id = item['id']
                if item_id not in src_data:
                    src_data[item_id] = {
                        'name': item['name'],
                        'class': self.class_mapping[int(item['class'])],
                        'feature': dict()
                    }
                else:
                    if self.class_mapping[int(item['class'])] != src_data[item_id]['class']:
                        raise ValueError('class %d for %s in %s is different with others'
                                         % (self.class_mapping[int(item['class'])], item_id, src_marker))
                    if item['name'] != src_data[item_id]['name']:
                        raise ValueError('name %s for %s in %s is different with others'
                                         % (item['name'], item_id, src_marker))
                src_data[item_id]['feature'][src_marker] = [float(item[feat]) for feat in self.features]

        # marker mapping with missing values
        for dst_marker in self.markers:
            src_markers = [key for key in self.marker_mapping.keys() if self.marker_mapping[key] == dst_marker]
            for item_id, item in src_data.items():
                if self.default_missing_value is None:
                    if any([src_marker not in item['feature'] for src_marker in src_markers]):
                        continue
                feature = []
                for src_marker in src_markers:
                    if src_marker not in item['feature']:
                        feature += [self.default_missing_value for _ in range(len(self.features))]
                    else:
                        feature += item['feature'][src_marker]
                self.sample_data[dst_marker][item_id] = {
                    'name': item['name'],
                    'class': item['class'],
                    'feature': feature
                }

        # data clean
        if self.data_clean is not None:
            for marker in self.markers:
                self.outlier_samples[marker] = dict()
                for class_i in self.classes:
                    if self.data_clean == 'isolation_forest':
                        clf = IsolationForest(n_estimators=50, random_state=int(self.random_seed * 100))
                    elif self.data_clean == 'local_outlier_factor':
                        clf = LocalOutlierFactor(n_neighbors=5)
                    elif self.data_clean == 'robust_covariance':
                        clf = EllipticEnvelope(random_state=int(self.random_seed * 100))
                    elif self.data_clean == 'one_class_svm':
                        clf = OneClassSVM(kernel='linear')
                    else:
                        raise AttributeError('unrecognized data clean method: %s' % self.data_clean['method'])

                    sample_ids = [s_id for s_id in list(self.sample_data[marker].keys())
                                  if self.sample_data[marker][s_id]['class'] == class_i]
                    sample_features = [self.sample_data[marker][s_id]['feature'] for s_id in sample_ids]
                    sample_labels = clf.fit_predict(sample_features)
                    outlier_ids = [sample_ids[i] for i in range(len(sample_ids)) if sample_labels[i] < 0]
                    for outlier_id in outlier_ids:
                        self.outlier_samples[marker][outlier_id] = self.sample_data[marker].pop(outlier_id)

        # split fold
        for marker in self.markers:
            self.num_samples[marker] = []
            sample_ids = list(self.sample_data[marker].keys())
            random.shuffle(sample_ids)
            sample_classes = [self.sample_data[marker][sample_id]['class'] for sample_id in sample_ids]

            samples_all_fold = []
            for _, test_indexes in self.k_fold_splitter.split(sample_ids, sample_classes):
                sample_ids_fold = [sample_ids[test_index] for test_index in test_indexes]
                self.num_samples[marker] = len(sample_ids_fold)
                samples_in_fold = dict()
                for sample_id in sample_ids_fold:
                    samples_in_fold[sample_id] = self.sample_data[marker][sample_id]
                samples_all_fold.append(samples_in_fold)
            self.sample_data[marker] = samples_all_fold

        # generate bags from samples
        for marker in self.markers:
            self.bag_data[marker] = []
            for fold_i in range(self.num_split_fold):
                class_ids = dict()
                for class_i in self.classes:
                    class_ids[class_i] = []
                for item_id, item in self.sample_data[marker][fold_i].items():
                    class_ids[item['class']].append(item_id)

                all_bags = []
                for class_i in self.classes:
                    if len(class_ids[class_i]) < self.num_samples_in_each_bag:
                        raise ValueError('num_samples_in_each_bag (%d) is less than '
                                         'num_samples (%d) for class %s in marker %s of fold %s'
                                         % (self.num_samples_in_each_bag, len(class_ids[class_i]),
                                            class_i, marker, fold_i))
                    # if perm(len(class_ids[class_i]), self.num_samples_in_each_bag) <= self.num_bags_limitation:
                    #     print('marker %s class %s fold %s use permutation.' % (marker, class_i, fold_i))
                    #     for bag in itertools.permutations(class_ids[class_i], self.num_samples_in_each_bag):
                    #         all_bags.append(bag)
                    if comb(len(class_ids[class_i]), self.num_samples_in_each_bag) <= self.num_bags_limitation:
                        print('marker %s class %s fold %s use combination.' % (marker, class_i, fold_i))
                        for bag in itertools.combinations(class_ids[class_i], self.num_samples_in_each_bag):
                            all_bags.append(bag)
                    else:
                        print('marker %s class %s fold %s use random select.' % (marker, class_i, fold_i))
                        for _ in range(self.num_bags_limitation):
                            all_bags.append(random.sample(class_ids[class_i], k=self.num_samples_in_each_bag))
                random.shuffle(all_bags)
                self.bag_data[marker].append(all_bags)

    def get_all_data(self, marker):
        assert marker in self.markers, 'marker %s not in self.markers' % marker

        all_x = []
        all_y = []
        split_index = [0]
        for fold_i in range(self.num_split_fold):
            fold_bags = self.bag_data[marker][fold_i]
            keep_num = min(len(self.bag_data[marker][fold_i]), self.num_bags_limitation//(self.num_split_fold*10))
            fold_bags = random.sample(fold_bags, k=keep_num)
            split_index.append(split_index[-1] + keep_num)
            fold_x = []
            fold_y = []
            for bag in fold_bags:
                x = []
                y = None
                for sample_id in bag:
                    x += self.sample_data[marker][fold_i][sample_id]['feature']
                    if y is not None and y != self.sample_data[marker][fold_i][sample_id]['class']:
                        raise ValueError('class id dis-match in marker %s' % marker)
                    y = self.sample_data[marker][fold_i][sample_id]['class']
                fold_x.append(x)
                fold_y.append(y)
            all_x += fold_x
            all_y += fold_y
        cv_index = []
        for split_s, split_e in zip(split_index[:-1], split_index[1:]):
            train_index = list(range(len(all_x)))
            test_index = []
            for index in range(split_s, split_e):
                train_index.remove(index)
                test_index.append(index)
            cv_index.append((train_index, test_index))
        return all_x, all_y, cv_index

    def get_split_data(self, marker):
        assert marker in self.markers, 'marker %s not in self.markers' % marker

        for test_fold_i in range(self.num_split_fold):
            train_x = []
            train_y = []
            test_x = []
            test_y = []
            for fold_i in range(self.num_split_fold):
                for bag_sample_ids in self.bag_data[marker][fold_i]:
                    bag_x = []
                    bag_y = None
                    for sample_id in bag_sample_ids:
                        bag_x += self.sample_data[marker][fold_i][sample_id]['feature']
                        if bag_y is not None and bag_y != self.sample_data[marker][fold_i][sample_id]['class']:
                            raise ValueError('class id dis-match in marker %s' % marker)
                        bag_y = self.sample_data[marker][fold_i][sample_id]['class']
                    if test_fold_i == fold_i:
                        test_x.append(bag_x)
                        test_y.append(bag_y)
                    else:
                        train_x.append(bag_x)
                        train_y.append(bag_y)
            yield train_x, train_y, test_x, test_y

    def plot_data_clean_distribution(self, ax, marker):
        clean_data = dict()
        for fold_data in self.sample_data[marker]:
            clean_data = {**clean_data, **fold_data}
        dirty_data = self.outlier_samples[marker]

        labels = []
        for class_id in self.classes:
            labels.append(str(class_id))
            labels.append('%s_dirty' % class_id)
        all_samples = []
        all_labels = []
        for clean_sample_id, clean_sample_data in clean_data.items():
            all_labels.append(str(clean_sample_data['class']))
            all_samples.append(clean_sample_data['feature'])
        for dirty_sample_id, dirty_sample_data in dirty_data.items():
            all_labels.append('%s_dirty' % dirty_sample_data['class'])
            all_samples.append(dirty_sample_data['feature'])
