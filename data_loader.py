import itertools
import os
import random

from scipy.special import comb
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from utils import read_dict_csv
from visualization import plot_feature_distribution


class MarkerExpressionDataset:
    def __init__(self, config):
        self.data_root = config['data_root']
        self.marker_mapping = config['marker_mapping']
        self.features = config['features']
        self.data_clean = config.get('data_clean', None)
        self.num_split_fold_single = int(config['num_split_fold'])
        self.num_split_fold_repeat_times = int(config.get('num_repeat_split_fold', '1'))
        self.num_fold = self.num_split_fold_single * self.num_split_fold_repeat_times
        self.random_seed = float(config['split_random_seed'])
        self.class_mapping = config['class_mapping']
        self.default_missing_value = config['default_missing_value']
        self.num_samples_in_each_bag = int(config.get('num_samples_in_each_bag', 1))
        self.num_bags_limitation = int(config.get('num_bags_limitation', 10000))

        # one bag for the mode contains multiple samples with the same class in original data
        self.sample_data = dict()  # {marker: {id: {name, class, feature}}.
        self.outlier_samples = dict()  # {marker: {id: {name, class, feature}}}
        self.sample_data_split_fold = dict()  # {marker: [fold_0: {train: [train_ids], test: [test_ids]}, fold_1]}
        # {marker: [fold_0: {train: [[bag_0_sample_ids], [bag_1_sample_ids], ...], test}, fold_1, ...]}
        self.bag_data_split_fold = dict()
        self.k_fold_splitter = None
        self.classes = []
        self.markers = []

        if not os.path.exists(self.data_root):
            raise FileNotFoundError('cannot locate data root dir %s' % self.data_root)
        for marker in self.marker_mapping.keys():
            if not os.path.exists(os.path.join(self.data_root, marker + '.csv')):
                raise FileNotFoundError('cannot find %s' % os.path.join(self.data_root, marker + '.csv'))
        if self.num_split_fold_single > 10 or self.num_split_fold_single < 2:
            raise ValueError('num_split_fold must be within 2 and 10')
        assert 0 < self.random_seed < 1, 'random_seed must be 0-1'
        if self.default_missing_value is not None:
            self.default_missing_value = float(self.default_missing_value)
        class_mapping = dict()
        for k, v in self.class_mapping.items():
            class_mapping[int(k)] = int(v)
        self.class_mapping = class_mapping
        self.k_fold_splitter = RepeatedStratifiedKFold(
            n_splits=self.num_split_fold_single,
            n_repeats=self.num_split_fold_repeat_times,
            random_state=int(100*self.random_seed))
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
            sample_ids = list(self.sample_data[marker].keys())
            random.shuffle(sample_ids)
            sample_classes = [self.sample_data[marker][sample_id]['class'] for sample_id in sample_ids]

            samples_all_fold = []
            for train_indexes, test_indexes in self.k_fold_splitter.split(sample_ids, sample_classes):
                sample_ids_train = [sample_ids[train_index] for train_index in train_indexes]
                sample_ids_test = [sample_ids[test_index] for test_index in test_indexes]
                samples_in_fold = {'train': sample_ids_train, 'test': sample_ids_test}
                samples_all_fold.append(samples_in_fold)
            self.sample_data_split_fold[marker] = samples_all_fold

        # generate bags from samples
        for marker in self.markers:
            self.bag_data_split_fold[marker] = []
            for fold_i in range(self.num_fold):
                fold_train_sample_ids_per_class = dict()
                fold_test_sample_ids_per_class = dict()
                for class_i in self.classes:
                    fold_train_sample_ids_per_class[class_i] = []
                    fold_test_sample_ids_per_class[class_i] = []
                sample_data_split = self.sample_data_split_fold[marker][fold_i]
                for train_sample_id in sample_data_split['train']:
                    sample_class = self.sample_data[marker][train_sample_id]['class']
                    fold_train_sample_ids_per_class[sample_class].append(train_sample_id)
                for test_sample_id in sample_data_split['test']:
                    sample_class = self.sample_data[marker][test_sample_id]['class']
                    fold_test_sample_ids_per_class[sample_class].append(test_sample_id)

                fold_train_bags = []
                fold_test_bags = []
                for class_i in self.classes:
                    if len(fold_test_sample_ids_per_class[class_i]) < self.num_samples_in_each_bag:
                        raise ValueError('num_samples_in_each_bag (%d) is less than '
                                         'num_test_samples (%d) for class %s in marker %s'
                                         % (self.num_samples_in_each_bag, len(fold_test_sample_ids_per_class[class_i]),
                                            class_i, marker))
                    if comb(len(fold_train_sample_ids_per_class[class_i]), self.num_samples_in_each_bag) \
                            <= self.num_bags_limitation:
                        print('marker %s class %s train bags use combination.' % (marker, class_i))
                        for bag in itertools.combinations(fold_train_sample_ids_per_class[class_i],
                                                          self.num_samples_in_each_bag):
                            fold_train_bags.append(bag)
                    else:
                        print('marker %s class %s train bags use random select.' % (marker, class_i))
                        for _ in range(self.num_bags_limitation):
                            fold_train_bags.append(random.sample(fold_train_sample_ids_per_class[class_i],
                                                                 k=self.num_samples_in_each_bag))
                    if comb(len(fold_test_sample_ids_per_class[class_i]), self.num_samples_in_each_bag) \
                            <= self.num_bags_limitation:
                        print('marker %s class %s test bags use combination.' % (marker, class_i))
                        for bag in itertools.combinations(fold_test_sample_ids_per_class[class_i],
                                                          self.num_samples_in_each_bag):
                            fold_test_bags.append(bag)
                    else:
                        print('marker %s class %s test bags use random select.' % (marker, class_i))
                        for _ in range(self.num_bags_limitation):
                            fold_test_bags.append(random.sample(fold_test_sample_ids_per_class[class_i],
                                                                 k=self.num_samples_in_each_bag))
                random.shuffle(fold_train_bags)
                random.shuffle(fold_test_bags)
                self.bag_data_split_fold[marker].append({'train': fold_train_bags, 'test': fold_test_bags})

    def get_bag_feature_class(self, marker, sample_ids):
        bag_class = None
        bag_feature = []
        for sample_id in sample_ids:
            item = self.sample_data[marker][sample_id]
            if bag_class is None:
                bag_class = item['class']
            elif item['class'] != bag_class:
                raise ValueError('class id dis-match in marker %s' % marker)
            bag_feature += item['feature']
        return bag_feature, bag_class

    def get_all_data(self, marker):
        assert marker in self.markers, 'marker %s not in self.markers' % marker

        all_bag_feature = []
        all_bag_class = []
        index_pointer = 0
        cv_index = []  # [(fold_0_train_indexes, fold_0_test_indexes)...]

        for fold_i in range(self.num_fold):
            train_bags = self.bag_data_split_fold[marker][fold_i]['train']
            test_bags = self.bag_data_split_fold[marker][fold_i]['test']
            train_indexes = []
            test_indexes = []
            for train_bag in train_bags:
                bag_feature, bag_class = self.get_bag_feature_class(marker, train_bag)
                all_bag_feature.append(bag_feature)
                all_bag_class.append(bag_class)
                train_indexes.append(index_pointer)
                index_pointer += 1
            for test_bag in test_bags:
                bag_feature, bag_class = self.get_bag_feature_class(marker, test_bag)
                all_bag_feature.append(bag_feature)
                all_bag_class.append(bag_class)
                test_indexes.append(index_pointer)
                index_pointer += 1
            cv_index.append((train_indexes, test_indexes))

        return all_bag_feature, all_bag_class, cv_index

    def get_split_data(self, marker):
        assert marker in self.markers, 'marker %s not in self.markers' % marker

        for fold_i in range(self.num_fold):
            train_feature = []
            train_class = []
            test_feature = []
            test_class = []

            train_bags = self.bag_data_split_fold[marker][fold_i]['train']
            test_bags = self.bag_data_split_fold[marker][fold_i]['test']
            for train_bag in train_bags:
                bag_feature, bag_class = self.get_bag_feature_class(marker, train_bag)
                train_feature.append(bag_feature)
                train_class.append(bag_class)
            for test_bag in test_bags:
                bag_feature, bag_class = self.get_bag_feature_class(marker, test_bag)
                test_feature.append(bag_feature)
                test_class.append(bag_class)
            yield train_feature, train_class, test_feature, test_class

    def plot_data_clean_distribution(self, ax, marker):
        clean_data = self.sample_data[marker]
        if self.data_clean is not None:
            dirty_data = self.outlier_samples[marker]

        all_samples = []
        all_hues = []
        all_styles = []
        for clean_sample_id, clean_sample_data in clean_data.items():
            all_hues.append(str(clean_sample_data['class']))
            all_styles.append('Y')
            all_samples.append(clean_sample_data['feature'])
        if self.data_clean is not None:
            for dirty_sample_id, dirty_sample_data in dirty_data.items():
                all_hues.append(str(dirty_sample_data['class']))
                all_styles.append('N')
                all_samples.append(dirty_sample_data['feature'])
        class_str = [str(item) for item in self.classes]
        plot_feature_distribution(all_samples, ax, t_sne=True,
                                  hue=all_hues, hue_order=class_str,
                                  style=all_styles, style_order=['Y', 'N'],
                                  x_lim='min_max_extend', y_lim='min_max_extend')
