import os
from shutil import copyfile
from utils import maybe_create_path

models = ['adaboost', 'bayes', 'rdf', 'svm']
marker_mode = ['joint', 'separate']
feature_mode = ['all_feature', 'prime_feature']
class_mode = ['biclass', 'multiclass']

src_items = ['conf_mat.png']
dst_items = ['figures']
src_path = 'exp'
dst_path = 'results'

if __name__ == '__main__':
    for dst_i in dst_items:
        maybe_create_path(os.path.join(dst_path, dst_i))

    for model in models:
        for marker in marker_mode:
            for feature in feature_mode:
                for class_ in class_mode:
                    exp_folder = '%s_%s_%s' % (marker, feature, class_)
                    for src_i, dst_i in zip(src_items, dst_items):
                        copyfile(os.path.join(src_path, model, exp_folder, src_i),
                                 os.path.join(dst_path, dst_i, model + '_' + exp_folder + '.' + src_i.split('.')[-1]))
