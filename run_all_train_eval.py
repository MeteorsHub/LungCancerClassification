import os

models = ['voting']
marker_mode = ['joint']
feature_mode = ['prime_feature']
class_mode = ['biclass']

if __name__ == '__main__':
    for model in models:
        for marker in marker_mode:
            for feature in feature_mode:
                for class_ in class_mode:
                    config_file = '%s/%s_%s_%s' % (model, marker, feature, class_)
                    command_str = 'python train_eval.py -c=%s -o -r' % config_file
                    os.system(command_str)
