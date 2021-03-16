import argparse
import os

from sklearn import base

from data_loader import MarkerExpressionDataset
from utils import load_ymal, get_model, eval_results

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, required=False, default='default',
                    help='config name. default: \'default\'')
args = parser.parse_args()


def feature_analysis(config):
    dataset = MarkerExpressionDataset(config)

    model = get_model(config)
    for marker in dataset.markers:
        all_features, all_labels, _ = dataset.get_all_data(marker, feature_selection=False, dup_reduce=True)
        model = base.clone(model)
        all_pred_labels = [0 for _ in range(len(all_labels))]
        all_pred_score = [0 for _ in range(len(all_labels))]
        for i in range(len(all_features)):
            train_features = all_features.copy()
            train_labels = all_labels.copy()
            del train_features[i]
            del train_labels[i]
            # model.fit(all_features, all_labels)
            model.fit(train_features, train_labels)
            all_pred_score[i] = model.predict_proba([all_features[i]])[0]
            all_pred_labels[i] = model.predict([all_features[i]])[0]
        tps = sum([y_true == y_pred for y_true, y_pred in zip(all_labels, all_pred_labels)])
        acc = tps / len(all_features)
        results = eval_results(all_labels, all_pred_score, dataset.classes)
        print('marker %s: acc %1.2f' % (marker, 100 * acc))
        print(results)


if __name__ == '__main__':
    config = load_ymal(os.path.join('config', args.config + '.yaml'))
    feature_analysis(config)
