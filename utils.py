import csv
import os

import numpy as np
import yaml


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


def get_bounds(feature, method='min_max'):
    assert method in ['min_max', 'box']

    if method == 'min_max':
        lower_bound = min(feature)
        upper_bound = max(feature)
    elif method == 'box':
        l, u = np.percentile(feature, (25, 75), interpolation='midpoint')
        lower_bound = l - 1.5 * (u - l)
        upper_bound = u + 1.5 * (u - l)
    else:
        return
    return lower_bound, upper_bound
