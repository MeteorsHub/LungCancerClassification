import csv
import yaml
import os


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
