import ast
import csv
import numpy as np
import os

def csv_to_dict(file, numeric=True):
    """
    read CSV file and create dict with key for each head containing a list of column entries for respective head
    :param file: path to CSV file to read
    :param numeric: expect numeric values
    :return: created dictionary
    """
    d = {}
    with open(file) as fin:
        reader = csv.reader(fin)
        headers = next(reader, None)
        for h in headers:
            d[h.lower()] = []
        for row in reader:
            for h, entry in zip(headers,row):
                if numeric:
                    if entry[0] == '[':
                        # remove chains of whitespaces for replacement
                        entry = '[' + entry[1:-1].strip() + ']'
                        entry = ' '.join(entry.split())
                        l = ast.literal_eval(entry.replace(' ', ','))
                        l = [float(e) for e in l]
                        d[h.lower()].append(l)
                    elif entry == "False" or entry == "True":
                        d[h.lower()].append(bool(entry))
                    else:
                        d[h.lower()].append(float(entry))
                else:
                    d[h.lower()].append(entry)
    return d

def extract_value(d, name, dtype='float'):
    """
    extract values from dict
    :param d: dict of CSV file of dataset
    :return: values (numpy array)
    """
    values = np.array(d[name], dtype=dtype)
    return values

def collect_values(dicts, strings):
    """
    Collect values of dicts that end with strings
    :param dicts: list of dicts to collect from
    :param strings: list of strings of key endings to collect from
    :return: list of numpy arrays of values
    """
    values = []
    for d in dicts:
        d_values = [[] for _ in strings]
        for k in d:
            for i, s in enumerate(strings):
                if k.endswith(s):
                    d_values[i].append(extract_value(d, k))
        if len(strings) > 1:
            values.append(np.asarray([np.array(d_vals) for d_vals in d_values]))
        else:
            values.append(np.asarray(d_values[0]))
    return values

def combine_arrays(arrays, transpose=False):
    """
    Combine list of numpy arrays
    :param arrays: list of N numpy arrays of (dim,)
    :return: concatenated numpy array (N, dim)
    """
    if transpose:
        arrays = [arr.T for arr in arrays]
    arrays = [np.expand_dims(arr, 0) for arr in arrays]
    return np.concatenate(arrays, axis=0)
