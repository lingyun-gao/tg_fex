import requests


MODULES_SPECS_URL = 'https://s3.amazonaws.com/tg-training-data/lyst_external/modules_specs.json'
CANONICAL_LABELS_URL = 'https://s3.amazonaws.com/tg-training-data/lyst_external/canonical_labels.txt'


def get_modules_specs():
    r = requests.get(MODULES_SPECS_URL)
    return r.json()


def get_num2names():
    r = requests.get(CANONICAL_LABELS_URL)
    lines = r.text.strip().split('\n')

    num2names = []
    for line in lines:
        num, name = line.strip().split(' ', 1)
        num2names.append((int(num), name))
    return num2names


def get_maps():
    modules_specs = get_modules_specs()
    num2names = get_num2names()

    # initialize maps
    num2module = {}
    num2name = {}
    num2idx = {mod_name: {} for mod_name in modules_specs}
    idx2num = {mod_name: {} for mod_name in modules_specs}
    idx2name = {mod_name: {} for mod_name in modules_specs}

    idx = {mod_name: 0 for mod_name in modules_specs}
    for num, name in num2names:
        for mod_name, mod_set in modules_specs.items():
            if num in mod_set:
                num2module[num] = mod_name
                num2name[num] = name
                num2idx[mod_name][num] = idx[mod_name]
                idx2num[mod_name][idx[mod_name]] = num
                idx2name[mod_name][idx[mod_name]] = name
                idx[mod_name] += 1

    return {
        'num2module': num2module,
        'num2name': num2name,
        'num2idx': num2idx,
        'idx2num': idx2num,
        'idx2name': idx2name,
    }


maps = get_maps()


class LystExternal:
    num2module = maps['num2module']
    num2name = maps['num2name']
    num2idx = maps['num2idx']  # per module
    idx2num = maps['idx2num']  # per module
    idx2name = maps['idx2name']  # per module
    num_classes = len(maps['idx2name'])
    modules_specs = []
    modules_weights = []
    for mod_name, mod_specs in num2idx.items():
        modules_specs.append((mod_name, len(mod_specs)))
        modules_weights.append((mod_name, len(mod_specs) / len(num2module)))
