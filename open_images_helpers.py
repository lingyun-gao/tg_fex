import requests

LABELS_URL = 'https://s3.amazonaws.com/tg-open-images/canonical_oi_labels.txt'

open_images_mod_name = 'open_images'


def get_idxs_wnids_names():
    r = requests.get(LABELS_URL)
    lines = r.text.strip().split('\n')

    idxs_wnids_names = []
    for idx, line in enumerate(lines):
        wnid, name = line.strip().split(',', 1)
        idxs_wnids_names.append((idx, wnid, name))
    return idxs_wnids_names


def get_maps():
    idxs_wnids_names = get_idxs_wnids_names()

    # initialize maps
    wnid2name = {}
    wnid2idx = {}
    idx2wnid = {}
    idx2name = {}

    for idx, wnid, name in idxs_wnids_names:
        wnid2name[wnid] = name
        wnid2idx[wnid] = idx
        idx2wnid[idx] = wnid
        idx2name[idx] = name

    return {
        'wnid2name': wnid2name,
        'wnid2idx': wnid2idx,
        'idx2wnid': idx2wnid,
        'idx2name': idx2name,
    }


maps = get_maps()


class OpenImages:
    wnid2name = maps['wnid2name']
    wnid2idx = maps['wnid2idx']
    idx2wnid = maps['idx2wnid']
    idx2name = maps['idx2name']
    num_classes = len(wnid2name)
    modules_specs = [(open_images_mod_name, num_classes)]
