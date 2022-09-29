import os
import pickle

import filelock
import torch
from torch.utils.data import Subset, random_split
from torch_geometric.data import DataLoader


def get_data_loaders(dataset, batch_size, random_split_flag=True, data_split_ratio=None, seed=5):

    if not random_split_flag and hasattr(dataset, 'supplement'):
        assert 'split_indices' in dataset.supplement.keys(), "split idx"
        split_indices = dataset.supplement['split_indices']
        train_indices = torch.where(split_indices == 0)[0].numpy().tolist()
        dev_indices = torch.where(split_indices == 1)[0].numpy().tolist()
        test_indices = torch.where(split_indices == 2)[0].numpy().tolist()

        train = Subset(dataset, train_indices)
        eval = Subset(dataset, dev_indices)
        test = Subset(dataset, test_indices)
    else:
        num_train = int(data_split_ratio[0] * len(dataset))
        num_eval = int(data_split_ratio[1] * len(dataset))
        num_test = len(dataset) - num_train - num_eval

        train, eval, test = random_split(dataset, lengths=[num_train, num_eval, num_test],
                                         generator=torch.Generator().manual_seed(seed))

    dataloader_train = DataLoader(train, batch_size=batch_size, shuffle=True)
    dataloader_proj = DataLoader(train, batch_size=256, shuffle=True)
    dataloader_eval = DataLoader(eval, batch_size=batch_size, shuffle=False)
    dataloader_test = DataLoader(test, batch_size=batch_size, shuffle=False)

    return dataloader_train, dataloader_eval, dataloader_test, dataloader_proj



def _get_data_split_from_tdc(task_name: str,
                             dataset_name: str,
                             assay_name: str,
                             split_method: str,
                             split_frac,
                             split_seed: int):
    import tdc.single_pred
    task = getattr(tdc.single_pred, task_name)
    data = task(name=dataset_name, label_name=assay_name)
    split = data.get_split(method=split_method, seed=split_seed, frac=split_frac)

    return {
        'train': {'IDs': split['train']['Drug_ID'].to_list(),
                  'X': split['train']['Drug'].to_list(),
                  'Y': split['train']['Y'].to_numpy()},
        'valid': {'IDs': split['valid']['Drug_ID'].to_list(),
                  'X': split['valid']['Drug'].to_list(),
                  'Y': split['valid']['Y'].to_numpy()},
        'test': {'IDs': split['test']['Drug_ID'].to_list(),
                 'X': split['test']['Drug'].to_list(),
                 'Y': split['test']['Y'].to_numpy()},
    }


def get_data_split(task_name: str,
                   dataset_name: str,
                   assay_name: str = None,
                   split_method: str = "random",
                   split_frac = (0.8, 0.1, 0.1),
                   split_seed = None) -> dict:


    return _get_data_split_from_tdc(task_name, dataset_name, assay_name,
                                         split_method, split_frac, split_seed)

def _encodings_cached(dataset_name) -> bool:
    return os.path.exists(_get_encodings_cache_filepath(dataset_name))

def _get_encodings_cache_filepath(dataset_name) -> str:
    filename = f'{dataset_name}.save'
    filepath = os.path.join("./dataset/", filename)
    return os.path.expanduser(filepath)

def _dump_encodings_to_cache(split, dataset_name) -> None:
    data = {}
    for target in split.keys():
        for smiles, x in zip(split[target]['S'], split[target]['X']):
            data[smiles] = x

    filepath = _get_encodings_cache_filepath(dataset_name)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    lock_path = filepath + '.lock'
    with filelock.FileLock(lock_path):
        with open(filepath, 'wb') as fp:
            pickle.dump(data, fp)



def _load_encodings_from_cache(split, split_name):
    filepath = _get_encodings_cache_filepath(split_name)
    lock_path = filepath + '.lock'
    with filelock.FileLock(lock_path):
        with open(filepath, 'rb') as fp:
            data_dict = pickle.load(fp)

    for target in split.keys():
        for i, smiles in enumerate(split[target]['S']):
            split[target]['X'][i] = data_dict[smiles]

    return split


def get_data_loaders_rmat(featurizer, *,
                     batch_size: int,
                     num_workers: int = 0,
                     cache_encodings: bool = False,
                     task_name: str = None,
                     dataset_name: str = None,
                     split_seed: int = None,
                     split_method:str = "random"):

    split_name = f"{dataset_name}"

    if task_name and dataset_name:
        split = get_data_split(task_name=task_name, dataset_name=dataset_name, split_method=split_method, split_seed=split_seed)
        split['train']['S'] = split['train']['X']
        split['valid']['S'] = split['valid']['X']
        split['test']['S'] = split['test']['X']

    else:
        raise AttributeError(
            'Both `task_name` and `dataset_name` attributes must be set either to None or to str values')

    if cache_encodings and _encodings_cached(split_name):
        split = _load_encodings_from_cache(split, split_name)
    else:
        split['train']['X'] = featurizer.encode_smiles_list(split['train']['X'], split['train']['Y'])
        split['valid']['X'] = featurizer.encode_smiles_list(split['valid']['X'], split['valid']['Y'])
        split['test']['X'] = featurizer.encode_smiles_list(split['test']['X'], split['test']['Y'])

    if cache_encodings and not _encodings_cached(split_name):
        _dump_encodings_to_cache(split, split_name)

    train_data = [x for x in split['train']['X'] if x.bond_features.shape[1] ]
    valid_data = [x for x in split['valid']['X'] if x.bond_features.shape[1]  ]
    test_data = [x for x in split['test']['X'] if x.bond_features.shape[1]]

    print(f'Train samples: {len(train_data)}')
    print(f'Validation samples: {len(valid_data)}')
    print(f'Test samples: {len(test_data)}')

    train_loader = featurizer.get_data_loader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    proj_loader = featurizer.get_data_loader(train_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    valid_loader = featurizer.get_data_loader(valid_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = featurizer.get_data_loader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader, test_loader, proj_loader

