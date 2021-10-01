import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


def getData(config):
    sys.path.insert(0, '../')
    from MileStone_1_Training_Validation_data_Setup.load_data import load_data_from_ImageFolder
    SOURCE_DIR = "../Dataset_Willys_2020/ORGINAL/"
    test_path = config.get('data_path', SOURCE_DIR)
    print(test_path)
    dataset = load_data_from_ImageFolder(
        root=test_path
    )

    valid_procentage = config.get('valid_procentage', 0.15)
    test_procentage = config.get('test_procentage', 0.1)
    batch_size = config.get('batch_size', 400)

    n_jobs = config.get('num_workers', 32)

    num_train = len(dataset)
    val_indices = list(range(num_train))
    valid_split = int(valid_procentage*num_train)

    valid_idx = np.random.choice(val_indices, size=valid_split, replace=False)
    train_idx = list(set(val_indices)-set(valid_idx))

    test_split = int(test_procentage*len(train_idx))
    test_indices = list(range(len(train_idx)))
    test_idx = np.random.choice(test_indices, size=test_split, replace=False)

    train_idx = list(set(test_indices) - set(test_idx))

    print(
        f"total number of images {num_train}, which {len(train_idx)} of them is used for training")
    print(f"And {len(valid_idx)} for validation and {len(test_idx)} for testing")

    assert len(train_idx) + len(valid_idx) + len(test_idx) == num_train != 0

    train_sampler = SubsetRandomSampler(train_idx)
    train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=batch_size,
                              num_workers=n_jobs, pin_memory=True, drop_last=True)
    # print(
    #     f"train sampler length {len(train_sampler)}, train loader length {len(train_loader)}")
    assert len(train_sampler) != 0
    assert len(train_loader) != 0

    valid_sampler = SubsetRandomSampler(valid_idx)
    valid_loader = DataLoader(dataset, sampler=valid_sampler, batch_size=batch_size,
                              num_workers=n_jobs, pin_memory=True, drop_last=True)
    # print(
    # f"valid sampler length {len(valid_sampler)}, train loader length {len(valid_loader)}")
    assert len(valid_sampler) != 0
    assert len(valid_loader) != 0

    test_sampler = SubsetRandomSampler(test_idx)
    test_loader = DataLoader(dataset, sampler=test_sampler, batch_size=batch_size,
                             num_workers=n_jobs, pin_memory=True, drop_last=True
                             )
    # print(
    #     f"test sampler length {len(test_sampler)}, train loader length {len(test_loader)}")
    assert len(test_sampler) != 0
    assert len(test_loader) != 0

    config["num_iters_per_epoch"] = len(train_loader)

    # assert len(valid_loader) != 0
    # assert len(test_loader) != 0
    return (train_loader, train_sampler), (valid_loader, valid_sampler), (test_loader, test_sampler)
