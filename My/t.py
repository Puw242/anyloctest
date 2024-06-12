import sys
sys.path.append('/Users/puw/Workspace/VPR_test/seqNet')
from datasets import Dataset
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from os.path import join
from itertools import product


prefix_data = "./data/"


def get_dataset(opt):

    print("Here")
    dataset = Dataset(
        "nordland",
        "nordland_train_d-40_d2-10.db",
        "nordland_test_d-1_d2-1.db",
        "nordland_val_d-1_d2-1.db",
    )  # train, test, val structs
    if "sw" == opt:
        ref, qry = "summer", "winter"
    elif "sf" == opt:
        ref, qry = "spring", "fall"

    skip_rate = 1

    # ft1,ft2 are the data files
    ft1 = np.load(
        join(prefix_data, "descData/{}/nordland-clean-{}.npy".format('netvlad-pytorch', ref))
    )
    ft2 = np.load(
        join(prefix_data, "descData/{}/nordland-clean-{}.npy".format('netvlad-pytorch', qry))
    )

    trainInds = np.arange(0, 15000, skip_rate)
    testInds = np.arange(15100, 18100, skip_rate)
    valInds = np.arange(18200, 21200, skip_rate)

    # modify dataset indicies based on skip_rate
    dataset.trainInds = [trainInds, trainInds]
    dataset.valInds = [valInds, valInds]
    dataset.testInds = [testInds, testInds]
    encoder_dim = dataset.loadPreComputedDescriptors(ft1, ft2)
    return dataset, encoder_dim

get_dataset('sw')