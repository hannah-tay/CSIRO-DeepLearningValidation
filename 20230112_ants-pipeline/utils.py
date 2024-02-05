import monai
from typing import Any, Callable, Generic, Mapping, TypeVar


T = TypeVar("T")
def identity(x: Generic[T]) -> T:
    return x
DictTransform = Callable[[Mapping[str, Any]], Mapping[str, Any]]


class PairedDataset(monai.data.CSVDataset):
    """Modified from Mirorrnet source code:
    https://bitbucket.csiro.au/projects/MIRORRNET/repos/mirorrnet/browse/src/mirorrnet/datasets/generic.py

    A Dataset builds from an input `orig_dataset` that returns all possible 2-combinations
    of values for all keys.
    
    E.g., orig_dataset: [{"k": 0}, {"k": 1}]
    -> [{"k": (0, 0)}, {"k": (0, 1)}, {"k": (1, 0)}, {"k": (1, 1)}]
    """
    def __init__(
        self,
        orig_dataset: monai.data.CSVDataset,
        transform: DictTransform = identity,
    ):
        self.orig_dataset = orig_dataset
        self.transform = transform

    def __len__(self):
        return len(self.orig_dataset) ** 2

    def __getitem__(self, idx):
        orig_len = len(self.orig_dataset)
        idx_1 = idx // orig_len
        idx_2 = idx % orig_len
        ret_1 = {f"{k}_1": v for k, v in self.orig_dataset[idx_1].items()}
        ret_2 = {f"{k}_2": v for k, v in self.orig_dataset[idx_2].items()}
        return self.transform({**ret_1, **ret_2})