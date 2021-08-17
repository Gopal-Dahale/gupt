import torch


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, transform, target_transform):
        super().__init__()
        if len(data) != len(targets):
            print("Length of Targets must match with length of Data")
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        """All subclasses should overwrite __getitem__(), supporting fetching a data sample for a given key.

        Args:
            idx (int): index of the data point
        """
        data_point = self.data[idx]
        label = self.targets[idx]

        # If transform is available then return transformed dataPoint and label
        if self.transform is not None:
            data_point = self.transform(data_point)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return data_point, label

    def __len__(self):
        """expected to return the size of the dataset
        """
        return len(self.data)
