from torch.utils.data import Dataset

from taming.data.base import ImagePaths, ImagePathsHolistic


class CustomBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example


class CustomTrain(CustomBase):
    def __init__(self, size, training_images_list_file, coord=False, random_crop=False, scaler=None, holistic=None):
        super().__init__()
        with open(training_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, random_crop=random_crop, coord=coord, rescale=scaler, holistic=holistic)


class CustomTest(CustomBase):
    def __init__(self, size, test_images_list_file, coord=False, random_crop=False, scaler=None, holistic=None):
        super().__init__()
        with open(test_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, random_crop=random_crop, coord=coord, rescale=scaler, holistic=holistic)


class CustomTrainHolistic(CustomBase):
    def __init__(self, size, training_images_list_file, coord=False, random_crop=False, scaler=None, holistic=None, clip_emb=None):
        super().__init__()
        with open(training_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePathsHolistic(paths=paths, size=size, random_crop=random_crop, coord=coord, rescale=scaler, holistic=holistic, clip_emb=clip_emb)


class CustomTestHolistic(CustomBase):
    def __init__(self, size, test_images_list_file, coord=False, random_crop=False, scaler=None, holistic=None, clip_emb=None):
        super().__init__()
        with open(test_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePathsHolistic(paths=paths, size=size, random_crop=random_crop, coord=coord, rescale=scaler, holistic=holistic, clip_emb=clip_emb)