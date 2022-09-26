import os
import faiss
import numpy as np
import albumentations
from PIL import Image
from torch.utils.data import Dataset


class ImagePaths(Dataset):
    def __init__(self, paths, size=None, random_crop=False, labels=None, coord=False, rescale=None, holistic=None):
        self.size = size
        self.rescale = rescale
        self.random_crop = random_crop
        self.coord = coord
        self.holistic = holistic

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)

        if self.holistic is not None:
            assert isinstance(self.holistic, int)
            assert self.holistic <= self.size
            self.holistic_rescaler = albumentations.SmallestMaxSize(max_size=self.holistic, interpolation=3)

        if self.size is not None and self.size > 0:
            if self.rescale is not None:
                assert isinstance(self.rescale, int)
                self.rescaler = albumentations.SmallestMaxSize(max_size = self.size, interpolation=3)
            else:
                self.rescaler = None
            if not self.random_crop:
                self.cropper = albumentations.CenterCrop(height=self.size,width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size,width=self.size)
            if self.coord:
                if self.rescaler is not None:
                    self.preprocessor = albumentations.Compose([self.rescaler, self.cropper], additional_targets={"coord": "image"})
                else:
                    self.preprocessor = albumentations.Compose([self.cropper], additional_targets={"coord": "image"})
            else:
                if self.rescaler is not None:
                    self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
                else:
                    self.preprocessor = albumentations.Compose([self.cropper])
        else:
            self.preprocessor = lambda **kwargs: kwargs

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        ret = {}
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        if self.holistic is not None:
            holistic_img = self.holistic_rescaler(image=image)["image"]
            holistic_img = (holistic_img/127.5 - 1.0).astype(np.float32)
            ret['holistic'] = holistic_img
        if self.coord:
            # pre-process later in __getitem__
            pass
        else:
            image = self.preprocessor(image=image)["image"]
            image = (image / 127.5 - 1.0).astype(np.float32)
        ret['image'] = image
        return ret

    def __getitem__(self, i):
        example = dict()
        ret = self.preprocess_image(self.labels["file_path_"][i])
        example["image"] = ret['image']
        holistic = ret.get('holistic')
        if holistic is not None:
            example['holistic'] = holistic

        if self.coord:
            height, width, _ = example["image"].shape
            # generate spherical coordinates
            xx, yy = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
            screen_points = np.stack([xx, yy], axis=-1)
            coord = (screen_points * 2 - 1) * np.array([np.pi, np.pi/2])

            out = self.preprocessor(image=example["image"], coord=coord)
            example["image"] = (out["image"] / 127.5 -1).astype(np.float32)
            example["coord"] = out["coord"]
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example

class ImagePathsHolistic(ImagePaths):
    def __init__(self, paths, size=None, random_crop=False, labels=None, coord=False, rescale=None, holistic=None, clip_emb=None):
        super().__init__(paths, size, random_crop, labels, coord, rescale, holistic)
        self.clip_emb = None
        if clip_emb is not None:
            if os.path.isfile(clip_emb):
                self.clip_emb = np.load(clip_emb).astype('float32')
            elif isinstance(clip_emb, np.array):
                assert clip_emb.shape[-1] == 512
                self.clip_emb = clip_emb
            else:
                raise NotImplementedError("This type of CLIP embeddings [{}] is not supported.".format(clip_emb))
            self.index = faiss.IndexFlatIP(self.clip_emb.shape[-1])
            self.index.add(self.clip_emb)
            self.k = 5
            # assuming the order in train.txt and clip embedding of certain instance is the same

    def __getitem__(self, i):
        example = dict()
        image_path = self.labels["file_path_"][i]
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.holistic_rescaler(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        example["image"] = image

        if self.clip_emb is not None:
            emb = self.clip_emb[i, :]
            noise = np.random.normal(0., 1., (512)).astype('float32')
            img_fts = emb
            alpha = 0.25
            revised_img_fts = alpha * img_fts/np.linalg.norm(img_fts) + (1-alpha)*noise/np.linalg.norm(noise)
            revised_img_fts = revised_img_fts/np.linalg.norm(revised_img_fts)
            _, idx = self.index.search(revised_img_fts[None, :], self.k)
            example['knn'] = self.clip_emb[idx][0] #[k, 512]
            example['psed_emb'] = img_fts
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example

    def get_knn(self, txt_emb):
        if self.clip_emb is not None:
            dist, idx = self.index.search(txt_emb, self.k)
            return self.clip_emb[idx], idx #[bs, k, 512]
        else:
            return None