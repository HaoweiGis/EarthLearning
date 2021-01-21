# """Pascal Gid Semantic Segmentation Dataset."""
# import os
# import torch
# import numpy as np

# # from PIL import Image
# from .segbase import SegmentationDataset


# class GidSegmentation(SegmentationDataset):
#     """Pascal Gid Semantic Segmentation Dataset.

#     Parameters
#     ----------
#     root : string
#         Path to Giddevkit folder. Default is './datasets/caigangfang'
#     split: string
#         'train', 'val' or 'test'
#     transform : callable, optional
#         A function that transforms the image
#     Examples
#     --------
#     >>> from torchvision import transforms
#     >>> import torch.utils.data as data
#     >>> # Transforms for Normalization
#     >>> input_transform = transforms.Compose([
#     >>>     transforms.ToTensor(),
#     >>>     transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
#     >>> ])
#     >>> # Create Dataset
#     >>> trainset = GidSegmentation(split='train', transform=input_transform)
#     >>> # Create Training Loader
#     >>> train_data = data.DataLoader(
#     >>>     trainset, 4, shuffle=True,
#     >>>     num_workers=4)
#     """
#     BASE_DIR = 'caigangfang'
#     NUM_CLASS = 2

#     def __init__(self, root=r'../datasets/', split='train', mode=None, transform=None, **kwargs):
#         super(GidSegmentation, self).__init__(root, split, mode, transform, **kwargs)
#         _voc_root = os.path.join(root, self.BASE_DIR)
#         _mask_dir = os.path.join(_voc_root, 'label')
#         print(_mask_dir)
#         _image_dir = os.path.join(_voc_root, 'image')
#         # train/val/test splits are pre-cut
#         _splits_dir = os.path.join(_voc_root, 'index')
#         if split == 'train':
#             _split_f = os.path.join(_splits_dir, 'train.txt')
#         elif split == 'val':
#             _split_f = os.path.join(_splits_dir, 'val.txt')
#         elif split == 'test':
#             _split_f = os.path.join(_splits_dir, 'val.txt')
#         else:
#             raise RuntimeError('Unknown dataset split.')

#         self.images = []
#         self.masks = []
#         with open(os.path.join(_split_f), "r") as lines:
#             for line in lines:
#                 _image = os.path.join(_image_dir, line.rstrip('\n') + ".png")
#                 assert os.path.isfile(_image)
#                 self.images.append(_image)
#                 if split != 'test':
#                     _mask = os.path.join(_mask_dir, line.rstrip('\n') + ".png")
#                     assert os.path.isfile(_mask)
#                     self.masks.append(_mask)

#         if split != 'test':
#             assert (len(self.images) == len(self.masks))
#         print('Found {} images in the folder {}'.format(len(self.images), _voc_root))

#     def __getitem__(self, index):
#         img = Image.open(self.images[index]).convert('RGB')
#         img = 
#         if self.mode == 'test':
#             img = self._img_transform(img)
#             if self.transform is not None:
#                 img = self.transform(img)
#             return img, os.path.basename(self.images[index])
#         mask = Image.open(self.masks[index])
#         # synchronized transform
#         if self.mode == 'train':
#             img, mask = self._sync_transform(img, mask)
#         elif self.mode == 'val':
#             img, mask = self._val_sync_transform(img, mask)
#         else:
#             assert self.mode == 'testval'
#             img, mask = self._img_transform(img), self._mask_transform(mask)
#         # general resize, normalize and toTensor
#         if self.transform is not None:
#             img = self.transform(img)

#         return img, mask, os.path.basename(self.images[index])

#     def __len__(self):
#         return len(self.images)

#     def _mask_transform(self, mask):
#         target = np.array(mask).astype('int32')
#         target[target == 255] = -1
#         return torch.from_numpy(target).long()

#     @property
#     def classes(self):
#         """Category names."""
#         return ('background', 'industrial','urban','rural','traffic','paddy','irrigated','dry','garden','arbor','shrub','natural','artificial','river','lake','pond')


# if __name__ == '__main__':
#     dataset = GidSegmentation()