import os
import numpy as np
import laspy
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm


class LASDataset(Dataset):
    def __init__(self, split='train', data_root='data/las/train', num_point=4096, block_size=1.0, sample_rate=1.0,
                 transform=None):
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform
        las_files = sorted([f for f in os.listdir(data_root) if f.endswith('.las')])

        train_files, val_files = train_test_split(las_files, test_size=0.2, random_state=42)
        if split == 'train':
            self.files = train_files
        else:
            self.files = val_files

        self.room_points, self.room_labels = [], []
        self.room_coord_min, self.room_coord_max = [], []
        num_point_all = []
        labelweights = np.zeros(2)  # 只有两类

        for las_file in tqdm(self.files, total=len(self.files)):
            las_path = os.path.join(data_root, las_file)
            las_data = laspy.read(las_path)
            points = np.vstack((las_data.X, las_data.Y, las_data.Z, las_data.red, las_data.green,
                                las_data.blue)).transpose()  # 假设XYZRGB格式
            labels = np.array(las_data.classification)  # 转换 SubFieldView 为 NumPy 数组
            tmp, _ = np.histogram(labels, bins=2, range=(0, 2))
            labelweights += tmp
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.room_points.append(points)
            self.room_labels.append(labels)
            self.room_coord_min.append(coord_min)
            self.room_coord_max.append(coord_max)
            num_point_all.append(labels.size)

        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)

        sample_prob = num_point_all / np.sum(num_point_all)
        num_iter = int(np.sum(num_point_all) * sample_rate / num_point)
        room_idxs = []
        for index in range(len(self.files)):
            room_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))
        self.room_idxs = np.array(room_idxs)
        print("Totally {} samples in {} set.".format(len(self.room_idxs), split))

    def __getitem__(self, idx):
        room_idx = self.room_idxs[idx]
        points = self.room_points[room_idx]  # N * 6
        labels = self.room_labels[room_idx]  # N
        N_points = points.shape[0]

        while True:
            center = points[np.random.choice(N_points)][:3]
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) &
                                  (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
            if point_idxs.size > 1024:
                break

        # 添加剩余点的逻辑
        if point_idxs.size < self.num_point:
            remaining_idxs = np.setdiff1d(np.arange(N_points), point_idxs)
            extra_idxs = np.random.choice(remaining_idxs, self.num_point - point_idxs.size, replace=True)
            selected_point_idxs = np.concatenate((point_idxs, extra_idxs))
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)

        # normalize
        selected_points = points[selected_point_idxs, :]  # num_point * 6
        current_points = np.zeros((self.num_point, 9))  # num_point * 9
        current_points[:, 6] = selected_points[:, 0] / self.room_coord_max[room_idx][0]
        current_points[:, 7] = selected_points[:, 1] / self.room_coord_max[room_idx][1]
        current_points[:, 8] = selected_points[:, 2] / self.room_coord_max[room_idx][2]
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        selected_points[:, 3:6] /= 255.0
        current_points[:, 0:6] = selected_points
        current_labels = labels[selected_point_idxs]
        if self.transform is not None:
            current_points, current_labels = self.transform(current_points, current_labels)
        return current_points, current_labels

    def __len__(self):
        return len(self.room_idxs)


if __name__ == '__main__':
    data_root = 'data/las/train'
    num_point, block_size, sample_rate = 4096, 1.0, 1.0

    train_data = LASDataset(split='train', data_root=data_root, num_point=num_point, block_size=block_size,
                            sample_rate=sample_rate, transform=None)
    val_data = LASDataset(split='val', data_root=data_root, num_point=num_point, block_size=block_size,
                          sample_rate=sample_rate, transform=None)
    print('Train data size:', len(train_data))
    print('Validation data size:', len(val_data))

    import torch, time, random

    manual_seed = 123
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)


    def worker_init_fn(worker_id):
        random.seed(manual_seed + worker_id)


    train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True, num_workers=16, pin_memory=True,
                                               worker_init_fn=worker_init_fn)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=16, shuffle=False, num_workers=16, pin_memory=True,
                                             worker_init_fn=worker_init_fn)

    for idx in range(4):
        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            print('Train time: {}/{}--{}'.format(i + 1, len(train_loader), time.time() - end))
            end = time.time()

        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            print('Val time: {}/{}--{}'.format(i + 1, len(val_loader), time.time() - end))
            end = time.time()























































# import os
# import numpy as np
# import laspy
# from sklearn.model_selection import train_test_split
# from torch.utils.data import Dataset
# from tqdm import tqdm
#
#
# class LASDataset(Dataset):
#     def __init__(self, split='train', data_root='data/las/train', num_point=4096, block_size=1.0, sample_rate=1.0,
#                  transform=None):
#         super().__init__()
#         self.num_point = num_point
#         self.block_size = block_size
#         self.transform = transform
#         las_files = sorted([f for f in os.listdir(data_root) if f.endswith('.las')])
#
#         train_files, val_files = train_test_split(las_files, test_size=0.2, random_state=42)
#         if split == 'train':
#             self.files = train_files
#         else:
#             self.files = val_files
#
#         self.room_points, self.room_labels = [], []
#         self.room_coord_min, self.room_coord_max = [], []
#         num_point_all = []
#         labelweights = np.zeros(2)  # 只有两类
#
#         for las_file in tqdm(self.files, total=len(self.files)):
#             las_path = os.path.join(data_root, las_file)
#             las_data = laspy.read(las_path)
#             points = np.vstack((las_data.X, las_data.Y, las_data.Z, las_data.red, las_data.green,
#                                 las_data.blue)).transpose()  # 假设XYZRGB格式
#             labels = np.array(las_data.classification)  # 假设classification字段为标签
#             print(f"Type of labels: {type(labels)}")
#             print(f"Size of labels: {labels.size}")
#             tmp, _ = np.histogram(labels, bins=2, range=(0, 2))
#             labelweights += tmp
#             coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
#             self.room_points.append(points)
#             self.room_labels.append(labels)
#             self.room_coord_min.append(coord_min)
#             self.room_coord_max.append(coord_max)
#             num_point_all.append(labels.size)
#
#         labelweights = labelweights.astype(np.float32)
#         labelweights = labelweights / np.sum(labelweights)
#         self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
#
#         sample_prob = num_point_all / np.sum(num_point_all)
#         num_iter = int(np.sum(num_point_all) * sample_rate / num_point)
#         room_idxs = []
#         for index in range(len(self.files)):
#             room_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))
#         self.room_idxs = np.array(room_idxs)
#         print("Totally {} samples in {} set.".format(len(self.room_idxs), split))
#
#     def __getitem__(self, idx):
#         room_idx = self.room_idxs[idx]
#         points = self.room_points[room_idx]  # N * 6
#         labels = self.room_labels[room_idx]  # N
#         N_points = points.shape[0]
#
#         while True:
#             center = points[np.random.choice(N_points)][:3]
#             block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
#             block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
#             point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) &
#                                   (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
#             if point_idxs.size > 1024:
#                 break
#
#         if point_idxs.size >= self.num_point:
#             selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
#         else:
#             selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)
#
#         # normalize
#         selected_points = points[selected_point_idxs, :]  # num_point * 6
#         current_points = np.zeros((self.num_point, 9))  # num_point * 9
#         current_points[:, 6] = selected_points[:, 0] / self.room_coord_max[room_idx][0]
#         current_points[:, 7] = selected_points[:, 1] / self.room_coord_max[room_idx][1]
#         current_points[:, 8] = selected_points[:, 2] / self.room_coord_max[room_idx][2]
#         selected_points[:, 0] = selected_points[:, 0] - center[0]
#         selected_points[:, 1] = selected_points[:, 1] - center[1]
#         selected_points[:, 3:6] /= 255.0
#         current_points[:, 0:6] = selected_points
#         current_labels = labels[selected_point_idxs]
#         if self.transform is not None:
#             current_points, current_labels = self.transform(current_points, current_labels)
#         return current_points, current_labels
#
#     def __len__(self):
#         return len(self.room_idxs)
#
#
# if __name__ == '__main__':
#     data_root = 'data/las/train'
#     num_point, block_size, sample_rate = 4096, 1.0, 1.0
#
#     train_data = LASDataset(split='train', data_root=data_root, num_point=num_point, block_size=block_size,
#                             sample_rate=sample_rate, transform=None)
#     val_data = LASDataset(split='val', data_root=data_root, num_point=num_point, block_size=block_size,
#                           sample_rate=sample_rate, transform=None)
#     print('Train data size:', len(train_data))
#     print('Validation data size:', len(val_data))
#
#     import torch, time, random
#
#     manual_seed = 123
#     random.seed(manual_seed)
#     np.random.seed(manual_seed)
#     torch.manual_seed(manual_seed)
#     torch.cuda.manual_seed_all(manual_seed)
#
#
#     def worker_init_fn(worker_id):
#         random.seed(manual_seed + worker_id)
#
#
#     train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True, num_workers=16, pin_memory=True,
#                                                worker_init_fn=worker_init_fn)
#     val_loader = torch.utils.data.DataLoader(val_data, batch_size=16, shuffle=False, num_workers=16, pin_memory=True,
#                                              worker_init_fn=worker_init_fn)
#
#     for idx in range(4):
#         end = time.time()
#         for i, (input, target) in enumerate(train_loader):
#             print('Train time: {}/{}--{}'.format(i + 1, len(train_loader), time.time() - end))
#             end = time.time()
#
#         end = time.time()
#         for i, (input, target) in enumerate(val_loader):
#             print('Val time: {}/{}--{}'.format(i + 1, len(val_loader), time.time() - end))
#             end = time.time()

