import os
import numpy as np
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
# import data_prep_util
import
import glob
import h5py

# Constants
# indoor3d_data_dir = os.path.join(data_dir, 'mydata_h5')
NUM_POINT = 4096
H5_BATCH_SIZE = 1000
data_dim = [NUM_POINT, 6]
label_dim = [NUM_POINT]
data_dtype = 'float32'
label_dtype = 'uint8'

# Set paths
# filelist = os.path.join(BASE_DIR, 'meta/all_data_label.txt')
# data_label_files = [os.path.join(indoor3d_data_dir, line.rstrip()) for line in open(filelist)]

output_dir = os.path.join(ROOT_DIR, 'data/mydata_h5')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
output_filename_prefix = os.path.join(output_dir, 'ply_data_all')
output_room_filelist = os.path.join(output_dir, 'all_files.txt')
fout_room = open(output_room_filelist, 'w')

# --------------------------------------
# ----- BATCH WRITE TO HDF5 -----
# --------------------------------------
batch_data_dim = [H5_BATCH_SIZE] + data_dim
batch_label_dim = [H5_BATCH_SIZE] + label_dim
h5_batch_data = np.zeros(batch_data_dim, dtype=np.float32)
h5_batch_label = np.zeros(batch_label_dim, dtype=np.uint8)
buffer_size = 0  # state: record how many samples are currently in buffer
h5_index = 0  # state: the next h5 file to save


def save_h5(h5_filename, data, label, data_dtype='uint8', label_dtype='uint8'):
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset(
        'data', data=data,
        compression='gzip', compression_opts=4,
        dtype=data_dtype)
    h5_fout.create_dataset(
        'label', data=label,
        compression='gzip', compression_opts=1,
        dtype=label_dtype)
    h5_fout.close()


def insert_batch(data, label, last_batch=False):
    global h5_batch_data, h5_batch_label
    global buffer_size, h5_index
    data_size = data.shape[0]
    # If there is enough space, just insert
    if buffer_size + data_size <= h5_batch_data.shape[0]:
        h5_batch_data[buffer_size:buffer_size + data_size, ...] = data
        h5_batch_label[buffer_size:buffer_size + data_size] = label
        buffer_size += data_size
    else:  # not enough space
        capacity = h5_batch_data.shape[0] - buffer_size
        assert (capacity >= 0)
        if capacity > 0:
            h5_batch_data[buffer_size:buffer_size + capacity, ...] = data[0:capacity, ...]
            h5_batch_label[buffer_size:buffer_size + capacity, ...] = label[0:capacity, ...]
            # Save batch data and label to h5 file, reset buffer_size
        h5_filename = output_filename_prefix + '_' + str(h5_index) + '.h5'
        save_h5(h5_filename, h5_batch_data, h5_batch_label, data_dtype, label_dtype)
        fout_room.write('mydata_h5' + '\'' + h5_filename)
        print('Stored {0} with size {1}'.format(h5_filename, h5_batch_data.shape[0]))
        h5_index += 1
        buffer_size = 0
        # recursive call
        insert_batch(data[capacity:, ...], label[capacity:, ...], last_batch)
    if last_batch and buffer_size > 0:
        h5_filename = output_filename_prefix + '_' + str(h5_index) + '.h5'
        save_h5(h5_filename, h5_batch_data[0:buffer_size, ...], h5_batch_label[0:buffer_size, ...], data_dtype,
                label_dtype)
        fout_room.write('mydata_h5/ply_data_all' + '_' + str(h5_index) + '.h5')
        print('Stored {0} with size {1}'.format(h5_filename, buffer_size))
        h5_index += 1
        buffer_size = 0
    return


path = os.path.join(BASE_DIR + '/mydata_withlabel', '*.asc')
files = glob.glob(path)
points_list = []

for f in files:
    print(f)
    points = np.loadtxt(f)
    print(points.shape)
    sample = np.random.choice(points.shape[0], NUM_POINT)
    sample_data = points[sample, ...]
    print(sample_data.shape)
    points_list.append(sample_data)

data_label = np.stack(points_list, axis=0)
print(data_label.shape)

data = data_label[:, :, 0:6]
label = data_label[:, :, 6]

print(data.shape)
print(label.shape)
sample_cnt = 0

insert_batch(data, label, True)

# for i, data_label_filename in enumerate(data_label_files):
#     print(data_label_filename)
#     data, label = indoor3d_util.room2blocks_wrapper_normalized(data_label_filename, NUM_POINT, block_size=1.0, stride=0.5,
#                                                  random_sample=False, sample_num=None)
#     print('{0}, {1}'.format(data.shape, label.shape))
#     for _ in range(data.shape[0]):
#         fout_room.write(os.path.basename(data_label_filename)[0:-4]+'\n')
#
#     sample_cnt += data.shape[0]
#     insert_batch(data, label, i == len(data_label_files)-1)
#
fout_room.close()
# print("Total samples: {0}".format(sample_cnt))












