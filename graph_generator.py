import nibabel as nib
import os
import sys
import time
import numpy as np
import logging
from brainiak.fcma.util import compute_correlation

format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# if want to output log to a file instead of outputting log to the console,
# replace "stream=sys.stdout" with "filename='fcma.log'"
logging.basicConfig(level=logging.INFO, format=format, stream=sys.stdout)
logger = logging.getLogger(__name__)

def read_activity_data(dir, file_extension, mask_file):
    """ read data in NIfTI format and apply the spatial mask to them

    Parameters
    ----------
    dir: str
        the path to all subject files
    file_extension: str
        the file extension, usually nii.gz or nii
    mask_file: str
        the absolute path of the mask file, we apply the mask right after
        reading a file for saving memory

    Returns
    -------
    activity_data:  list of 2D array in shape [nVoxels, nTRs]
        the masked activity data organized in voxel*TR formats
        len(activity_data) equals the number of subjects
    """
    time1 = time.time()
    mask_img = nib.load(mask_file)
    mask = mask_img.get_data().astype(np.bool)
    count = 0
    for index in np.ndindex(mask.shape):
        if mask[index] != 0:
            count += 1
    files = [f for f in sorted(os.listdir(dir))
             if os.path.isfile(os.path.join(dir, f))
             and f.endswith(file_extension)]
    activity_data = []
    for f in files:
        img = nib.load(os.path.join(dir, f))
        data = img.get_data()[mask, :]
        activity_data.append(data)
        logger.info(
            'file %s is loaded and masked, with data shape %s' %
            (f, data.shape)
        )
    time2 = time.time()
    logger.info(
        'data reading done, takes %.2f s' %
        (time2 - time1)
    )
    return activity_data

def generate_epochs_info(epoch_list):
    """ use epoch_list to generate epoch_info defined below

    Parameters
    ----------
    epoch\_list: list of 3D (binary) array in shape [condition, nEpochs, nTRs]
        Contains specification of epochs and conditions,
        Assumption: 1. all subjects have the same number of epochs;
                     2. len(epoch_list) equals the number of subjects;
                     3. an epoch is always a continuous time course.

    Returns
    -------
    epoch\_info: list of tuple (label, sid, start, end).
        label is the condition labels of the epochs;
        sid is the subject id, corresponding to the index of raw_data;
        start is the start TR of an epoch (inclusive);
        end is the end TR of an epoch(exclusive).
        Assuming len(labels) labels equals the number of epochs and
        the epochs of the same sid are adjacent in epoch_info
    """
    time1 = time.time()
    epoch_info = []
    for sid, epoch in enumerate(epoch_list):
        for cond in range(epoch.shape[0]):
            sub_epoch = epoch[cond, :, :]
            for eid in range(epoch.shape[1]):
                r = np.sum(sub_epoch[eid, :])
                if r > 0:   # there is an epoch in this condition
                    start = np.nonzero(sub_epoch[eid, :])[0][0]
                    epoch_info.append((cond, sid, start, start+r))
    time2 = time.time()
    logger.info(
        'epoch separation done, takes %.2f s' %
        (time2 - time1)
    )
    return epoch_info

def generate_graph(raw_data, epoch_info, thres):
    dir = './results'
    if not os.path.exists(dir):
        os.makedirs(dir)
    for idx, epoch in enumerate(epoch_info):
        label = epoch[0]
        sid = epoch[1]
        start = epoch[2]
        end = epoch[3]
        mat = raw_data[sid][:, start:end]
        mat = np.ascontiguousarray(mat, dtype=np.float32)
        logger.info(
            'start to compute correlation for subject %d epoch %d with label %d' %
            (sid, idx, label)
        )
        corr = compute_correlation(mat, mat)
        logger.info(
            'start to construct the graph by ruling out correlations less than %.2f' %
            thres
        )
        indices = np.where(corr>thres)
        filename = str(label) + '_' + str(sid) + '_' + str(idx) + '.txt'
        logger.info(
            'in total %d pairs, writing to %s' %
            ((len(indices[0])-mat.shape[0])/2, os.path.join(dir, filename))
        )
        fp = open(os.path.join(dir, filename), 'w')
        for i in range(len(indices[0])):
                if indices[0][i] < indices[1][i]:
                    fp.write(str(indices[0][i]) + ' ' + str(indices[1][i]) + '\n')
        fp.close()

# python graph_generator.py face_scene bet.nii.gz face_scene/mask.nii.gz face_scene/fs_epoch_labels.npy graph 0.8
if __name__ == '__main__':
    data_dir = sys.argv[1]
    extension = sys.argv[2]
    mask_file = sys.argv[3]
    epoch_file = sys.argv[4]
    thres = float(sys.argv[5])

    raw_data =read_activity_data(data_dir, extension, mask_file)

    epoch_list = np.load(epoch_file)
    epoch_info = generate_epochs_info(epoch_list)

    generate_graph(raw_data, epoch_info, thres)
