import nibabel as nib
import os
import sys
import time
import numpy as np
import logging
from brainiak.fcma.util import compute_correlation
import brainiak.fcma.io as io

format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# if want to output log to a file instead of outputting log to the console,
# replace "stream=sys.stdout" with "filename='fcma.log'"
logging.basicConfig(level=logging.INFO, format=format, stream=sys.stdout)
logger = logging.getLogger(__name__)

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

# python graph_generator.py face_scene bet.nii.gz face_scene/mask.nii.gz face_scene/fs_epoch_labels.npy 0.8
if __name__ == '__main__':
    data_dir = sys.argv[1]
    extension = sys.argv[2]
    mask_file = sys.argv[3]
    epoch_file = sys.argv[4]
    thres = float(sys.argv[5])

    raw_data = io.read_activity_data(data_dir, extension, mask_file)

    epoch_list = np.load(epoch_file)
    epoch_info = io.generate_epochs_info(epoch_list)

    generate_graph(raw_data, epoch_info, thres)
