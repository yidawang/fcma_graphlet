# fcma_graphlet

Applying graphlet technique to analyze brain network obtained from FCMA

## generate graphs

Step 1: download sample data

```sh
curl --location -o face_scene.zip https://www.dropbox.com/s/11adrjdkt0w1tr3/face_scene.zip?dl=0
unzip -qo face_scene.zip
rm -f face_scene.zip
```

Step 2: generate graph by applying some threshold `thres`

```sh
python graph_generator.py face_scene bet.nii.gz face_scene/mask.nii.gz face_scene/fs_epoch_labels.npy 0.8
```
