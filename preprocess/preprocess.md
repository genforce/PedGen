### Downloading YouTube videos
Assume all data is under directory `$DATA_ROOT`

Download videos and images with the following script:

```
bash download_video.sh youtube_ids.txt $DATA_ROOT
```
,where  `preprocess/youtube_ids.txt` contains all youtube video ids used to train PedGen.
Note that some videos are no longer publicly available as the channel owner has deleted them.
The script will also generate the depth and segmentation maps of the images to get the scene context label.

### Generating pseudo labels

After downloading data, we use [WHAM](https://wham.is.tue.mpg.de/) to extract pseudo-labels of the pedestrian movements. Our custom code for running WHAM is at `third_party/WHAM` and we recommend installing WHAM in a separate python environment following their [official instructions](https://github.com/yohanshin/WHAM/blob/main/docs/INSTALL.md). Next, run the follows under the WHAM environment:
```
bash run_wham.sh $DATA_ROOT
```
After data preprocessing, the `$DATA_ROOT` directory should contain the following folders `image video depth semantic wham` as well as the label files `train.pkl val.pkl`.
 