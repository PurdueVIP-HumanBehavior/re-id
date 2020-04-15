## run_reid_pipeline.py setup
final.py is the script used to generate the output video from last semester.
To reproduce it, you need to do 5 things:
1. use the reid-repo.yml file in the repo to create a conda environment capable of running the code. [here for reference](https://gist.github.com/pratos/e167d4b002f5d888d0726a5b5ddcca57)
   - if all of the packages don't get downloaded, try removing all the gibberish after the version number?
2. download the raw video data from the drive, keep them in on directory (for example, "msee2_data")
   - can be found on drive: Human Behavior Spring 2020/Data/MSEE
3. change directory path in line 17 to the one with video data made in step 2, global variable datapath
4. triggering needs a reference image
   - download 00000.jpg from drive folder (in root directory)
   - change line 18 to path of that image, wherever you've saved it
5. download the MGN weights from the drive
   - Human Behavior Spring 2020/Weights/MGN.pt
   - **rename it to model.pt and save in same directory as final.py**

## run run_reid_pipeline.py
Script usage is the following

```python
usage: run_reid_pipeline.py [-h] [-d {FasterRCNN}] [-r {dot_product}]
                            [-l {video}] [-g {trigger}] [-v {MGN}]
                            [-i INTERVAL] --video_path VIDEO_PATH
                            --ref_image_path REF_IMAGE_PATH --weights_path
                            WEIGHTS_PATH [--gallery_path GALLERY_PATH]
```

Example:
```python
python3 run_reid_pipeline.py --video_path=data/ --ref_image_path=00000.jpg --weights_path=<path to mgn weights> --interval=2 --gallery_path=tmpgal/
```
