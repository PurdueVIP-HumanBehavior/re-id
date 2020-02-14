## Disclaimer
 - Please read what the purdue [HELPS page](https://purduehelps.org/labmachine.html) has to say about this first.

## Login to HELPS machine
 - login to the computer with the following command: ```ssh <your purdue login>@ee220clnx1.ecn.purdue.edu```. Say yes to the security warning if it is your first time. Use your mypurdue password to login. 

## Activating the conda environment
 - run the following
```conda activate /local/b/mrasheed/HB_S20/```
your prompt should look something like this now probably
```(/local/b/mrasheed/HB_S20) $ ```

 - you can now run the reid-pipeline.py thing with just ```(/local/b/mrasheed/HB_S20) $ python run_reid_pipeline.py```

## But where is the data?
 - all our data is stored locally on that computer at this path: ```/local/b/cam2/data/HB/S20```
 - I have the weights for MGN stored locally there as well here: ```/local/b/mrasheed/model_weights/MGN.pt```
 - also, when you run it for the first time, the FasterRCNN libraries will download its weights onto your shay account. It's not that big but might put you over the edge if you're already near the 5GB limit on shay

## How do I move files between?
 - use scp command
 - if not familiar with it, can get an app such as WinSCP for windows or idk for mac lol. figure it out? let me know what you find.
