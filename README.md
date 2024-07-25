# SAM Enhanced masks

### **SORT Tracker**

First of all you need to build an image:
```
docker build -t sam-image .
```

#### 1. Evaluate / Training mode

You need to run the container and mount all of the necessary directories. Example how to launch inference:
```
sudo docker run --gpus all --shm-size=16g -v ./video/:/projects_data -v ./video_1:/input -v ./output:/output -it --rm sam-image --input_data /input/deeplab.json --demo_mode --clipes_mode
```



Apart from default keys: input_data (json file in input directory in format of tracking output file for inference
and inference output file for training mode) and --work_format_training (flag that marks training mode)
there also --demo_mode flag for inference that allows to save output bounder boxes and masks images in output
directory. In current version of code every 10th frame is processed (if bounder boxes are correct for such frames).

Also 2 modes are supported in the container: 1) default mode takes as an input output file in format of output from https://github.com/olegapa/deeplab/tree/master
and enhances masks in ```markup_vector['mask']``` . 2) clip-es mode can be activated by additional flag --clipes_mode, in this mode input is like in https://github.com/olegapa/deeplab/tree/master , but optional instead of required ```markup_vector['mask']``` parameters.
Instaed it generates masks using ```markup_vector['mask']``` parameter - which stands for a list of clothes labels presented in bounder box

Format output and input file:
```json
{
	files: [
		{
			file_name: <files.name>,
			file_chains: [
				{
					chain_name: <chains.name>,
					chain_vector: <chains.vector>,
					chain_markups: [
						{
							markup_parent_id: <markups.parent_id **>,
							markup_frame: <пересчет в markups.mark_time ***>,
							markup_path: <markups.mark_path>,
							markup_vector: <markups.vector>
						},
						... <список примитивов>
					]
				},
				... <список цепочек>
			]
		},
		... <список файлов>
	]
}
```
где markup_path:

```json
{
  x: <rect_x>,
  y: <rect_y>,
  width: <rect_width>,
  height: <rect_height>,
  mask: <base_64_encoded_mask_image>,
  "labels": <detcted_clothes_classes>
}
```
List of clothes classes presented can be obtained here https://github.com/olegapa/deeplab/tree/master.

#### 2. Pretrained weights
In order to use this solution 2 pretrained models should be loaded:
1. https://drive.google.com/file/d/1-4I8ig_akX0E7m5pYf1ckCGp6iNhgN4G/view?usp=sharing Place it in <code_root>/CLIP-ES/pretrained_models/clip/ folder
2. https://drive.google.com/file/d/1QkYmBN85tXUK4FDXrPjm6wPpm-hHMI7T/view?usp=sharing Place it in <code_root>/segment-anything/sam_checkpoint/ folder
Pretrained weights can be downloaded: https://drive.google.com/file/d/18N3ZRyCcno1cLnV4GGHNOGfRMLYolTeI/view?usp=sharing
    
