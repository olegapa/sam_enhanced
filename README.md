# SAM Enhanced masks

### **SAM Enhanced Pseudo Labels**

First of all you need to build an image:
```
docker build -t sam-image .
```

#### 1. Evaluate / Training mode

You need to run the container and mount all the necessary directories. Example how to launch inference:
```
sudo docker run --gpus all --shm-size=16g -v family:/family -v ./projects_data/:/projects_data -v ./new_videos:/input_videos  -v ./new_sam_output/sam1:/output -v /var/run/docker.sock:/var/run/docker.sock -v ./new_sam_input/test:/input_data -it --rm sam-image --host_web 'http://127.0.0.1:5555' --input_data '{"frame_frequency": "100"}'
```
Additional parameters are passed to SAM container via `--input_data` dictionary-based parameter
#### Additional flags:

`--demo_mode` - Execute in demo mode. All temporal files like images will be saved in output directory. Colored masks are saved as well.

`--min_height` and `--min_width` - Restriction on bounder box minimum size. Only bounder boxes of bigger size are processed

#### Contents of `input_data` dictionary parameter:
`frame_frequency` - Filters frames to be processed by frame numbers. E.g. if `frame_frequency = 10` then only frames with numbers 1, 11, 21... are processed

`visualize` - If true then videos with processed masks are generated. *In current version of implementation this option is untested*

`approx_eps` - Determines polygon approximation scale. The more value is the fewer polygons are in output. Default value 0.02

`score_thresholds` - Determines confidence score thresholds for each of the segmentation classes, thus enabling to process only objects with higher confidence score then the threshold. 

*Example*:
`sudo docker run --gpus all --shm-size=16g -v ./projects_data/:/projects_data -v /family/projects_data/4e0e4e28-0f0e-11f0-b6ef-0242ac140002/4e129866-0f0e-11f0-b6ef-0242ac140002/videos:/input_videos -v /family:/family -v ./new_sam_output/big_data/2:/output -v /var/run/docker.sock:/var/run/docker.sock -v ./new_sam_input/big_data/2:/input_data -it --rm sam-image --host_web 'http://127.0.0.1:5555' --input_data '{"score_thresholds": "{1: 0.43, 2: 0.64, 3: 0.7, 4: 0.8, 5: 0.5, 6: 0.42, 7: 0.61, 8: 0.48}"}' --demo_mode`

#### Format for input/output file:
SAM container takes as an input output file in format of output from https://github.com/olegapa/deeplab/tree/master 
output and input file:
```json
{
	files: [
		{
			file_id: <files.id>,		
			file_name: <files.name>,
			file_chains: [
				{
					chain_name: <chains.name>,
					chain_vector: <chains.vector>,
					chain_markups: [
						{
							markup_parent_id: <markups.parent_id **>,
							markup_frame: <пересчет в markups.mark_time ***>,
							markup_time: <markups.mark_time>,
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
  polygons: <mask_polygons>,
  class: <mask_class>
}
```
List of clothes classes presented can be obtained here https://github.com/olegapa/deeplab/tree/master.

#### 2. Pretrained weights
In order to use this solution 2 pretrained models should be loaded:
1. https://drive.google.com/file/d/1-4I8ig_akX0E7m5pYf1ckCGp6iNhgN4G/view?usp=sharing Place it in <code_root>/CLIP-ES/pretrained_models/clip/ folder
2. https://drive.google.com/file/d/1QkYmBN85tXUK4FDXrPjm6wPpm-hHMI7T/view?usp=sharing Place it in <code_root>/segment-anything/sam_checkpoint/ folder

    
