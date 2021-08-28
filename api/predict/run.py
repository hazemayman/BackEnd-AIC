import os
import glob
import argparse
import shutil
import sys
# sys.path.append('src/utils/')
from .utils  import sentinel_roi_generation
from .utils  import read_merged_shape_file
from .utils import sen2_cropping
import predict.predict as predict
from .utils import merge_results_one_tile
import requests

def predict_tile(config,checkpoint_file,bands_dir,shape_file_path,temp_directory,results_path):

	print("starting")
	try:

		temp_dir = os.path.join(temp_directory, 'temp')
		os.makedirs(temp_dir, exist_ok=True)

		output_path = os.path.join(temp_dir, "before/")
		os.makedirs(output_path, exist_ok=True)
		print("roi generation")
		sentinel_roi_generation.main(bands_dir, shape_file_path, output_path)

		if shape_file_path is not None:
			output_path_name = os.path.join(output_path, "merged_lc.tiff")
			read_merged_shape_file.main(bands_dir, shape_file_path, output_path_name)
		else:
			output_path_name = None
		print('Cropping')
		output_crop = os.path.join(temp_dir, "after/")
		os.makedirs(output_crop, exist_ok=True)
		sen2_cropping.cut_sen2_files(output_path, output_path_name, output_crop , 'region')

		output_results = os.path.join(temp_dir, "results/")
		os.makedirs(output_results, exist_ok=True)

		if shape_file_path is None:
			dataset = 'tiff_dir'
			score = False
			gt_id = "pred"
		else:
			dataset = 'dfc2020_val'
			score = True
			gt_id = "dfc"
		print("Predicting")
		predict.main(config, checkpoint_file, output_crop, output_results, dataset=dataset, score=score)

		lc_file_name = os.path.join(output_path, "*s2.tiff")
		print("output file : " , lc_file_name)
		lc_file_name = glob.glob(lc_file_name)[0]
		print("output file2 : " , lc_file_name)
		print("output results :",output_results )
		os.makedirs(results_path, exist_ok=True)
		merge_results_one_tile.merge_sen2_files(output_results, lc_file_name, results_path, target_size=256, gt_id=gt_id)


	finally:
	         #pass
		shutil.rmtree(temp_dir)
	
	# path to the results
	return lc_file_name 


def predict_and_call_backend(config,checkpoint_file,bands_dir,shape_file_path,temp_directory,results_path):
	print("Predicting...")
	
	output_path = predict_tile(config,checkpoint_file,\
									bands_dir,\
									shape_file_path,temp_directory\
									,results_path)

	# sending request to the server to save the output file
	path_split = output_path.split('/')
	print(path_split)
	filepath = '/'.join(path_split[:-1])[1:]
	filename = path_split[-1]

	url = 'http://localhost:5000/savedata'
	data={
		'file_path':os.path.join(results_path , "result__L2A_T36RTV_A020396_20210131T085044_2021-01-31.tif"),
		'file_name': "result__L2A_T36RTV_A020396_20210131T085044_2021-01-31.tif"
	}
	print("---------------------------",'\n',data,'\n','----------------------')
	res = requests.post(url, data=data)
	print(res.status_code)




def main():
	parser = argparse.ArgumentParser()

	parser.add_argument('--config', type=str, required=True,
                        help='path to config file')
	parser.add_argument('--checkpoint_file', type=str, required=True,
                        help='path to checkpoint file')

	parser.add_argument('--bands_dir', type=str, required=True,
                        help='directory to the bands')
	parser.add_argument('--shape_file_path', type=str, default=None,
                        help='path to the shape file (optional)')

	parser.add_argument('--temp_directory', type=str, default='./',
                        help='directory to put temporary files/directories')

	parser.add_argument('--results_path', type=str, required=True,
                        help='path to the results_file')

	args = parser.parse_args()
	predict_tile(args.config,args.checkpoint_file,args.bands_dir,args.shape_file_path,args.temp_directory,args.results_path)






if __name__ == '__main__':
	main()

