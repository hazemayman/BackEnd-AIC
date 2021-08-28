import os
import glob
import argparse
import shutil
import sys
# sys.path.append('src/utils/')
from utils  import sentinel_roi_generation
from utils  import read_merged_shape_file
from utils import sen2_cropping
import predict
from utils import merge_results_one_tile

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
	print("starting")
	try:

		temp_dir = os.path.join(args.temp_directory, 'temp')
		os.makedirs(temp_dir, exist_ok=True)

		output_path = os.path.join(temp_dir, "before/")
		os.makedirs(output_path, exist_ok=True)
		print("roi generation")
		sentinel_roi_generation.main(args.bands_dir, args.shape_file_path, output_path)

		if args.shape_file_path is not None:
			output_path_name = os.path.join(output_path, "merged_lc.tiff")
			read_merged_shape_file.main(args.bands_dir, args.shape_file_path, output_path_name)
		else:
			output_path_name = None
		print('Cropping')
		output_crop = os.path.join(temp_dir, "after/")
		os.makedirs(output_crop, exist_ok=True)
		sen2_cropping.cut_sen2_files(output_path, output_path_name, output_crop , 'region')

		output_results = os.path.join(temp_dir, "results/")
		os.makedirs(output_results, exist_ok=True)

		if args.shape_file_path is None:
			dataset = 'tiff_dir'
			score = False
			gt_id = "pred"
		else:
			dataset = 'dfc2020_val'
			score = True
			gt_id = "dfc"
		print("Predicting")
		predict.main(args.config, args.checkpoint_file, output_crop, output_results, dataset=dataset, score=score)

		lc_file_name = os.path.join(output_path, "*s2.tiff")
		lc_file_name = glob.glob(lc_file_name)[0]
		os.makedirs(args.results_path, exist_ok=True)
		merge_results_one_tile.merge_sen2_files(output_results, lc_file_name, args.results_path, target_size=256, gt_id=gt_id)


	finally:
	         #pass
		shutil.rmtree(temp_dir)


if __name__ == '__main__':
	main()

