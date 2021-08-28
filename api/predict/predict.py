import os
import argparse
import numpy as np
import pickle as pkl
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from .datasets import SEN12MS, DFC2020, TIFFDir
from .data_utils import mycmap as dfc_cmap
from .data_utils import mypatches as dfc_legend
from torch import jit
from .dataset_type_utils import get_dataset_type_class

def main(config, checkpoint_file, data_dir, out_dir, dataset='dfc2020_val', batch_size=2, workers=4, score=False, dataset_type='sentinel_egypt', use_egypt_data=True, use_gpu=True):
    train_args = pkl.load(open(config, "rb"))

    # set flags for GPU processing if available
    if torch.cuda.is_available():
        use_gpu = True
    else:
        use_gpu = False

    # load dataset
    if dataset == "tiff_dir":
        assert not score
        dataset = TIFFDir(data_dir,
                          no_savanna=train_args.no_savanna,
                          use_s2hr=train_args.use_s2hr,
                          use_s2mr=train_args.use_s2mr,
                          use_s2lr=train_args.use_s2lr,
                          use_s1=train_args.use_s1,
                          use_rgb=train_args.use_rgb,
                          dataset_type=train_args.dataset_type)
        gt_id = "pred"
    else:
        dfc2020_subset = dataset.split("_")[-1]

        gt_id = "dfc"

        dataset = SEN12MS(data_dir,
                          subset="validation",
                          no_savanna=train_args.no_savanna,
                          use_s2hr=train_args.use_s2hr,
                          use_s2mr=train_args.use_s2mr,
                          use_s2lr=train_args.use_s2lr,
                          use_s1=train_args.use_s1,
                          use_rgb=train_args.use_rgb,
                          dataset_type=train_args.dataset_type,
                          use_egypt_data=use_egypt_data)

    n_classes = dataset.n_classes
    n_inputs = dataset.n_inputs

    # set up dataloader
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=workers,
                            pin_memory=True,
                            drop_last=False)


    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
        use_gpu = True
    else:
        map_location = 'cpu'
        use_gpu = False
    model = jit.load(checkpoint_file, map_location=map_location)
    print("loaded checkpoint")
    if use_gpu:
        model = model.cuda()
    model.eval()


    # initialize scoring if ground-truth is available
    if score:
        import predict.metrics as metrics
        conf_mat = metrics.ConfMatrix(n_classes)

    # predict samples
    n = 0

    plt.rcParams['toolbar'] = 'None'
    if score:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 4))
        ax3.set_title("label")
        ax3.axis("off")
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2)

    fig.canvas.set_window_title('image prediction')
    fig.set_figwidth(10)
    fig.set_figheight(7)
    ax1.set_title("input")
    ax1.axis("off")
    ax2.set_title("prediction")
    ax2.axis("off")
    lgd = plt.legend(loc='lower left', bbox_to_anchor=(1.1, 0),
                     handles=dfc_legend(dataset_type), ncol=2, title="Classes")

    plt.tight_layout()

    # f = plt.figure(figsize=(80, 60))
    # dataset_type
    # plt.close()


    for batch in tqdm(dataloader, desc="[Pred batch]"):

        # unpack sample
        image = batch['image']
        # print(image.shape)
        if score:
            target = batch['label']

        # move data to gpu if model is on gpu
        if use_gpu:
            image = image.cuda()
            if score:
                target = target.cuda()

        # forward pass
        with torch.no_grad():
            prediction = model(image)


        # convert to 256x256 numpy arrays
        prediction = prediction.cpu().numpy()
        prediction = np.argmax(prediction, axis=1)


        if score:
            target = target.cpu().numpy()

        # save predictions
        for i in tqdm(range(prediction.shape[0]), desc="[sample]"):

            n += 1
            id = batch["id"][i].replace("_s2_", "_" + gt_id + "_")

            # output = labels_to_dfc(prediction[i, :, :], train_args.no_savanna)
            output = prediction[i, :, :]
            output = output.astype(np.uint8)
            output_img = Image.fromarray(output)
            output_img.save(os.path.join(out_dir, id))

            # update error metrics
            if score:
                gt = target[i, :, :]
                # print(prediction[i].shape, target[i].shape)
                # gt = labels_to_dfc(target[i, :, :], train_args.no_savanna)
                conf_mat.add(target[i, :, :], prediction[i, :, :])

                # save preview
            
    if score:
        print("#####################################")

        print("Final Average Accuracy (AA) = \t", conf_mat.get_aa_micro())
        print("Final mean Intersection over Union (mIoU) = \t", conf_mat.get_mIoU_micro(), flush=True)

        aa_per_class = conf_mat.get_aa_per_class()
        mIoU_per_class = conf_mat.get_mIoU_per_class()
        class_names = get_dataset_type_class(dataset_type).class_names()[:8]
        for i in range(len(class_names)):
            class_ = class_names[i]
            print('Class: ', class_, ": AA = ", aa_per_class[i], ", mIoU = ", mIoU_per_class[i])

    plt.close()



if __name__ == '__main__':

    # define and parse arguments
    parser = argparse.ArgumentParser()

    # config
    parser.add_argument('--config', type=str, default="args.pkl",
                        help='path to config file (default: ./args.conf)')
    parser.add_argument('--checkpoint_file', type=str, default="checkpoint.pth",
                        help='path to checkpoint file (default: ./checkpoint.pth)')
    parser.add_argument('--config_file2', type=str, default="args.pkl",
                        help='path to config file (default: ./args.conf)')
    parser.add_argument('--checkpoint_file2', type=str, default="checkpoint.pth",
                        help='path to checkpoint file (default: ./checkpoint.pth)')

    # general
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size for prediction (default: 32)')
    parser.add_argument('--workers', type=int, default=4,
                        help='number of workers for dataloading (default: 4)')
    parser.add_argument('--score', action='store_true', default=False,
                        help='score prediction results using ground-truth data')

    # data
    parser.add_argument('--dataset', type=str, default="dfc2020_val",
                        choices=['dfc2020_val', 'tiff_dir'],
                        help='type of dataset (default: dfc2020_val)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='path to dataset')
    parser.add_argument('--out_dir', type=str, default="results",
                        help='path to output dir (default: ./results)')
    parser.add_argument('--preview_dir', type=str, default=None,
                        help='path to preview dir (default: no previews)')

    parser.add_argument('--dataset_type', type=str,
                        default='lc', choices=['lc', 'roads', 'roads_only', 'sentinel', 'sentinel_egypt'],
                        help='Deepglobe lc or roads')

    # add visualization option
    parser.add_argument('--visualize', action='store_true', default=False,
                        help='visualize the input image, ground truth, and the predicted image')

    parser.add_argument('--predict_mix', action='store_true', default=False,
                        help='predict multiple models')

    parser.add_argument('--use_egypt_data', action='store_true',
                            default=False,
                            help='Use egypt_data')

    parser.add_argument('--load_weights', action='store_true',
                            default=False,
                            help='Load weights not model')

    args = parser.parse_args()
    print("=" * 20, "PREDICTION CONFIG", "=" * 20)
    for arg in vars(args):
        print('{0:20}  {1}'.format(arg, getattr(args, arg)))
    print()

    # load config
    train_args = pkl.load(open(args.config, "rb"))
    print("=" * 20, "TRAIN CONFIG", "=" * 20)
    for arg in vars(train_args):
        print('{0:20}  {1}'.format(arg, getattr(train_args, arg)))
    print()

    # if args.use_egypt_data:
    #     train_args.dataset_type = 'sentinel_egypt'

    # create output dir
    os.makedirs(args.out_dir, exist_ok=True)

    # create preview dir
    if args.preview_dir is not None:
        os.makedirs(args.preview_dir, exist_ok=True)

    # set flags for GPU processing if available
    if torch.cuda.is_available():
        args.use_gpu = True
        if torch.cuda.device_count() > 1:
            raise NotImplementedError("multi-gpu prediction not implemented! "
                                      + "try to run script as: "
                                      + "CUDA_VISIBLE_DEVICES=0 predict.py")
    else:
        args.use_gpu = False

    # load dataset
    if args.dataset == "tiff_dir":
        assert not args.score
        dataset = TIFFDir(args.data_dir,
                          no_savanna=train_args.no_savanna,
                          use_s2hr=train_args.use_s2hr,
                          use_s2mr=train_args.use_s2mr,
                          use_s2lr=train_args.use_s2lr,
                          use_s1=train_args.use_s1,
                          use_rgb=train_args.use_rgb,
                          dataset_type=train_args.dataset_type)
        gt_id = "pred"
    else:
        dfc2020_subset = args.dataset.split("_")[-1]

        gt_id = "dfc"

        dataset = SEN12MS(args.data_dir,
                          subset="validation",
                          no_savanna=train_args.no_savanna,
                          use_s2hr=train_args.use_s2hr,
                          use_s2mr=train_args.use_s2mr,
                          use_s2lr=train_args.use_s2lr,
                          use_s1=train_args.use_s1,
                          use_rgb=train_args.use_rgb,
                          dataset_type=train_args.dataset_type,
                          use_egypt_data=args.use_egypt_data)

    n_classes = dataset.n_classes
    n_inputs = dataset.n_inputs

    # set up dataloader
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.workers,
                            pin_memory=True,
                            drop_last=False)
  
    
    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
        use_gpu = True
    else:
        map_location = 'cpu'
        use_gpu = False
    model = jit.load(checkpoint_file, map_location=map_location)
    print("loaded checkpoint")
    if use_gpu:
        model = model.cuda()
    model.eval()
    


    # initialize scoring if ground-truth is available
    if args.score:
        import metrics

        conf_mat = metrics.ConfMatrix(n_classes)

    # predict samples
    n = 0

    plt.rcParams['toolbar'] = 'None'
    if args.score:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 4))
        ax3.set_title("label")
        ax3.axis("off")
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2)

    fig.canvas.set_window_title('image prediction')
    fig.set_figwidth(10)
    fig.set_figheight(7)
    ax1.set_title("input")
    ax1.axis("off")
    ax2.set_title("prediction")
    ax2.axis("off")
    lgd = plt.legend(loc='lower left', bbox_to_anchor=(1.1, 0),
                     handles=dfc_legend(args.dataset_type), ncol=2, title="Classes")

    plt.tight_layout()

    # f = plt.figure(figsize=(80, 60))
    # dataset_type
    # plt.close()


    for batch in tqdm(dataloader, desc="[Pred batch]"):

        # unpack sample
        image = batch['image']
        # print(image.shape)
        if args.score:
            target = batch['label']

        # move data to gpu if model is on gpu
        if args.use_gpu:
            image = image.cuda()
            if args.score:
                target = target.cuda()

        # forward pass
        with torch.no_grad():
            prediction = model(image)


        # convert to 256x256 numpy arrays
        prediction = prediction.cpu().numpy()
        prediction = np.argmax(prediction, axis=1)


        if args.score:
            target = target.cpu().numpy()

        # save predictions
        for i in tqdm(range(prediction.shape[0]), desc="[sample]"):

            n += 1
            id = batch["id"][i].replace("_s2_", "_" + gt_id + "_")

            # output = labels_to_dfc(prediction[i, :, :], train_args.no_savanna)
            output = prediction[i, :, :]
            output = output.astype(np.uint8)
            output_img = Image.fromarray(output)
            output_img.save(os.path.join(args.out_dir, id))

            # update error metrics
            if args.score:
                gt = target[i, :, :]
                # gt = labels_to_dfc(target[i, :, :], train_args.no_savanna)
                conf_mat.add(target[i, :, :], prediction[i, :, :])

                # save preview
            if args.preview_dir is not None:

                    # colorize labels
                    cmap = dfc_cmap(args.dataset_type)
                    # output = (output - 1) / 10
                    output = cmap(output)[:, :, 0:3]
                    if args.score:
                        # gt = (gt - 1) / 10
                        gt = cmap(gt)[:, :, 0:3]
                    display_channels = dataset.display_channels
                    brightness_factor = dataset.brightness_factor

                    # if args.score:
                    #     fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
                    # else:
                    #     fig, (ax1, ax2) = plt.subplots(1, 2)
                    img = image.cpu().numpy()[i, display_channels, :, :]
                    img = np.rollaxis(img, 0, 3)
                    s1 = image.cpu().numpy()[i, -2:-1, :, :]
                    s1 = np.rollaxis(s1, 0, 3)
                    ax1.imshow(np.clip(img * brightness_factor, 0, 1))
                    ax2.imshow(output)
                    if args.score:
                        ax3.imshow(gt)
                    ttl = fig.suptitle(id, y=0.75)
                    plt.savefig(os.path.join(args.preview_dir, id),
                                bbox_extra_artists=(lgd, ttl,), bbox_inches='tight')

                    # add some delay to avoid plt display errors
            if args.visualize:
                plt.waitforbuttonpress(1)

                # if args.visualize:
                #     plt.waitforbuttonpress(1)

    if args.score:
        print("#####################################")

        print("Final Average Accuracy (AA) = \t", conf_mat.get_aa_micro())
        print("Final mean Intersection over Union (mIoU) = \t", conf_mat.get_mIoU_micro(), flush=True)

        aa_per_class = conf_mat.get_aa_per_class()
        mIoU_per_class = conf_mat.get_mIoU_per_class()
        class_names = get_dataset_type_class(args.dataset_type).class_names()[:8]
        for i in range(len(class_names)):
            class_ = class_names[i]
            print('Class: ', class_, ": AA = ", aa_per_class[i], ", mIoU = ", mIoU_per_class[i])

    plt.close()
