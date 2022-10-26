import os
import cv2
import time
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
from skimage.io import imread

import models
from loss_functions import *
from config import cfg, cfg_from_file
from utils.flow_viz import flow_to_image
from user_input import user_input
from datasets import FlowDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EvalDataset(FlowDataset):
    def __init__(self, args, image_size=None, do_augument=False, root='datasets/Sintel/training', dstype='clean'):
        super(EvalDataset, self).__init__(args, image_size, do_augument)

        self.root = root
        self.dstype = dstype
        self.flow_list = []
        self.image_list = []
        filenames = sorted(os.listdir(root))
        for i in range(args.img_step, len(filenames)):
            im1 = os.path.join(root, filenames[i-args.img_step])
            im2 = os.path.join(root, filenames[i])
            self.image_list.append([im1, im2])
            self.flow_list.append(im1)


def evaluate_video_DICL(args):
    # Load config file
    if args.cfg is not None:
        cfg_from_file(args.cfg)
        assert cfg.TAG == os.path.splitext(os.path.basename(args.cfg))[0], 'TAG name should be file name'

    # Set random seed
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    test_dataset = EvalDataset(args, root=args.data, dstype='clean')
    val_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, shuffle=False)

    # create model
    if args.pretrained:
        pretrained_dict = torch.load(args.pretrained)
        if 'state_dict' in pretrained_dict.keys():
            pretrained_dict['state_dict'] =  {k:v for k,v in pretrained_dict['state_dict'].items()}

    model = models.__dict__['dicl_wrapper'](None)

    assert(args.solver in ['adam', 'sgd'])

    if args.pretrained:
        if 'state_dict' in pretrained_dict.keys():
            model.load_state_dict(pretrained_dict['state_dict'],strict=False)
        else:
            model.load_state_dict(pretrained_dict,strict=False)
        del pretrained_dict
        torch.cuda.empty_cache()

    model = torch.nn.DataParallel(model)

    if torch.cuda.is_available():
        model = model.cuda()

    # Evaluation
    if args.evaluate:
        with torch.no_grad():
            model.eval()
            out = cv2.VideoWriter(args.video_out_name, cv2.VideoWriter_fourcc(*'DIVX'), args.fps_out, args.frameSize)
            time_lst = []
            counter = 0
            for i, (input1, target, valid) in enumerate(val_loader):
                if i < args.img_start:
                    continue
                input1 = torch.cat(input1, 1).to(device)
                raw_shape = input1.shape

                # Pad the input images to N*128
                height_new = int(np.ceil(raw_shape[2] / cfg.MIN_SCALE) * cfg.MIN_SCALE)
                width_new = int(np.ceil(raw_shape[3] / cfg.MIN_SCALE) * cfg.MIN_SCALE)
                padding = (0, width_new - raw_shape[3], 0, height_new - raw_shape[2])

                if cfg.PAD_BY_CONS:
                    input1 = torch.nn.functional.pad(input1, padding, "constant", cfg.PAD_CONS)
                else:
                    input1 = torch.nn.functional.pad(input1, padding, "replicate", 0)

                if cfg.CLAMP_INPUT:
                    input1 = torch.clamp(input1, -1, 1)

                # Compute outputs
                start = time.time()
                output = model(input1)

                b, _, h, w = target.size()
                if type(output) is not tuple:
                    flow1 = output
                    output = {output}
                else:
                    flow1 = output[0]

                # Upsample the highest resolution flow to raw resolution
                realflow = F.interpolate(flow1, (h, w), mode='bilinear', align_corners=True)
                realflow[:, 0, :, :] = realflow[:, 0, :, :] * (w / flow1.shape[3])
                realflow[:, 1, :, :] = realflow[:, 1, :, :] * (h / flow1.shape[2])

                cur_flow = output[0]

                up_flow = F.interpolate(cur_flow, (h, w), mode='bilinear', align_corners=True)
                up_flow[:, 0, :, :] = up_flow[:, 0, :, :] * (w / cur_flow.shape[3])
                up_flow[:, 1, :, :] = up_flow[:, 1, :, :] * (h / cur_flow.shape[2])

                end = time.time()
                elapsed_time = end - start
                time_lst.append(elapsed_time)
                print(f"Elapsed time of iter {counter}: {elapsed_time}")
                counter += 1

                realflow_vis = flow_to_image(up_flow[0].cpu().detach().numpy().transpose(1, 2, 0), None)
                out.write(realflow_vis[:, :, [2, 1, 0]])
                # out.write(realflow_vis)
            print(f"Average elapsed time: {np.mean(time_lst)}")
            out.release()


if __name__ == '__main__':
    # User input
    base_directory = "D:\\AirSim simulator\\FDD\\Optical flow\\DICL-Flow"
    # img_folder = "D:\\AirSim simulator\\FDD\\Optical flow\\example_images\\Sintel_clean_ambush"
    # img_folder = "D:\\AirSim simulator\\FDD\\Optical flow\\example_images\\Coen_city_256_144"
    # img_folder = "D:\\AirSim simulator\\FDD\\Optical flow\\example_images\\Coen_City_512_288"
    img_folder = "D:\\AirSim simulator\\FDD\\Optical flow\\example_images\\Coen_City_1024_576_2"
    video_path = os.path.join("D:\\AirSim simulator\\FDD\\Optical flow\\video_storage",
                              img_folder.split("\\")[-1])
    fps_out = 2
    start_frame = 0
    step = 1
    if "Coen" in img_folder:
        fps_out = 30
        start_frame = 39
        step = 1
    elif "KITTI" in img_folder:
        fps_out = 1
        start_frame = 0
        step = 1

    parser, group = user_input()

    frameSize = imread(os.path.join(img_folder, os.listdir(img_folder)[0])).shape[1::-1]
    filename = os.path.join(video_path,
                            video_path.split("\\")[-1] + f"_DICL_s{start_frame}_f{fps_out}_k{step}.avi")

    parser.add_argument('--fps_out', default=fps_out, help='frames per second of the output video')
    parser.add_argument('--img_start', default=start_frame, help='starting image')
    parser.add_argument('--img_step', default=step, help='number of frames to look back to create the flow')
    parser.add_argument('--frameSize', default=frameSize, help='frames per second of the output video')
    parser.add_argument('--video_out_name', default=filename, help='filepath of output video name')
    args = parser.parse_args()

    args.b, args.batch_size = 1, 1
    args.e, args.evaluate = True, True
    args.pretrained = os.path.join(base_directory, "pretrained\\ckpt_sintel.pth.tar")
    args.cfg = os.path.join(base_directory, "cfgs\\dicl4_sintel.yml")
    args.data = img_folder
    args.exp_dir = os.path.join(base_directory, "logging")
    args.dataset = "mpi_sintel_final"
    args.visual_all = True
    evaluate_video_DICL(args)
