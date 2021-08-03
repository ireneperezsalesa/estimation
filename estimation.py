import numpy
import torch
import sys
import os
import cv2
import argparse
import zipfile
import math
import itertools
import time
import matplotlib.pyplot as plt
from scipy.linalg import expm
from detectron2.config import get_cfg
from predictor import VisualizationDemo
from detectron2.data.detection_utils import read_image
sys.path.insert(0, './refine_lstm_head/rpg_e2vid/')
from refine_lstm_head.rpg_e2vid.utils.inference_utils import events_to_voxel_grid_pytorch
from refine_lstm_head.rpg_e2vid.image_reconstructor_switching import ImageReconstructorSwitching
from refine_lstm_head.rpg_e2vid.options.inference_options import set_inference_options

def get_parser():
    parser = argparse.ArgumentParser(description=" ")
    parser.add_argument(
        "--input",
        metavar="FILE",
        help="Path to txt or zip file containing events",
    )
    parser.add_argument(
        "--detection_output_folder",
        help="Directory to save output frames showing object instances predicted by Detectron2. "
    )
    parser.add_argument(
        "--scheduling_method",
        help="Scheduling policy: 'greedy', 'greedy2', 'mh', 'fixed'. "
    )
    parser.add_argument(
        "--schedule",
        help="List containing the desired schedule (any length, the schedule will be repeated to cover the total time. "
    )
    parser.add_argument(
        "--txt_file",
        help="txt file to save estimation data (timestamp, (x,y) observed, (x,y) estimated). "
    )
    parser.add_argument('--mh_T', default=0, type=float)
    parser.add_argument('--skipevents', default=0, type=int)
    parser.add_argument('--suboffset', default=0, type=int)
    parser.add_argument('--compute_voxel_grid_on_cpu', dest='compute_voxel_grid_on_cpu', action='store_true')
    parser.set_defaults(compute_voxel_grid_on_cpu=False)
    return parser


if __name__ == "__main__":
    parser = get_parser()
    set_inference_options(parser)
    args = parser.parse_args()

    if args.scheduling_method == 'fixed':
        schedule = list(map(int, args.schedule.strip('[]').split(',')))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load reconstruction nn model
    model = torch.load('./refine_lstm_head/rpg_e2vid/checkpoints/mixed_modes_8_32ch/m_149_473.pth')
    model = model.to(device)

    # Load detectron models
    cfg_mode0 = get_cfg()
    cfg_mode0.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
    cfg_mode0.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/model_final_2d9806.pkl"
    cfg_mode0.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
    cfg_mode0.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg_mode0.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5
    cfg_mode0.freeze()

    detector_mode0 = VisualizationDemo(cfg_mode0)

    cfg_mode1 = get_cfg()
    cfg_mode1.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg_mode1.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
    cfg_mode1.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
    cfg_mode1.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg_mode1.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5
    cfg_mode1.freeze()

    detector_mode1 = VisualizationDemo(cfg_mode1)

    # Parameters for every processing mode
    processing_time_0 = 0.4
    processing_time_1 = 0.1
    processing_time = [processing_time_0, processing_time_1]
    R0 = numpy.matrix('0.01669749 0.00561141; 0.00561141 0.00435193')
    R1 = numpy.matrix('0.06494701 0.0298462; 0.0298462 0.02491636')
    Rk = [R0, R1]

    # Matrices to describe the dynamic of the system
    A = numpy.matrix('0 0 1 0; 0 0 0 1; 0 0 0 0; 0 0 0 0')
    B = numpy.matrix('0 0; 0 0; 1 0; 0 1')
    H = numpy.matrix('1 0 0 0; 0 1 0 0')
    H_t = numpy.transpose(H)

    Ad0 = expm(A * processing_time_0)
    Ad1 = expm(A * processing_time_1)
    Ad = [Ad0, Ad1]

    Q0 = numpy.matrix('0.5 0 0 0; 0 0.5 0 0; 0 0 0.5 0; 0 0 0 0.5')  # [m/s²]
    Qd0 = numpy.matrix('1.0667e-2 0 4e-2 0; 0 1.0667e-2 0 4e-2; 4e-2 0 2e-1 0; 0 4e-2 0 2e-1')
    Qd1 = numpy.matrix('1.6667e-4 0 2.5e-3 0; 0 1.6667e-4 0 2.5e-3; 2.5e-3 0 5e-2 0; 0 2.5e-3 0 5e-2')
    Qd = [Qd0, Qd1]

    # Camera parameters
    K = numpy.transpose(numpy.matrix('244.3580 0 0; 0 243.8489 0; 128.6480 162.3639 1'))  # intrinsic matrix
    distCoeffs = numpy.matrix('-0.39108 0.14139 0 0') # k1, k2, p1, p2 (k_i:radial distortion; p_i:tangential distortion)
    R = numpy.transpose(numpy.matrix('0.4272 0.3382 -0.8385; -0.9041 0.1480 -0.4009; -0.0115 0.9294 0.3690'))  # rotation matrix
    T = numpy.transpose(numpy.matrix('0.2467 0.5615 2.4545'))  # translation vector [m]
    projMatrix = K * numpy.concatenate((R[:, 0:2], T), axis=1)

    # Open event file
    if args.input.endswith('zip'):
        zip_file = zipfile.ZipFile(args.input)
        files_in_archive = zip_file.namelist()
        assert (len(files_in_archive) == 1)  # make sure there is only one text file in the archive
        event_file = zip_file.open(files_in_archive[0], 'r')
    else:
        event_file = open(args.input, 'r')

    # get width and height from text file header
    for line in event_file:
        if args.input.endswith('zip'):
            line = line.decode("utf-8")
        width, height = line.split(' ')
        width, height = int(width), int(height)
        break
    # ignore header + the first start_index lines
    for i in range(1):  # + range(1 + start_index) if we want to ignore more lines
        event_file.readline()

    # Configure image reconstructor
    reconstructor = ImageReconstructorSwitching(model, height, width, model.num_bins, args)

    # Penalties for the cost function
    rpk0 = 1
    rpk1 = 1
    rpk = [rpk0, rpk1]

    # Set initial values
    x_k = numpy.matrix('0; 0; 0; 0')
    P_k = numpy.matrix('10000 0 0 0; 0 10000 0 0; 0 0 10000 0; 0 0 0 10000')
    k = 0
    p = 0
    sim_time = 0
    total_cost = 0

    states = None
    last_stamp = None
    last_processing_time = None
    start_index = 0
    z_img = numpy.matrix([[height], [width]], numpy.float32)  # start in lower right corner (for this dataset)
    x_k_accum = x_k[0:2,:]
    z_k_accum = x_k[0:2,:]
    time_stamp_list = numpy.matrix('0')
    list_of_modes = numpy.matrix('1000') # ignore this value later

    # Total time
    total_time = 15  # s

    # Algorithm for estimation
    while sim_time < total_time:

        # MEASUREMENT ------------------------------------------------------------------------------------------------------

        if last_processing_time is None:
            duration = 0.1  # 100ms window to get some initial data. After this, the duration window will correspond
                            # to the time spent processing data in the previous step
        else:
            duration = last_processing_time

        # Get events in event window
        event_list = []
        for line in event_file:
            if args.input.endswith('zip'):
                line = line.decode("utf-8")
            t, x, y, pol = line.split(' ')
            t, x, y, pol = float(t), int(x), int(y), int(pol)
            if last_stamp is None:
                last_stamp = t
            if t > last_stamp + duration: # stop adding events when time exceeds the desired time window
                last_stamp = t
                event_window = numpy.array(event_list)
                time_stamp_list = numpy.concatenate((time_stamp_list, numpy.matrix([t])), axis=0)
                break
            if t >= last_stamp:
                event_list.append([t, x, y, pol])

        # Turn events into torch tensor
        event_tensor = events_to_voxel_grid_pytorch(event_window,
                                                    num_bins=5,
                                                    width=width,
                                                    height=height,
                                                    device=device)

        # SCHEDULING POLICY -------------------------------------------------------------------------------------------------

        # Decide processing method using current information (P_k), with one of the following rules:
        # Greedy
        if args.scheduling_method == 'greedy':
            J_0 = numpy.trace(P_k) * processing_time_0
            J_1 = numpy.trace(P_k) * processing_time_1
            if J_0 < J_1:
                mode = 0
            else:
                mode = 1

        # Greedy2
        if args.scheduling_method == 'greedy2':
            # starting with mode 0
            Jk_0 = numpy.trace(P_k) * processing_time_0
            Lk_0 = Ad0 * P_k * H_t * numpy.linalg.inv(H * P_k * H_t + R0)
            Pk1_0 = (Ad0 - Lk_0 * H) * P_k * numpy.transpose(Ad0) + Qd0
            Jk_00 = numpy.trace(Pk1_0) * processing_time_0
            Jk_01 = numpy.trace(Pk1_0) * processing_time_1
            J_0 = min((Jk_0 + Jk_00), (Jk_0 + Jk_01))
            # starting with mode1
            Jk_1 = numpy.trace(P_k) * processing_time_1
            Lk_1 = Ad1 * P_k * H_t * numpy.linalg.inv(H * P_k * H_t + R1)
            Pk1_1 = (Ad1 - Lk_1 * H) * P_k * numpy.transpose(Ad1) + Qd1
            Jk_10 = numpy.trace(Pk1_1) * processing_time_0
            Jk_11 = numpy.trace(Pk1_1) * processing_time_1
            J_1 = min((Jk_1 + Jk_10), (Jk_1 + Jk_11))
            #choose option with minimum cost
            if J_0 < J_1:
                mode = 0
            else:
                mode = 1

        # Fixed length, moving horizon
        if args.scheduling_method == 'mh':
            #Compute calendar length
            l = math.ceil(args.mh_T / min(processing_time))
            #Compute all posible schedules to cover a time interval of length mh_T
            schedules = list(map(list, itertools.product([0, 1], repeat=l)))
            #For every schedule, compute attention and cost
            cost_list = []
            for sched in schedules:
                #attention
                sum = 0
                for i in range(l):
                    if sum < args.mh_T:
                        att = i
                    sum = sum + processing_time[sched[i]]
                #cost
                cost = 0
                P = P_k
                sum_delta_pk = 0
                lambda_alpha = 0.05
                for i in range(att+1):
                    if i < att:
                        cost = cost + numpy.trace(P) * processing_time[sched[i]] + lambda_alpha * rpk[sched[i]]
                        sum_delta_pk = sum_delta_pk + processing_time[sched[i]]
                        #compute next P
                        L = Ad[sched[i]] * P * H_t * numpy.linalg.inv(H * P * H_t + Rk[sched[i]])
                        P = (Ad[sched[i]] - L * H) * P * numpy.transpose(Ad[sched[i]]) + Qd[sched[i]]
                    else:
                        cost = cost + numpy.trace(P) * (args.mh_T - sum_delta_pk) + lambda_alpha * rpk[sched[i]]
                cost_list.append(cost)
            #check minimum cost
            min_cost_index = cost_list.index(min(cost_list))
            mode = schedules[min_cost_index][0]

        # Fixed schedule, decided beforehand
        if args.scheduling_method == 'fixed':
            lambda_alpha = 0.1
            mode = schedule[p]
            if p < len(schedule)-1:
                p = p + 1
            else:
                p = 0
            # add to the total cost value
            if k >= 10: #ignore the first several iterations (the value of P is very high)
                if sim_time + processing_time[mode] <= total_time:
                    total_cost = total_cost + numpy.trace(P_k) * processing_time[mode] + lambda_alpha * rpk[mode]
                else:
                    total_cost = total_cost + numpy.trace(P_k) * (total_time - sim_time) + lambda_alpha * rpk[mode]

        print('Chosen mode', mode)

        list_of_modes = numpy.concatenate((list_of_modes, numpy.matrix([[mode]])), axis = 0)

        # DATA PROCESSING ---------------------------------------------------------------------------------------------------

        # Reconstruct image with last chosen method
        num_events_in_window = event_window.shape[0]
        event_tensor_id = start_index + num_events_in_window
        reconstructor.update_reconstruction(event_tensor, event_tensor_id, mode, last_stamp)
        start_index += num_events_in_window

        # Detect person
        frame_path = os.path.join(args.output_folder, 'reconstruction/frame_{:010d}.png'.format(event_tensor_id))
        img = read_image(frame_path, format="BGR")
        img = numpy.rot90(img) #rotate image so that the person can be detected
        if mode == 0:
            predictions, visualized_output = detector_mode0.run_on_image(img)
        if mode == 1:
            predictions, visualized_output = detector_mode1.run_on_image(img)
        out_filename = os.path.join(args.detection_output_folder, 'detection_{:010d}.png'.format(k))
        visualized_output.save(out_filename)

        # Compute image coordinates for the lower edge of the bounding box (frame coordinates)
        if "instances" in predictions:
            instances = predictions["instances"]
            if len(instances.pred_boxes) >= 1: #if there are no instances, the measurement from the last iteration will be used again
                bbox = instances[0].pred_boxes.tensor.cpu().numpy()
                bbox = bbox[0,:] #if there are more than one bbox, take only the first (class 0 = 'person')
                zx = (bbox[0] + bbox[2]) / 2
                zy = bbox[3]
                z_img[0, 0] = zx.item()
                z_img[1, 0] = zy.item()

        # Turn into world coordinates
        z_und = cv2.undistortPoints(z_img, K, distCoeffs, P=K)
        z_und = numpy.matrix([[z_und[0, 0, 0]], [z_und[0, 0, 1]], [1]])
        z_w = numpy.linalg.inv(projMatrix) * z_und
        z_k = z_w[0:2, 0] / z_w[2, 0]

        # ESTIMATION --------------------------------------------------------------------------------------------------------

        if mode == 0:
            L_k = Ad0 * P_k * H_t * numpy.linalg.inv(H * P_k * H_t + R0)
            x_k = Ad0 * x_k + L_k * (z_k - H * x_k)
            P_k = (Ad0 - L_k * H) * P_k * numpy.transpose(Ad0) + Qd0
            sim_time = sim_time + processing_time_0
            last_processing_time = processing_time_0

        if mode == 1:
            L_k = Ad1 * P_k * H_t * numpy.linalg.inv(H * P_k * H_t + R1)
            x_k = Ad1 * x_k + L_k * (z_k - H * x_k)
            P_k = (Ad1 - L_k * H) * P_k * numpy.transpose(Ad1) + Qd1
            sim_time = sim_time + processing_time_1
            last_processing_time = processing_time_1

        z_k_accum = numpy.concatenate((z_k_accum, z_k), axis=1)  # every column corresponds to an iteration
        x_k_accum = numpy.concatenate((x_k_accum, x_k[0:2,:]), axis=1) #every column corresponds to an iteration

        k = k + 1


        #PLOT Z_K AND X_K OVER THE IMAGE
        implot = plt.imshow(img)

        xy = numpy.matrix([[x_k[0,:].item()], [x_k[1,:].item()], [1]], numpy.float32)
        xy_img = projMatrix * xy;
        xy_img = xy_img[0:2, :] / xy_img[2, :]
        xy_dist = cv2.undistortPoints(xy_img, K, -distCoeffs, P=K) # approximation for small distortions
        x = numpy.squeeze(numpy.asarray(xy_dist[0,0,0]))
        y = numpy.squeeze(numpy.asarray(xy_dist[0,0,1]))
        zx = numpy.squeeze(numpy.asarray(z_img[0, :]))
        zy = numpy.squeeze(numpy.asarray(z_img[1, :]))

        plt.scatter(x, y, s=10, c='b', marker="s", label='x_img')
        plt.scatter(zx, zy, s=10, c='r', marker="o", label='z_img')
        plot_path = os.path.join('./outputs/estim_traj/frame_{:010d}.png'.format(event_tensor_id))
        plt.savefig(plot_path)

    # End of loop

    # TRANSFER DATA TO TXT FILE
    data = numpy.concatenate((list_of_modes, time_stamp_list, numpy.transpose(z_k_accum), numpy.transpose(x_k_accum)), axis=1)
    with open(args.txt_file, 'w') as f:
        if args.scheduling_method == 'fixed':
            f.write(str(schedule) + '\n')
            f.write(str(total_cost) + '\n')
        for line in data:
            string = ''
            for i in range(6): # write mode, timestamp, zk_x, zk_y, xk_x, xk_y
                string = string + str(line[0,i]) + ' '
            string = string + '\n'
            f.write(string)


    # PLOT TRAJECTORY ON THE FLOOR
    x = x_k_accum[0,:]
    y = x_k_accum[1,:]
    x = numpy.squeeze(numpy.asarray(x))
    y = numpy.squeeze(numpy.asarray(y))
    plt.plot(x, y)
    plt.show()

    zx = numpy.squeeze(numpy.asarray(z_k_accum[0,:]))
    zy = numpy.squeeze(numpy.asarray(z_k_accum[1, :]))

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.plot(x, y, c='b', label='x_k')
    ax1.plot(zx, zy, c='r', label='z_k')
    plt.legend(loc='upper left');
    plt.show()