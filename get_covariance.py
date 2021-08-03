#Compute covariance of the noise in measurements

import json
import os
import cv2
import numpy
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description=" ")
    parser.add_argument(
        "--input_dir",
        metavar="FILE",
        help="Path to directory containing json files",
    )
    return parser

args = get_parser().parse_args()

#Camera calibration parameters (will be used to compute world coordinates for detected points)
K= numpy.transpose(numpy.matrix('244.3580 0 0; 0 243.8489 0; 128.6480 162.3639 1')) #intrinsic matrix
distCoeffs = numpy.matrix('-0.39108 0.14139 0 0') #k1, k2, p1, p2 (k_i:radial distortion; p_i:tangential distortion)
R = numpy.transpose(numpy.matrix('0.4272 0.3382 -0.8385; -0.9041 0.1480 -0.4009; -0.0115 0.9294 0.3690')) #rotation matrix
T = numpy.transpose(numpy.matrix('0.2467 0.5615 2.4545')) #translation vector [m]
projMatrix = K * numpy.concatenate((R[:, 0:2], T), axis=1)

#Read annotations from json file (one file for groundtruth and another from the processing method that we want to evaluate,
#containing predictions from detectron2)


# Get all json files
files = sorted(os.listdir(args.input_dir))
points_gt = None
points_m0 = None
points_m1 = None

for file in files:
    file_path = os.path.join(args.input_dir, file)
    json_file = open(file_path, "r")
    json_data = json.load(json_file)
    json_file.close()
    if file.startswith('gt'): #add all "person" instances detected
        for ann in json_data:
            detected_person = False
            if ann == []:
                # No se detecta nada
                p = numpy.matrix([[0],[0]])
            else:
                for inst in ann:
                    if inst["category_id"] == 0 and detected_person is False: #persona
                        bbox = inst["bbox"] #box mode is XYWH_ABS; XY is the top left corner
                        #Compute center of bbox's lower edge; origin (0,0) is the top left corner of the image
                        x = bbox[0] + bbox[2]/2
                        y = bbox[1] + bbox[3]
                        p = numpy.matrix([[x], [y]])
                        detected_person = True
            if points_gt is None:
                points_gt = p
            else:
                points_gt = numpy.concatenate((points_gt, p), axis = 1)
        print('shape points_gt after every file: ', points_gt.shape)


    if file.startswith('modo0'):
        for ann in json_data:
            detected_person = False
            if ann == []:
                # No se detecta persona
                p = numpy.matrix([[0],[0]])
            else:
                for inst in ann:
                    if inst["category_id"] == 0 and detected_person is False:  # persona
                        bbox = inst["bbox"]  # box mode is XYWH_ABS; XY is the top left corner
                        # Compute center of bbox's lower edge; origin (0,0) is the top left corner of the image
                        x = bbox[0] + bbox[2] / 2
                        y = bbox[1] + bbox[3]
                        p = numpy.matrix([[x], [y]])
                        detected_person = True
            if points_m0 is None:
                points_m0 = p
            else:
                points_m0 = numpy.concatenate((points_m0, p), axis=1)  # every column of the matrix corresponds to one point



    if file.startswith('modo1'):
        for ann in json_data:
            detected_person = False
            if ann == []:
                # No se detecta persona
                p = numpy.matrix([[0],[0]])
            else:
                for inst in ann:
                    if inst["category_id"] == 0 and detected_person is False:  # persona
                        bbox = inst["bbox"]  # box mode is XYWH_ABS; XY is the top left corner
                        # Compute center of bbox's lower edge; origin (0,0) is the top left corner of the image
                        x = bbox[0] + bbox[2] / 2
                        y = bbox[1] + bbox[3]
                        p = numpy.matrix([[x], [y]])
                        detected_person = True

            if points_m1 is None:
                points_m1 = p
            else:
                points_m1 = numpy.concatenate((points_m1, p), axis=1)  # every column of the matrix corresponds to one point

print('shape pgt, pm0, pm1 pre: ', points_gt.shape, points_m0.shape, points_m1.shape)

# Remove images with no prediction (z = [0, 0])
p_gt = None
for i in range(len(numpy.transpose(points_gt))):
    point_gt = points_gt[:, i]
    point_m0 = points_m0[:, i]
    point_m1 = points_m1[:, i]
    if numpy.sum(point_gt) != 0 and numpy.sum(point_m1) != 0 and numpy.sum(point_m0) != 0:
        if p_gt is None:
            p_gt = point_gt
            p_m0 = point_m0
            p_m1 = point_m1
        else:
            p_gt = numpy.concatenate((p_gt, point_gt), axis=1)
            p_m0 = numpy.concatenate((p_m0, point_m0), axis=1)
            p_m1 = numpy.concatenate((p_m1, point_m1), axis=1)

print('shape pgt, pm0, pm1 post: ', p_gt.shape, p_m0.shape, p_m1.shape)

#World coordinates for the detected image points
#undistort points
points_gt = cv2.undistortPoints(p_gt, K, distCoeffs, P=K)
points_m0 = cv2.undistortPoints(p_m0, K, distCoeffs, P=K)
points_m1 = cv2.undistortPoints(p_m1, K, distCoeffs, P=K)

ones = numpy.ones((len(points_gt),1))
points_gt = numpy.transpose(numpy.concatenate((points_gt[:,0,:], ones),axis=1))
points_m0 = numpy.transpose(numpy.concatenate((points_m0[:,0,:], ones),axis=1))
points_m1 = numpy.transpose(numpy.concatenate((points_m1[:,0,:], ones),axis=1))

#compute world coordinates
points_gt_w = numpy.linalg.inv(projMatrix) * points_gt
points_m0_w = numpy.linalg.inv(projMatrix) * points_m0
points_m1_w = numpy.linalg.inv(projMatrix) * points_m1
points_gt_w = points_gt_w[0:2, :] / points_gt_w[2, :]
points_m0_w = points_m0_w[0:2, :] / points_m0_w[2, :]
points_m1_w = points_m1_w[0:2, :] / points_m1_w[2, :]

#The value of the "noise" (v_pk) in the measurement is the difference between the coordinates predicted by the method and the
#groundtruth value
v_m0 = points_m0_w - points_gt_w
v_m1 = points_m1_w - points_gt_w
print('mean m0', v_m0.mean(axis=1))
print('mean m1', v_m1.mean(axis=1))
#zero mean
v_m0 = v_m0 - v_m0.mean(axis=1)
v_m1 = v_m1 - v_m1.mean(axis=1)

#Compute covariance
R_m0 = numpy.cov(v_m0)
R_m1 = numpy.cov(v_m1)

print('Rm0', R_m0)
print('Rm1', R_m1)