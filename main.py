import argparse
import numpy as np
import matplotlib.image as mpimg
from utils.navi_data import Navi
from utils import imgproc, file_utils
from utils.myutils import make_mask, template_matching_si, template_matching_
from utils.config import normal_vector as cam_surface_normal

parser = argparse.ArgumentParser(description='Optimal viewpoint selection')
parser.add_argument('--data_folder', default='data/optimal_viewpoint/', type=str, help='folder path to input images')
parser.add_argument('--verbose', default=False, type=bool, help='verbose')
args = parser.parse_args()

img_dir = args.data_folder + 'image/'
sf_dir = args.data_folder + 'surface_normal/'
depth_dir = args.data_folder + 'depth/'
detect_dir = args.data_folder + 'det/'
anno_dir = args.data_folder + 'anno/'
NV = Navi()

image_list, sf_list, bbox_list, depth_list = file_utils.get_files(args.data_folder)
im_paths, target_pois = file_utils.get_annos(anno_dir)
im_cnt = 0
tot_acc = []
tot_acc_template = []
tot_acc_template_increasement = []
tot_test = len(im_paths)
num_processed = 0
M = 19.2
acc_stats = {}
temp_acc_stats = {}
for tt in np.unique(target_pois):
    acc_stats[tt] = []
    temp_acc_stats[tt] = []
case_acc_stats = {"c1": [], "c2": [], "c3": [], "c4": []}
error_stats = {"bbox": 0, "template_matching": 0, "street": 0}
for im_path, target_poi in zip(im_paths, target_pois):
    im_cnt += 1
    file = im_path
    init_view = file
    NV.file2curpos(file)

    print("Testing... {}/{}, Target: {}".format(im_cnt, tot_test, target_poi))
    templates, main_template, optim_view = file_utils.get_templates(args.data_folder, targetPOI=target_poi)
    if args.verbose:
        print("init loc:", file)
    img = mpimg.imread(img_dir+file+'.jpg')
    sf = imgproc.loadImage(sf_dir + file + '_sf.jpg')
    sf = sf / 255. * 2 - 1
    sf = sf / np.expand_dims(np.power(sf[:, :, 0] ** 2 + sf[:, :, 1] ** 2 + sf[:, :, 2] ** 2, 0.5), -1)
    h, w, _ = img.shape

    with open(detect_dir+file+'.txt', "r") as det_file:
        bbox = det_file.readlines()

    if len(bbox) > 0:
        # Read all boxes from the detection results
        bbs = []
        for bb in bbox:
            bbs.append(bb.rstrip().split(','))
        bbox = np.stack([np.float32(bbs[i]) for i in range(len(bbs))])
        bbox = np.reshape(bbox, [-1, 4, 2])

        # Template matching (Target POI and the boxes)
        template_matched, bbox, _, old_score = template_matching_si(img, bbox, templates, main_template)
        bbox = np.reshape(bbox, [-1, 4, 2])

        # Check that there is a bounding box which is matched with template (target POI)
        if len(bbox) > 0 and template_matched:
            bbox = bbox.astype(np.int32)
            # Take surface normal to decide the amount of rotation
            mask = make_mask(bbox, shape=[h, w])
            poi_sf_norm = np.median(sf[mask == 1], 0)

            POI_imgloc = ("left", "right")[(bbox[0, 2, 0] + bbox[0, 0, 0]) / 2 > w/2]
            poi_surface_mask = (np.dot(sf, poi_sf_norm) > 0.8).astype(np.int32)

            plane_sf_norm = np.median(sf[poi_surface_mask == 1], 0)

            # Check that the bounding box is on the left or right buildings (not street or sky)
            if abs(poi_sf_norm[1]) < 0.8:
                # theta_r = np.arccos(np.dot(sf_norm, center_sf_norm))
                depth_ = imgproc.loadImage(depth_dir + file + '_depth.jpg')
                depth_ = depth_/255.
                depth = depth_[mask == 1]
                center_depth = depth_[100:160, 240:270]

                D_pb = np.mean(depth) * M
                D_cr = np.mean(center_depth) * M  # d2
                A_plane_to_cam = np.arccos(np.dot(plane_sf_norm, cam_surface_normal))  # center point
                A_pcr = np.abs(np.pi / 2 - A_plane_to_cam)

                if D_pb == D_cr:
                    theta_r = 0
                else:
                    theta_r = np.arctan((abs(D_cr - D_pb) * np.tan(A_pcr)) / (np.maximum(D_pb, np.finfo(float).eps)))

                # 1. Align the POI and the camera center.
                rotated = False
                if round((180 / np.pi * theta_r)/30) > 0 and abs((bbox[0, 2, 0] + bbox[0, 0, 0])/2 - w/2) > w/4:
                    NV.turn(30 / 180 * np.pi, POI_imgloc, verbose=args.verbose)
                    rotated = True
                file = NV.curpos2file()

                if args.verbose:
                    print(">> Aligned the target POI on the center of the camera")
                    print("loc:", file)

                file = NV.curpos2file()
                depth_ = imgproc.loadImage(depth_dir + file + '_depth.jpg')
                depth_ = depth_/255.
                with open(detect_dir + file + '.txt', "r") as det_file:
                    bbox = det_file.readlines()

                if len(bbox) > 0:
                    bbs = []
                    for bb in bbox:
                        bbs.append(bb.rstrip().split(','))
                    bbs = np.stack([np.float32(bbs[i]) for i in range(len(bbs))])
                    bbs = np.reshape(bbs, [-1, 4, 2])
                    bbs = bbs.astype(np.int32)

                    # Find the bounding box for the target POI
                    template_matched, bbox, index, _ = template_matching_si(img, bbs, templates, main_template)
                    bbox = np.reshape(bbox, [-1, 4, 2])
                    mask = make_mask(bbox, shape=[h, w])
                    if np.sum(mask) > 0 and template_matched:
                        depth = depth_[mask == 1]
                        center_depth = depth_[130:160, 220:300]
                        if np.mean(depth) <= 0.01 and len(bbs) > 1:
                            indices = list(np.arange(len(bbs)))
                            indices.pop(index)
                            bbs = bbs[indices]
                            bbs = np.reshape(bbs, [-1, 4, 2])
                            template_matched, bbox, _, _ = template_matching_si(img, bbs, templates, main_template)
                            bbox = np.reshape(bbox, [-1, 4, 2])
                            mask = make_mask(bbox, shape=[h, w])
                            depth = depth_[mask == 1]

                        # TODO: Estimate the exact distance
                        D_pb = np.mean(depth) * M
                        D_cr = np.mean(center_depth) * M
                        cond = (D_cr < D_pb)
                        ratio = (abs(bbox[0, 3, 1] - bbox[0, 0, 1]) + abs(bbox[0, 1, 1] - bbox[0, 2, 1])) / 2 / h
                        if target_poi == "PARIS_BAGUETTE":
                            opt_ratio = 0.097
                        elif target_poi == "PASCUCCI":
                            opt_ratio = 0.08
                        elif target_poi == "CU":
                            opt_ratio = 0.07
                        elif target_poi == "FRANGCORS_FANCY":
                            opt_ratio = 0.08
                        elif target_poi == "HUE_GIMBAB":
                            opt_ratio = 0.08
                        elif target_poi == "LOTTERIA":
                            opt_ratio = 0.1
                        elif target_poi == "LOTTE_TOUR":
                            opt_ratio = 0.12

                        sf = imgproc.loadImage(sf_dir + file + '_sf.jpg')
                        sf = sf / 255. * 2 - 1
                        sf = sf / np.expand_dims(np.power(sf[:, :, 0] ** 2 + sf[:, :, 1] ** 2 + sf[:, :, 2] ** 2, 0.5), -1)
                        poi_sf_norm = np.median(sf[mask == 1], 0)

                        # Decide the POI is on the left or the right
                        POI_surf = ("left", "right")[poi_sf_norm[0] < 0]

                        if cond and POI_surf == "right":
                            case = "c1"
                        elif cond and POI_surf == "left":
                            case = "c2"
                        elif not cond and POI_surf == "right":
                            case = "c3"
                        elif not cond and POI_surf == "left":
                            case = "c4"

                        poi_surface_mask = (np.sum(sf * poi_sf_norm, -1) > 0.8).astype(np.int32)
                        plane_sf_norm = np.mean(sf[poi_surface_mask == 1], 0)
                        A_plane_to_cam = abs(np.arccos(np.dot(plane_sf_norm, cam_surface_normal)))
                        D_rb = np.abs(D_pb - D_cr) / np.tan(A_plane_to_cam)
                        D_pr = np.sqrt(D_pb ** 2 + D_rb ** 2)
                        D_pt = D_pr * (ratio / opt_ratio)
                        D_plane_to_cam = D_cr * np.cos(A_plane_to_cam)

                        A_prb = np.arctan(D_pb/(np.maximum(D_rb, np.finfo(float).eps)))
                        A_rpb = np.pi/2 - A_prb
                        A_rpt = A_rpb + A_plane_to_cam if cond else np.abs(A_rpb - A_plane_to_cam)
                        A_prt = np.arctan(D_pt * np.sin(A_rpt) / np.maximum(D_pr - D_pt * np.cos(A_rpt), np.finfo(float).eps))
                        D_robot_to_target = D_pt * abs(np.sin(A_rpt) / np.maximum(np.sin(A_prt), np.finfo(float).eps))
                        is_backward = True if D_pt > D_plane_to_cam else False
                        theta_move = abs(np.pi / 2 - A_rpt - A_prt)
                        NV.move_once(D_robot_to_target, theta_move, A_plane_to_cam, case, is_backward, D_pr, D_cr, verbose=args.verbose)
                        fin_view = NV.curpos2file()

                        if args.verbose:
                            print("Target: {}, Final loc: {}".format(target_poi, fin_view))

                        if NV.dist(optim_view, init_view) != 0:
                            acc = 1 if NV.dist(optim_view, fin_view) < 5 and NV.rotdist(optim_view, fin_view) <= 60 else 0
                            tot_acc.append(acc)
                            temp_acc = template_matching_(img, bbox, templates)
                            tot_acc_template.append(temp_acc)
                            num_processed += 1
                            acc_stats[target_poi].append(acc)
                            temp_acc_stats[target_poi].append(temp_acc)
                            case_acc_stats[case].append(acc)
                    else:
                        error_stats["bbox"] += 1
                        if args.verbose:
                            print("*Error*BBOX is not found on the rotated scene")
                else:
                    error_stats["bbox"] += 1
                    if args.verbose:
                        print("*Error*BBOX is not found on the rotated scene")
            else:
                error_stats["street"] += 1
                if args.verbose:
                    print("*Error*BBOX is found on the street")
        else:
            error_stats["template_matching"] += 1
            if args.verbose:
                print("*Error*There isn't any nearby POI")
    else:
        error_stats["bbox"] += 1
        if args.verbose:
            print("*Error*There isn't any BBOX on this image")

    if im_cnt % 20 == 0:
        print("Total Distance ACC", np.mean(tot_acc))
        print("Total Mean Distance ACC", np.mean([np.mean(values) for key, values in acc_stats.items() if len(values) > 0]))
        print("Total Template Matching ACC", np.mean(tot_acc_template))
        print("Total Mean Template Matching ACC", np.mean([np.mean(values) for key, values in temp_acc_stats.items() if len(values) > 0]))

print("Final Distance acc: ", np.mean(tot_acc))
print("Final Mean Distance ACC", np.mean([np.mean(values) for key, values in acc_stats.items()]))
[print(key, ": ", np.mean(values)) for key, values in acc_stats.items()]

print("Final Template Matching acc: ", np.mean(tot_acc_template))
print("Final Mean Template Matching ACC", np.mean([np.mean(values) for key, values in temp_acc_stats.items() if len(values) > 0]))
[print(key, ": ", np.mean(values)) for key, values in temp_acc_stats.items() if len(values) > 0]

print("Num processed: ", num_processed)