from multiprocessing import Pool
import os
import math
import mmengine
from tqdm import tqdm
from utils.file_io import read_text_lines
from create_bins import generate_bin_info,loaded_sensors_data_info



def pre_cache_left_cam_to_world(annotations_list, root_path):
    cache = []
    for path in tqdm(annotations_list, desc="Caching camera poses"):
        info = loaded_sensors_data_info(root_path, path)
        cache.append(info['left_cam_to_world'])
    return cache


def find_bin_end(bin_start, dists, min_bin_length):
    dist_acc = 0
    bin_end = 0
    bin_center = 0
    flag = False
    center_flag = False
    for i, dist in enumerate(dists):
        dist_acc += dist
        bin_end += 1
        if dist_acc >= min_bin_length / 2 and not center_flag:
            bin_center = bin_end
            center_flag = True
        if dist_acc >= min_bin_length:
            flag = True
            break
    if flag:
        return bin_end + bin_start, bin_center + bin_start
    return None, None


def create_kitti_infos_fast(args, annotation_path, current_seq_name):
    assert os.path.exists(args.out_dir)
    all_the_bins = {"bins": [], "adjacent_bins": []}

    annotations_list = read_text_lines(annotation_path)
    annotations_list = [os.path.join(args.root_path, f) for f in annotations_list]

    # Cache camera poses
    cam_pose_cache = pre_cache_left_cam_to_world(annotations_list, args.root_path)

    # Calculate distances
    dists = []
    for i in range(len(annotations_list) - 1):
        p0 = cam_pose_cache[i][:3, 3]
        p1 = cam_pose_cache[i + 1][:3, 3]
        dists.append(math.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2))

    bin_id = 0
    bin_start = 0
    bin_jobs = []

    if sum(dists) >= args.min_bin_length:
        while bin_start < len(annotations_list):
            bin_end, bin_center = find_bin_end(bin_start, dists[bin_start:], args.min_bin_length)
            if bin_end is None:
                break

            bin_token = f"scene{current_seq_name}_bin{bin_id:03d}"
            bin_jobs.append((args, bin_token, current_seq_name, annotations_list, bin_start, bin_end, bin_center, sum(dists[bin_start:bin_end])))

            bin_start += 1
            bin_id += 1

    else:
        _, bin_center = find_bin_end(bin_start, dists[bin_start:], args.min_bin_length)
        bin_token = f"scene{current_seq_name}_bin{bin_id:03d}"
        bin_jobs.append((args, bin_token, current_seq_name, annotations_list, 0, len(annotations_list)-1, bin_center, sum(dists)))

    os.makedirs(os.path.join(args.out_dir, f"bin_infos_{args.min_bin_length}"), exist_ok=True)

    def save_bin(bin_info):
        bin_token = bin_info['token']
        out_path = os.path.join(args.out_dir, f"bin_infos_{args.min_bin_length}", f"{bin_token}.pkl")
        mmengine.dump(bin_info, out_path)
        return bin_token

    # Run in parallel
    with Pool(processes=os.cpu_count()) as pool:
        bin_infos = list(tqdm(pool.starmap(generate_bin_info, bin_jobs), desc="Generating bins"))

    for info in bin_infos:
        all_the_bins['bins'].append(info['token'])
        all_the_bins['adjacent_bins'].append([info['token']])

    return all_the_bins


def kitti360_data_prep_fast(args):

    idx = 0
    all_sequence_names = sorted(os.listdir(args.filelist_folder))
    all_sequence_names = all_sequence_names[4:]
    
    for filename_list in all_sequence_names:
        seq_name = os.path.basename(filename_list)[:-9]
        print(f"Processing sequence {seq_name} ({idx}/{len(os.listdir(args.filelist_folder))})")
        idx = idx + 1
        current_annotations_fname = os.path.join(args.filelist_folder, filename_list)
        create_kitti_infos_fast(args, current_annotations_fname, seq_name)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="/media/zliu/data12/dataset/KITTI/VSRD_Format/")
    parser.add_argument("--filelist_folder", type=str, default="/home/zliu/Desktop/Project2025/KITTI360_for_feedforward/Preprocessing/filelist")
    parser.add_argument("--min_bin_length", type=float, default=8.0)
    parser.add_argument("--out_dir", type=str, default="/media/zliu/data12/dataset/KITTI/VSRD_Format/feedforward_bins")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    kitti360_data_prep_fast(args)
