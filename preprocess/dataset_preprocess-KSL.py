import re
import os
import cv2
import pdb
import glob
import pandas
import argparse
import numpy as np
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool


def csv2dict(anno_path, dataset_type):
    inputs_list = pd.read_csv(anno_path)

    # 4002번에서 6000번까지의 데이터만 선택
    inputs_list = inputs_list[(inputs_list['Num'] >= 4002) & (inputs_list['Num'] <= 6000)]

    info_dict = dict()
    info_dict['KSL'] = "../dataset/KSL/"

    print(f"Generate information dict from {anno_path}")
    for file_idx, row in tqdm(inputs_list.iterrows(), total=len(inputs_list)):
        fileid = row['Num']
        folder = row['Foldername']
        label = row['Kor']

        # 각 영상에 해당하는 프레임의 수 계산
        num_frames = len(glob.glob(f"{info_dict['KSL']}/{folder}/*.jpg"))

        # 딕셔너리에 정보 저장
        info_dict[file_idx] = {
            'fileid': fileid,
            'folder': f"{dataset_type}/{folder}",
            'label': label,
            'num_frames': num_frames,
            'original_info': row
        }

    return info_dict


def generate_gt_stm(info, save_path):
    with open(save_path, "w") as f:
        for k, v in info.items():
            if not isinstance(k, int):
                continue
            f.writelines(f"{v['fileid']} 1 {v['signer']} 0.0 1.79769e+308 {v['label']}\n")


def sign_dict_update(total_dict, info):
    for k, v in info.items():
        if not isinstance(k, int):
            continue
        split_label = v['label'].split()
        for gloss in split_label:
            if gloss not in total_dict.keys():
                total_dict[gloss] = 1
            else:
                total_dict[gloss] += 1
    return total_dict


def resize_img(img_path, dsize='210x260px'):
    dsize = tuple(int(res) for res in re.findall("\d+", dsize))
    img = cv2.imread(img_path)
    img = cv2.resize(img, dsize, interpolation=cv2.INTER_LANCZOS4)
    return img


def resize_dataset(video_idx, dsize, info_dict):
    info = info_dict[video_idx]
    img_list = glob.glob(f"{info_dict['KSL']}/{info['folder']}")
    for img_path in img_list:
        rs_img = resize_img(img_path, dsize=dsize)
        rs_img_path = img_path.replace("210x260px", dsize)
        rs_img_dir = os.path.dirname(rs_img_path)
        if not os.path.exists(rs_img_dir):
            os.makedirs(rs_img_dir)
            cv2.imwrite(rs_img_path, rs_img)
        else:
            cv2.imwrite(rs_img_path, rs_img)


def run_mp_cmd(processes, process_func, process_args):
    with Pool(processes) as p:
        outputs = list(tqdm(p.imap(process_func, process_args), total=len(process_args)))
    return outputs


def run_cmd(func, args):
    return func(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Data process for Visual Alignment Constraint for Continuous Sign Language Recognition.')
    parser.add_argument('--dataset', type=str, default='KSL',
                        help='save KSL')
    parser.add_argument('--dataset-root', type=str, default='../output_frames03/',
                        help='path to the dataset')
    parser.add_argument('--annotation-KSL', type=str, default='./KSL/NIA_SEN_test.csv',
                        help='annotation prefix')
    parser.add_argument('--output-res', type=str, default='256x256px',
                        help='resize resolution for image sequence')
    parser.add_argument('--process-image', '-p', action='store_true',
                        help='resize image')
    parser.add_argument('--multiprocessing', '-m', action='store_true',
                        help='whether adopts multiprocessing to accelate the preprocess')

    args = parser.parse_args()
    mode = ["dev", "test", "train"]
    sign_dict = dict()
    if not os.path.exists(f"./{args.dataset}"):
        os.makedirs(f"./{args.dataset}")
    for md in mode:
        # generate information dict
        information = csv2dict("KSL/NIA_SEN_test.csv", dataset_type=md)
        np.save(f"./{args.dataset}/{md}_info.npy", information)
        # update the total gloss dict
        sign_dict_update(sign_dict, information)
        # generate groudtruth stm for evaluation
        generate_gt_stm(information, f"./{args.dataset}/{args.dataset}-groundtruth-{md}.stm")
        # resize images
        video_index = np.arange(len(information) - 1)
        print(f"Resize image to {args.output_res}")
        if args.process_image:
            if args.multiprocessing:
                run_mp_cmd(10, partial(resize_dataset, dsize=args.output_res, info_dict=information), video_index)
            else:
                for idx in tqdm(video_index):
                    run_cmd(partial(resize_dataset, dsize=args.output_res, info_dict=information), idx)
    sign_dict = sorted(sign_dict.items(), key=lambda d: d[0])
    save_dict = {}
    for idx, (key, value) in enumerate(sign_dict):
        save_dict[key] = [idx + 1, value]
    np.save(f"./{args.dataset}/gloss_dict.npy", save_dict)

