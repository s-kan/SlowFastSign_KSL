import re
import os
import cv2
import pdb
import glob
import pandas as pd
import argparse
import numpy as np
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool

def csv2dict(anno_path, dataset_type):
    inputs_list = pd.read_csv(anno_path)

    # 4002번에서 6000번까지의 데이터만 선택
    # inputs_list = inputs_list[(inputs_list['Num'] >= 4002) & (inputs_list['Num'] <= 6000)]

    
    info_dict = dict()
    info_dict['KSL'] = "../dataset/KSL/fullFrame-1960x1080px"

    print(f"Generate information dict from {anno_path}")
    for file_idx, row in tqdm(inputs_list.iterrows(), total=len(inputs_list)):
        fileid = row['Num']
        folder = row['Foldername'] + '/*.jpg'
        label = row['Kor']
	
        tmp = folder + '|' + label
	
        # 각 영상에 해당하는 프레임의 수 계산
        num_frames = len(glob.glob(f"{info_dict['KSL']}/train/{folder}/*.jpg"))

        # 딕셔너리에 정보 저장
        info_dict[file_idx] = {
            'fileid': fileid,
            'folder': dataset_type + f"/{folder}",
            'label': label,
            'num_frames': num_frames,
            'original_info': tmp
        }

    return info_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Data process for Visual Alignment Constraint for Continuous Sign Language Recognition.')
    parser.add_argument('--dataset', type=str, default='KSL',
                        help='save KSL')
    parser.add_argument('--dataset-root', type=str, default='../dataset/KSL',
                        help='path to the dataset')
    parser.add_argument('--annotation-KSL', type=str, default='./KSL/NIA_SEN_{}.csv',
                        help='annotation prefix')



    args = parser.parse_args()
    mode = ["train", "dev"]
    sign_dict = dict()
    if not os.path.exists(f"./{args.dataset}"):
        os.makedirs(f"./{args.dataset}")
    for md in mode:
        # generate information dict
        information = csv2dict(f"{args.annotation_KSL.format(md)}", dataset_type=md)

        np.save(f"./{args.dataset}/{md}_info.npy", information)


