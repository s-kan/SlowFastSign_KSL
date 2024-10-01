import os
import shutil
import pandas as pd

# CSV 파일 경로와 폴더 경로 설정
# csv_file_path = './preprocess/KSL/NIA_SEN_dev.csv'
# train_folder_path = './dataset/KSL/fullFrame-1960x1080px/train'
# dev_folder_path = './dataset/KSL/fullFrame-1960x1080px/dev'

csv_file_path = './preprocess/KSL/NIA_SEN_test.csv'
train_folder_path = './dataset/KSL/fullFrame-1960x1080px/train'
dev_folder_path = './dataset/KSL/fullFrame-1960x1080px/test'

# dev 폴더가 존재하지 않으면 생성
if not os.path.exists(dev_folder_path):
    os.makedirs(dev_folder_path)

# CSV 파일을 읽어들여 DataFrame 생성
df = pd.read_csv(csv_file_path)

# 'Foldername' 열의 폴더 이름들을 리스트로 추출
folder_names_to_move = df['Foldername'].tolist()

# 폴더들을 이동
for folder_name in folder_names_to_move:
    source_folder = os.path.join(train_folder_path, folder_name)
    destination_folder = os.path.join(dev_folder_path, folder_name)
    
    # 폴더가 실제로 존재할 경우에만 이동
    if os.path.exists(source_folder):
        shutil.move(source_folder, destination_folder)
        print(f"Moved: {source_folder} to {destination_folder}")
    else:
        print(f"Folder not found: {source_folder}")

