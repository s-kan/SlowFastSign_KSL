import numpy as np

# npy 파일의 경로
file_path = 'gloss_dict.npy'

# 파일 읽기
data = np.load(file_path, allow_pickle=True)

# 데이터 사용 예시
print(data)
