import numpy as np

# npy 파일 경로 지정
file_name = input("give npy file name\n")

# npy 파일 읽기
data = np.load(file_name,  allow_pickle=True)

# 파일 내용을 출력 (필요에 따라 사용)
print(data)

