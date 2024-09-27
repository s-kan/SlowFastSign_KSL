import pandas as pd

# CSV 파일 불러오기 (여러 인코딩 시도)
file_path = 'NIA_SEN_val.csv'

# 우선 CP949로 시도해보고, 안 될 경우 다른 인코딩도 시도 가능
try:
    data = pd.read_csv(file_path, encoding='CP949')  # CP949 또는 EUC-KR로 시도
except UnicodeDecodeError:
    data = pd.read_csv(file_path, encoding='utf-8')  # UTF-8로 시도

# 1. 'Num' 값이 7999 이하인 행만 필터링
filtered_data = data[data['Num'] <= 6000]
filtered_data = filtered_data[4000 <= filtered_data['Num']]

# 2. 'Filename' 열에서 "_F.mp4" 제거
filtered_data['Filename'] = filtered_data['Filename'].str.replace('_F.mp4', '', regex=False)

# 결과를 새로운 CSV 파일로 저장 (UTF-8 with BOM으로 저장하면 Excel에서 한글 깨짐 방지)
output_path = 'NIA_SEN_dev.csv'
filtered_data.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"필터링된 데이터를 {output_path}에 저장했습니다.")

