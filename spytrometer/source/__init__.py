import pandas as pd

# Путь к вашему файлу
filtered_file = '/home/ninak/tailor.assign-confidence.filtered10.txt'

# Чтение файла в DataFrame
df = pd.read_csv(filtered_file, sep='\t')

# Вывод всех названий колонок
print(df.columns)
