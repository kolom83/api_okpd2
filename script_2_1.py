
# Очищаем данные
import pandas as pd
import numpy as np

# ОКПД2 валидные коды
okpd_file = 'okpd-po-prom-produktsii.xlsx'
okpd_df = pd.read_excel(okpd_file)
valid_okpd2_list = okpd_df['ОКПД2 код'].dropna().astype(str).tolist()
print(f"Валидных ОКПД2 кодов: {len(valid_okpd2_list)}")
print(f"Примеры: {valid_okpd2_list[:10]}")

# Реестр РЭП - пропускаем первые 2 строки (служебные)
registry_file = 'perechen-radio-elektronnoi-produktsii.xlsx'
registry_df = pd.read_excel(registry_file, header=None)
# Находим строку с заголовками
header_idx = registry_df[registry_df[0].astype(str).str.contains('Наименование РЭП', case=False, na=False)].index[0]
registry_df = pd.read_excel(registry_file, skiprows=header_idx, header=0)
registry_df.columns = ['Наименование РЭП', 'ОКПД2 код']

# Удаляем пустые строки
registry_df = registry_df.dropna(subset=['Наименование РЭП', 'ОКПД2 код'])
registry_df['ОКПД2 код'] = registry_df['ОКПД2 код'].astype(str)

#берем наиболее часто встречаемые значения ОКПД2 если для одного Наименования более одного ОКПД2
registry_df = registry_df.groupby('Наименование РЭП')['ОКПД2 код'] \
                         .agg(lambda x: x.value_counts().index[0]) \
                         .reset_index()

print(f"\nВсего записей в реестре: {len(registry_df)}")
print(f"\nПримеры из реестра:")
print(registry_df.head(10))

# Фильтруем по валидным ОКПД2
valid_set = set(valid_okpd2_list)
registry_filtered = registry_df[registry_df['ОКПД2 код'].isin(valid_set)].copy()
print(f"\nПосле фильтрации по валидным ОКПД2: {len(registry_filtered)} записей")
print(f"Исходных записей: {len(registry_df)}")
print(f"Отфильтровано: {len(registry_df) - len(registry_filtered)}")

# Статистика ОКПД2 в реестре
print("\nТоп 20 ОКПД2 в реестре:")
print(registry_filtered['ОКПД2 код'].value_counts().head(20))

# Сохраняем очищенные данные для использования в модели
registry_filtered.to_csv('registry_cleaned.csv', index=False)
print("\n✓ Очищенные данные сохранены в registry_cleaned.csv")
