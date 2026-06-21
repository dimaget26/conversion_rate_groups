import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import sqlite3

# ----- ГЕНЕРИРУЕМ ДАННЫЕ -----
np.random.seed(42)
n_users = 10000

data = {
    'user_id': range(1, n_users + 1),
    'group': np.random.choice([0, 1], size=n_users, p=[0.5, 0.5]),
    'time_on_site': np.random.exponential(scale=5, size=n_users),
    'pages_viewed': np.random.poisson(lam=5, size=n_users),
    'converted': np.random.binomial(1, p=0.1, size=n_users)
}

df = pd.DataFrame(data)

# Добавляем эффект теста (конверсия в группе B на 20% выше)
group_b_mask = df['group'] == 1
df.loc[group_b_mask, 'converted'] = np.random.binomial(
    1,
    p=0.12,
    size=df[group_b_mask].shape[0]
)

df['date'] = pd.date_range(start='2025-06-01', periods=n_users, freq='10min')[:n_users]

print("=== ПЕРВЫЕ 5 СТРОК ===")
print(df.head())
print(f"\nГруппа A: {len(df[df['group'] == 0])}")
print(f"Группа B: {len(df[df['group'] == 1])}")
# ----- СОХРАНЯЕМ В БАЗУ -----
conn = sqlite3.connect('A_B_test.db')
df.to_sql('test', conn, if_exists='replace', index=False)

# ----- ЗАПРАШИВАЕМ ДАННЫЕ ИЗ БАЗЫ -----
df_ab_test = pd.read_sql("""
    WITH conversion_A AS (
        SELECT 
            COUNT(DISTINCT user_id) AS total_user_A_group,
            COUNT(DISTINCT CASE WHEN converted = 1 THEN user_id END) AS count_converted_0,
            ROUND(
                100.0 * COUNT(DISTINCT CASE WHEN converted = 1 THEN user_id END) / NULLIF(COUNT(DISTINCT user_id), 0), 
                2
            ) AS conversion_rate_A_group
        FROM test
        WHERE "group" = 0
    ),
    conversion_B AS (
        SELECT 
            COUNT(DISTINCT user_id) AS total_user_B_group,
            COUNT(DISTINCT CASE WHEN converted = 1 THEN user_id END) AS count_converted_1,
            ROUND(
                100.0 * COUNT(DISTINCT CASE WHEN converted = 1 THEN user_id END) / NULLIF(COUNT(DISTINCT user_id), 0), 
                2
            ) AS conversion_rate_B_group
        FROM test
        WHERE "group" = 1
    )
    SELECT
        total_user_A_group,
        total_user_B_group, 
        conversion_rate_A_group,
        conversion_rate_B_group,
        ROUND(
            100.0 * (conversion_rate_B_group - conversion_rate_A_group) / NULLIF(conversion_rate_A_group, 0), 
            1
        ) AS diff_A_and_B_groups
    FROM conversion_A, conversion_B
""", conn)

conn.close()

# ----- ВЫВОДИМ РЕЗУЛЬТАТ -----
print("\n========================== РЕЗУЛЬТАТЫ A/B-ТЕСТА =============================")
print(df_ab_test)

print(f'Разница: 11.43% - 10.58% = 0.85% (абсолютный прирост)'
      f'\nОтносительный прирост (Lift): 0.85 / 10.58 * 100 = 8%')

#Сделаем визуализацию по разнице в конверсиях в группах
fig, axes = plt.subplots(1, 3, figsize=(17, 6))

# ----- ГРАФИК 1: Сравнение конверсии -----
groups = ['Группа A\n(контроль)', 'Группа B\n(тест)']
conv = [
    df_ab_test['conversion_rate_A_group'].iloc[0],
    df_ab_test['conversion_rate_B_group'].iloc[0]
]
colors = ['royalblue', 'coral']

axes[0].bar(groups, conv, color=colors)
axes[0].set_title('Сравнение конверсии', fontsize=14)
axes[0].set_ylabel('Конверсия (%)', fontsize=12)
axes[0].set_ylim(0, max(conv) * 1.2)
axes[0].grid(axis='y', alpha=0.3)
for i, v in enumerate(conv):
    axes[0].text(i, v + 0.1, f'{v:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

# ----- ГРАФИК 2: Прирост (Lift) -----
diff = conv[1] - conv[0]
lift = (diff / conv[0]) * 100
color = 'seagreen' if lift > 0 else 'red'
axes[1].bar(['Прирост (Lift)'], [lift], color=color)
axes[1].set_title('Относительный прирост', fontsize=14)
axes[1].set_ylabel('Прирост (%)', fontsize=12)
axes[1].grid(axis='y', alpha=0.3)
axes[1].text(6, lift + 0.5, f'{lift:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

# ----- ГРАФИК 3: Количество пользователей -----
n_users = [df_ab_test['total_user_A_group'].iloc[0], df_ab_test['total_user_B_group'].iloc[0]]
axes[2].bar(groups, n_users, color=['#4A7FB5', '#FF8C42'])
axes[2].set_title('Количество пользователей', fontsize=14)
axes[2].set_ylabel('Пользователи', fontsize=12)
axes[2].grid(axis='y', alpha=0.3)
for i, v in enumerate(n_users):
    axes[2].text(i, v + 20, f'{v:,}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.suptitle('A/B-тест: Новый дизайн', fontsize=17)
plt.tight_layout()
plt.show()



#Делаем подсчет на статистическую значимость двух
print('='* 70)
print('Вычисляем статистическую значимость A/B теста на p-value')
from statsmodels.stats.proportion import proportions_ztest

success = [537, 563]
nobs = [5076, 4924]

z_stat, p_value = proportions_ztest(success, nobs)
print(f'p-value = {p_value:.2f}')
if p_value < 0.05:
    print('Результат статистически значим!')
else:
    print('Результат не значим, p-value > 0.05')

print('='* 76)
print('='* 30, 'ПОДВОДИМ ИТОГИ', '=' * 30)

print('1. РЕЗУЛЬТАТЫ:'
      '\n- Группа A (старый дизайн): 10.58%'
      '\n- Группа B (новый дизайн): 11.43%'
      '\n- Прирост: +0.85% (относительный +8%)')

print(f'2. СТАТИСТИЧЕСКАЯ ЗНАЧИМОСТЬ:'
      f'\n- p-value = 0.17 (> 0.05)'
      f'\n- Результат НЕ значим')

print(f'3. ВЫВОД:'
      f'\n- Мы НЕ можем утверждать, что новый дизайн улучшает конверсию'
      f'\n- Разница в 8% может быть случайностью')

print(f'4. РЕКОМЕНДАЦИЯ:'
      f'\n- 🔄 Продолжить тест до набора статистической мощности'
      f'\n- 📈 Увеличить размер выборки до 20 000 пользователей'
      f'\n- 🎯 Сфокусироваться на сегменте (например, мобильные пользователи)')
