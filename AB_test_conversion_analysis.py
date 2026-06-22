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
# axes[1].text(6, lift + 0.5, f'{lift:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

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
fig.savefig('images/conversion_comparison.png', dpi=300, bbox_inches='tight')


print('\n')
#Делаем подсчет на статистическую значимость двух
print('='* 70)
print('              ВЫЧИСЛЕНИЕ СТАТИСТИЧЕСКОЙ ЗНАЧИМОСТИ A/B ТЕСТА')
print('='* 70)
from statsmodels.stats.proportion import proportions_ztest

success = [537, 563]
nobs = [5076, 4924]

z_stat, p_value = proportions_ztest(success, nobs)
print(f'p-value = {p_value:.2f}')
if p_value < 0.05:
    print('Результат статистически значим!')
else:
    print('Результат не значим, p-value > 0.05')

print('\n')
#Проверим правильность sample size теста
from statsmodels.stats.proportion import proportion_effectsize
from statsmodels.stats.power import NormalIndPower

#MDE 1%
base_conversion = 0.1058 #базовая конверсия
MDE = base_conversion * 0.1
p2 = base_conversion + MDE

effect_size = proportion_effectsize(base_conversion, p2)
#Уровень значимости базовая 5%
alpha = 0.05

#Статистическая мощность базовая 80%
power = 0.80

#Производим расчет
analysis = NormalIndPower()

sample_size_per_group = analysis.solve_power(
    effect_size=effect_size,
    nobs1=None,
    alpha=alpha,
    power=power,
    ratio=1.0,
    alternative='two-sided'
)

#Расчеты и выводы по размеру теста
print('=' * 70)
print('               РАСЧЕТ РАЗМЕРА ВЫБОРКИ ДЛЯ A/B ТЕСТА')
print('=' * 70)
print(f'Базовая конверсия: {base_conversion*100:.2f}%')
print(f'Уровень значимости (alpha): {alpha*100:.0f}%')
print(f'Статистическая мощность (power): {power*100:.0f}%')
print(f'Минимальный эффект (MDE): {MDE*100:.0f} п.п.')
print(f'Необходимый размер выборки (на одну группу): {int(sample_size_per_group):,} пользователей')
print(f'Необходимый размер всей выборки (обе группы): {int(sample_size_per_group*2):,} пользователей')
print('=' * 70)
print('Получаем вывод того, что изначально был неправильно рассчитан размер выборки по кол-ву пользователей')
print(f'В нашем тесте {len(df):,} пользователей на обе группы')
print(f'Необходимо пользователей - {int(sample_size_per_group*2):,}')
print(f'⚠️Проходим к тому, что тесту не хватило {int(sample_size_per_group*2) - len(df):,} пользователя, для обнаружения эффекта {MDE*100:.1f}% ')

#Сделаем визуализацию по различию количества пользователей того, сколько не хватило
x_show = ['Кол-во пользователей\nв тесте', 'Сколько должно быть\nв тесте']
a = len(df)
b = int(sample_size_per_group*2)
sample = [a, b]
diff = b - a
#График по sample size
plt.figure(figsize=(11, 7))
colors = ['red', 'green']
plt.bar(x_show, sample, color=colors)
plt.title('Сравнение кол-ва пользователей в тесте', fontsize=20)
plt.ylabel('Пользователи', fontsize=11)
plt.xticks(fontsize=11)
plt.grid(True, alpha=0.3)
for i, v in enumerate(sample):
    plt.text(i, v + 200, f'{v:,}', ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.text(0, (a + b)/2, f'⚠️ НЕ ХВАТАЕТ {diff:,} пользователей',
         ha='center', va='center', fontsize=12, fontweight='bold', color='#E74C3C',
         bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='red', alpha=0.9))
plt.savefig('images/sample_size_comparison.png', dpi=300, bbox_inches='tight')
plt.show()


#Расчет 95% доверительного интервала для подтверждения статистической не значимость p-value
#Конверсия (кол-во покупок)
print('\n')
print('=' * 70)
print('              ПОДСЧЕТ 95% ДОВЕРИТЕЛЬНОГО ИНТЕРВАЛА')
print('=' * 70)
conversions_A = 537 #Группа А
conversions_B = 563 #Группа B

#Размеры групп
users_A = 5076
users_B = 4924

#Конверсия в процентах
conv_A = conversions_A / users_A
conv_B = conversions_B / users_B

#Разница между конверсиями по группам (lift)
diff_conversion = conv_B - conv_A

#Стандартная ошибка (SE) разницы
SE = np.sqrt(
    (conv_A * (1 - conv_A) / users_A) +
    (conv_B * (1 - conv_B) / users_B)
)

# Z-значение для 95% доверительного интервала
z = 1.96

# CI - доверительный интервал, подсчет интервала идет по такой формуле, потому что у нас даны доли(конверсия) и тут другая формула, чем при обычном нормальном распределении (пример, средний чек)
ci_lower = diff_conversion - z * SE
ci_upper = diff_conversion + z * SE

print(f'95% доверительный интервал в диапазоне: [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]')

if ci_lower < 0 and ci_upper > 0:
    print("\n⚠️ Интервал содержит 0 → разница НЕ статистически значима")
    print("Это согласуется с p-value = 0.17 (> 0.05)")
else:
    print("\n✅ Интервал НЕ содержит 0 → разница статистически значима")
    print("Новый дизайн имеет эффект")

print(f'✅Получаем результат, что мы уверены на 95%, что истиннная разница конверсий'
      f'\nмежду группами А и B находится между -0.37% и +2.08%')
print(f'⚠️Интервал включает 0%, что говорит о том, что разница статистически'
      f'\nне значима (согласуется с p-value = 0.17 (> 0.05)')
print(f'📉Наихудший сценарий: новый дизайн проигрывает (-0.37%)'
      f'\n📈Наилучший сценарий: новый дизайн выигрывает (+2.08%)')


# Создаём фигуру
fig, ax = plt.subplots(figsize=(11, 5))

# 1. Рисуем доверительный интервал (горизонтальная линия)
ax.hlines(y=0, xmin=ci_lower*100, xmax=ci_upper*100,
          color='royalblue', linewidth=8, label='95% Доверительный интервал')

# 2. Рисуем точечную оценку (ромбик в центре)
ax.scatter(diff_conversion*100, 0, color='darkblue', s=150,
           marker='D', zorder=5, label=f'Точечная оценка: {diff_conversion*100:.2f} п.п.')

# 3. Добавляем вертикальную линию в 0 (нулевой эффект)
ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Нулевой эффект (0)')

# 4. Настройки графика
ax.set_title('Разница конверсий (Группа B - Группа A) с 95% доверительным интервалом', fontsize=16)
ax.set_xlabel('Разница в процентных пунктах', fontsize=12)
ax.set_yticks([])  # Убираем вертикальные метки
ax.set_ylim(-0.5, 0.5)
# 5. Добавляем подписи значений
# Подпись для точечной оценки
ax.text(diff_conversion*100, 0.15, f'{diff_conversion*100:.2f} п.п.',
        ha='center', va='bottom', fontsize=12, fontweight='bold', color='darkblue')

# Подписи для границ интервала
ax.text(ci_lower*100, -0.15, f'{ci_lower*100:.2f}%',
        ha='center', va='top', fontsize=10, color='royalblue')
ax.text(ci_upper*100, -0.15, f'{ci_upper*100:.2f}%',
        ha='center', va='top', fontsize=10, color='royalblue')

# 6. Добавляем текстовый комментарий о значимости
ax.text(0, -0.35, '⚠️ Интервал пересекает 0 → разница НЕ статистически значима',
        ha='center', va='center', fontsize=10, color='red', weight='bold',
        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='red', alpha=0.9))

# 7. Легенда и сетка
ax.legend(loc='upper right')
ax.grid(axis='x', alpha=0.3)

# 8. Сохраняем и показываем
fig.savefig('images/confidence_interval_plot.png', dpi=300, bbox_inches='tight')
plt.show()

print('\n'
      '\n')

#Подведем итоги по рекомандациям A/B теста
print('='* 76)
print('='* 30, 'ПОДВОДИМ ИТОГИ', '=' * 30)

print('1. ✅РЕЗУЛЬТАТЫ:'
      '\n- 📊Группа A (старый дизайн): 10.58%'
      '\n- 📊Группа B (новый дизайн): 11.43%'
      '\n- 📈Прирост: +0.85% (относительный +8%)')

print(f'2. 🔍СТАТИСТИЧЕСКАЯ ЗНАЧИМОСТЬ:'
      f'\n- ⚠️p-value = 0.17 (> 0.05)'
      f'\n- ❌Результат НЕ значим')

print(f'3. 📌ВЫВОД:'
      f'\n- 📋Мы НЕ можем утверждать, что новый дизайн улучшает конверсию'
      f'\n- 🎲Разница в 8% может быть случайностью')

print(f'4. 💡РЕКОМЕНДАЦИЯ:'
      f'\n- 🔄Продлить тест: Набрать выборку до 27 000 пользователей, чтобы получить статистически значимые результаты.'
      f'\n- 🎯Сфокусироваться на сегменте (например, мобильные пользователи)')

