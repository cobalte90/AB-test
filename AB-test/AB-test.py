import pandas as pd
import numpy as np
import scipy.stats as sps
from matplotlib import pyplot as plt
# Загружаем данные о транзакциях и принадлежности юзеров к группам
df = pd.read_csv('transactions.csv')
users = pd.read_csv('users.csv')

# Сливаем две таблицы в одну по ключу "id", чтобы видеть группу пользователя на каждой транзакции
df = df.merge(users, on='id')


# Смотрим на количество уникальных транзакций каждого юзера
print(df['id'].describe())
# Количество транзаций совпадает с количеством уникальных имен юзеров
# Следовательно, каждый юзер провел по одной транзакции

# Убедимся в этом, посчитав средние чеки и количество чеков для каждого юзера
grouped = df.groupby(['id'])
ACA = grouped.agg({'amount' : 'mean'})
print(df['amount'].sum() == ACA['amount'].sum())
# Возращает True, значит средний чек каждого юзера равен сумме его чеков

# Посчитаем средний чек для контрольной и тестовой групп
control = df.loc[df['group'] == 'control', 'amount']
treatment = df.loc[df['group'] == 'treatment', 'amount']
AA_control, AA_treatment = np.mean(control), np.mean(treatment)
'''
Средний чек на контрольной группе равен 1471.739368, на тестовой - 1474.884510
Теперь нужно понять, является ли эта разница статистически значимой
'''
# Построим графики наших групп
plt.hist(control, bins=100)
plt.hist(treatment, bins=100)
plt.legend('CT')

# Проверим с помощью теста Шапиро-Уилка, являются ли данные в группах нормально распределенными
stats, p = sps.shapiro(control)
alpha = 0.05 # Уровень значимости
if p > alpha: print('Распределение контрольной группы нормальное')
else: print('Распределение контрольной группы отличается от нормального')

stats, p = sps.shapiro(treatment)
if p > alpha: print('Распределение тестовой группы нормальное')
else: print('Распределение тестовой группы отличается от нормального')

# Посчитаем дисперсии в группах
var_control, var_treatment = np.var(control), np.var(treatment)

# Посчитаем размер каждой группы
n_control, n_treatment = len(control), len(treatment)

# Посчитаем значение t-критерия
t = abs(AA_control - AA_treatment) / ( (var_control**2)/n_control + (var_treatment**2)/n_treatment )**0.5
f = n_control + n_treatment - 2
print(t, f)

# При таком значении t-критерия и таком числе степеней свободы получаем p-value = 0.9999201
# p-value найдено с помощью калькулятора https://calculator-online.net/p-value-calculator/
p_value = 0.9999201

if p_value > alpha:
    print('Разница в группах статистически незначима, отклоняем альтернативную гипотезу')
else:
    print('Разница в группах статистически значима, принимаем альтернативную гипотезу')

plt.show()

# Отклоняем альтернативную гипотезу
# Это значит, что обновления в рекомендательной системе не повлиялия на средний чек пользователей
# Обновление не стоит раскрывать на всех юзеров
# Вот такой АБ-тест получился, спасибо за внимание)