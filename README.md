# AB-test
## Задача:  
Некоторый сервис вводит персонализированную систему рекомендаций товаров.  
Пользователи тестовой группы в течение 7 дней получали обновленные рекомендации.  
С помощью A/B теста необходимо определить, увеличивает ли экспериментальная система суммы средних чеков по сравнению с контрольной группой, а также ответить на вопрос: "следует ли раскатить новую систему на всех пользователей".  
Пользователи просплитованы 50 на 50 случайным образом.  
## Ход решения:  
1. Выгружаю данные из csv таблиц
2. Разделяю данные по группам (контрольная и тестовая)
3. Строю графики, проверяю распределение на нормальность
4. Устанавливаю значение alpha  
5. Считаю дисперсию в каждой группе  
6. Считаю размер каждой группы
7. Считаю значение t-критерия
8. Нахожу значение p-value
9. Отклоняю альтернативную гипотезу  
## Вывод: обновления в рекомендательной системе не повлияли на средний чек пользователей, обновление не стоит раскрывать на всех юзеров
График распределения чеков в двух группах (оранжевый - тестовая группа, синий - контрольная)
![Figure_1](https://github.com/user-attachments/assets/34a7fa38-c8b1-44bf-945b-ff9dce0d78fb)
![image_2025-03-01_22-58-09](https://github.com/user-attachments/assets/5abd2f99-4137-4248-bcf3-5a96bd02caa6)
