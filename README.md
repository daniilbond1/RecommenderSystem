# Рекомендательная система новостей

## Постановка задачи

Построить рекомендательную модель, обучить используя  представленный дата-сет и вывести топ 10 рекомендации фильмов, которые пользователь еще не видел. 
Нужно так же учитывать название фильмов при рекомендации. 
Решение предоставить в виде python скрипта/ов или jupyter notebooks (построение модели или использование существующей, обучение и вывод топ 10 рекомендации для пользователя)
Так же в качестве решения рассмотрим текстовое описание подхода к решению. 
Описание должно содержать описание алгоритмов, список используемых технологий и инструментов и принцип работы. 
Так же необходимо аргументировать свой выбор.

## Содержимое

- [movies.csv](./data/movies.csv) и [ratings.csv](./data/ratings.csv) -- использованные части исходного датасета;
- [Research.ipynb](./notebooks/Research.ipynb) -- `jupyter notebook`, в котором проверялись основные подходы к построению рекомендательных систем;
- [Classes.ipynb](./notebooks/Classes.ipynb) -- `jupyter notebook`, содержащий в себе финальные алгоритмы, обернутые в классы с методами `.fit()` и `.predict()`, а так же уникальными методами. 

## Ход работы

Было решено реализовать 3 принципиально различных подхода: 
- Collaborative filtering;
- Content based filtering;
- Popularity based filtering.
У первых двух подходов есть проблема холодного старта, поэтому для новых пользователей вклад Popularity based filtering должен быть больше, чем у других двух подходов, которые проявляют себя хорошо при наличии большого количества данных о пользователе. 

### Popularity based filtering
 
- Это самый очевидный подход, он рекомендует каждому пользователю самые популярные фильмы. Для его реализации было необходимо лишь вычислить вес каждого фильма и рекомендовать самые популярные. 

### Content based filtering

- Этот подход основан на векторизации всех итемов. В данном случае было решено воспользоваться `TF-IDF` векторизацей для названий фильмов. Далее вычисляется вектор юзера и находится наиболее близкие к нему вектора итемов.


### Collaborative filtering

- Первым делом, было решено реализовать baseline, то есть создать наивный алгоритм, точность которого в последствии мы будем улучшать. Таким baseline стал алгоритм, предсказывающий среднюю оценку по каждому фильму; 
- Первым испробованым алгоритмом был поиск наиболее похожих пользователей. Была создана матрица `user_item`, заполненная рейтингами, которые поставил `i`-й юзер `j`-му фильму. Затем было подсчитано косинусное расстояние между векторами каждого юзера, далее находится среднее между оценками похожих юзеров на каждый не просмотренный фильм у данного пользователя. Также была исследована зависимость точности от количества учтенных наиболее похожих юзеров;
- Второй алгоритм предсказывыал рейтинг, основывавыясь на среднем рейтинге конкретного пользователя, а так же на среднем рейтинге похожих пользователей. Сам алгоритм крайне похож на предыдущий, однако, по сути, теперь мы предсказываем отклонение человека от его средней оценки, что сильно помогло увеличить точность алгоритма;
- Третьим алгоритмом было сингулярное разложение матрицы или `SVD`. Данное разложение представляет матрицу как произведение трех матриц `U`, `S`, `V`, где `U` -- ортогональная левая сингулярная матрица, которая описывает взаимосвязь между пользователями и неявными факторами, `S` -- диагональная матрица, которая описывает вес каждого неявного фактора, а `V` -- диагональная правая сингулярная матрица, которая указывает на сходство между элементами и невными факторами. Так же была взята реализация этого метода из бибилиотеки `surprise`. Это библиотека, содержащая все методы коллаборативной фильтрации, а так же вспомогательные методы для выбора гиперпараметров, методов и т.д.