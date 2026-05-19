# Рекомендательная система музыки: методы решения проблемы холодного старта

Данный проект реализует рекомендательную систему музыки с помощью текстовых и числовых признаков трека

## Описание проекта

В проекте осуществлены 2 основные системы рекомендаций:
1. Классический подход и использованием **матрицы косинусного сходства** и **TF-IDF**
2. Система **на основе популярности** с параметром схожести

## Структура проекта

music_recommendation.py - основной код системы рекомендаций

data.csv - датасет c данными о музыке (см. ниже)

cluster_analysis.py - дополнительный код для анализа кластеризации

DBSCAN.png, Elbow Method.png, Optimal K.png - результаты анализа кластеров


## Описание датасета

Для работы системы необходим датасет в формате CSV с следующими столбцами:

name - название трека (str)

artists - исполнители (str формата list)

release_date - дата релиза - (str, в формате yyyy / yyyy-mm-dd)

valence - настроение трека (float, 0.0 - 1.0)

year - год релиза (int)

acousticness - уровень акустичности (0.0 - 1.0)

danceability - танцевальность (float, 0.0 - 1.0)

duration_ms - продолжительность в миллисекундах (int)

energy - энергия трека (float, 0.0 - 1.0)

explicit - наличие нецензурной лексики / провокационного контента (bool, 0 или 1)

instrumentalness - наличие инструментальной музыки (float, 0.0 - 1.0)

key - тональность трека по музыкальной нотации Camelot Wheel (int, 0 - 11)

liveness - живое исполнение (float, 0.0 - 1.0)

loudness - уровень громкости (float, -60 - 3.85)

mode - лад (bool, 0 - минор или 1 - мажор)

popularity - популярность трека (int, 0 - 100)

speechiness - насколько трек состоит из разговорной речи (float, 0.0 - 1.0)

tempo - темп в BPM (float, 0 - 244)

*Ссылка на исходный датасет из Kaggle: https://www.kaggle.com/code/vatsalmavani/music-recommendation-system-using-spotify-dataset/input*


## Установка зависимостей

```bash
pip install pandas numpy scikit-learn matplotlib
```


## Запуск программы
macOS / Linux
```python
python3 music_recommendation.py
```

Windows
```python
python music_recommendation.py
```
### Анализ кластеризации музыки
```python
python3 cluster_analysis.py
```

