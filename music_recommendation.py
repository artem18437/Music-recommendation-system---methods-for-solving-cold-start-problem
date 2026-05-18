import sys
import warnings
warnings.filterwarnings('ignore')


import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans



#Путь до датасета
def_path = 'data.csv'


class MusicRecommendationSystem:
    def __init__(self, data_path=def_path, data_frame=None): #датафрейм для первых тестов с явным результатом
        """
        Инициализация системы рекомендаций
        """
        
        if data_path:
            self.df = pd.read_csv(data_path, encoding='utf-8')
        elif data_frame is not None:
            self.df = data_frame
        else:
            
            self.df = None
    
        
        self.scaler = StandardScaler()
        self.feature_matrix = None
        self._prepare_features()
        
    
    def _prepare_features(self):
        """
        Подготовка признаков для машинного обучения
        """
        #Обработка текстовых данных в artists
        self.df['artists_clean'] = self.df['artists'].str.replace("[\[\]' ]", '', regex=True)
        
        #Создание текстового признака для TF-IDF
        self.df['text_features'] = (
            self.df['name'].fillna('') + ' ' + 
            self.df['artists_clean'].fillna('') + ' ' +
            self.df['release_date'].fillna('')
        )
        
        #Создание матрицы признаков
        numeric_features = ['valence', 'year', 'acousticness', 'danceability', 
                           'duration_ms', 'energy', 'explicit', 'instrumentalness', 
                           'key', 'liveness', 'loudness', 'mode', 'popularity', 
                           'speechiness', 'tempo']
        
        #Стандартизация числовых признаков
        self.feature_matrix = self.df[numeric_features].copy()
        self.feature_matrix_scaled = self.scaler.fit_transform(self.feature_matrix)
        
        #TF-IDF векторизация текстовых признаков
        tfidf = TfidfVectorizer(max_features=4444, stop_words='english')
        self.text_features = tfidf.fit_transform(self.df['text_features'])
        
        #Объединение всех признаков
        self.combined_features = np.hstack([
            self.feature_matrix_scaled,
            self.text_features.toarray()
        ])
        
    def get_similar_tracks(self, playlist_indices, n_recommendations=10):
        """
        Поиск похожих треков для заданного плейлиста
        
        Parameters:
        playlist_indices: список индексов треков
        n_recommendations: количество рекомендаций
        
        Returns:
        DataFrame с рекомендациями
        """
            
        if not playlist_indices:
            raise ValueError("Не найдено треков в плейлисте")
            
        #Среднее значение признаков для плейлиста
        playlist_features = self.combined_features[playlist_indices]
        playlist_avg = np.mean(playlist_features, axis=0).reshape(1, -1)
        
        #Вычисление сходства с остальными треками
        similarities = cosine_similarity(playlist_avg, self.combined_features)[0]
        
        #Исключить треки из плейлиста
        similarities[playlist_indices] = -1
        
        #Получить индексы самых похожих треков
        top_indices = np.argsort(similarities)[::-1][:n_recommendations]
        
        #Вернуть рекомендации
        recommendations = self.df.iloc[top_indices].copy()
        recommendations['similarity_score'] = similarities[top_indices]
        
        return recommendations[['name', 'artists', 'similarity_score']].reset_index(drop=True)
    
    def get_cluster_recommendations(self, playlist_indices, n_clusters=5, n_recommendations=10):
        """
        Рекомендации с использованием кластеризации
        
        Parameters:
        playlist_tracks: список названий треков или их индексов
        n_clusters: количество кластеров для кластеризации
        n_recommendations: количество рекомендаций
        
        Returns:
        DataFrame с рекомендациями
        """
        #Кластеризация всех треков
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(self.combined_features)
        self.df['cluster'] = cluster_labels

        
        #Получить кластеры для плейлиста
        playlist_clusters = set(self.df.loc[playlist_indices, 'cluster'])
        
        #Найти треки из тех же кластеров
        cluster_mask = self.df['cluster'].isin(playlist_clusters)
        
        #Исключить треки из плейлиста
        playlist_mask = self.df.index.isin(playlist_indices)
        final_mask = cluster_mask & ~playlist_mask
        
        #Получить рекомендации
        recommendations = self.df[final_mask].copy()
        recommendations['similarity_score'] = np.random.rand(len(recommendations))  #Для демонстрации
        
        return recommendations[['name', 'artists', 'similarity_score']].head(n_recommendations).reset_index(drop=True)
    
    def get_playlist_similarity(self, playlist1, playlist2):
        """
        Оценка схожести двух плейлистов
        
        Parameters:
        playlist1, playlist2: списки названий треков
        
        Returns:
        float: степень схожести (0-1)
        """
        if isinstance(playlist1[0], str):
            indices1 = self.df[self.df['name'].isin(playlist1)].index.tolist()
        else:
            indices1 = playlist1
            
        if isinstance(playlist2[0], str):
            indices2 = self.df[self.df['name'].isin(playlist2)].index.tolist()
        else:
            indices2 = playlist2
            
        if not indices1 or not indices2:
            return 0.0
            
        #Средние признаки для каждого плейлиста
        features1 = self.combined_features[indices1]
        features2 = self.combined_features[indices2]
        
        avg1 = np.mean(features1, axis=0).reshape(1, -1)
        avg2 = np.mean(features2, axis=0).reshape(1, -1)
        
        #Косинусное сходство
        similarity = cosine_similarity(avg1, avg2)[0][0]
        return max(0, min(1, similarity))  #Ограничиваем от 0 до 1



class SimilarityBasedPopularityRecommendation:
    def __init__(self, data_path=def_path, data_frame=None):
        """
        Инициализация рекомендательной системы на основе популярности с параметром схожести
        """
        if data_path:
            self.df = pd.read_csv(data_path, encoding='utf-8')
        elif data_frame is not None:
            self.df = data_frame
        else:
            self.df = None
        
        #Проверка наличия необходимых столбцов
        required_columns = ['name', 'artists', 'popularity', 'danceability', 'energy', 'valence']
        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"Необходимый столбец '{col}' отсутствует в данных")
        
        print(f"Загружено {len(self.df)} треков для рекомендаций")
        
        #Подготовка числовых признаков для схожести
        self.numeric_features = ['danceability', 'energy', 'valence']
        self.scaler = StandardScaler()
        
        #Стандартизация числовых признаков
        self.feature_matrix = self.df[self.numeric_features].copy()
        self.feature_matrix_scaled = self.scaler.fit_transform(self.feature_matrix)
        
    
    def get_similarity_based_recommendations(self, playlist_indices, similarity_threshold=0.7, n_recommendations=10):
        """
        Рекомендации самых популярных треков с учетом параметра схожести
        
        Parameters:
        playlist_indices: список индексов треков
        similarity_threshold: порог схожести (0.0 - 1.0), чем выше, тем строже критерий
        n_recommendations: количество рекомендаций
        
        Returns:
        DataFrame с рекомендациями
        """
            
        if not playlist_indices:
            raise ValueError("Не найдено треков в плейлисте")
            
        #Исключить треки из плейлиста
        mask = ~self.df.index.isin(playlist_indices)
        
        #Получить доступные треки
        available_tracks = self.df[mask]
        
        if len(available_tracks) == 0:
            raise ValueError("Нет доступных треков для рекомендаций")
        
        #Получить признаки плейлиста
        playlist_features = self.feature_matrix_scaled[playlist_indices]
        
        #Вычислить средние признаки плейлиста
        playlist_avg_features = np.mean(playlist_features, axis=0).reshape(1, -1)
        
        #Вычислить схожесть для всех доступных треков
        available_features = self.feature_matrix_scaled[mask]
        similarities = cosine_similarity(playlist_avg_features, available_features)[0]
        
        #Применить порог схожести
        similarity_mask = similarities >= similarity_threshold
        
        #Отфильтровать треки по схожести
        filtered_tracks = available_tracks[similarity_mask].copy()
        filtered_tracks['similarity'] = similarities[similarity_mask]
        
        #Сортировка по популярности (в первую очередь) и затем по схожести
        filtered_tracks = filtered_tracks.sort_values(['popularity', 'similarity'], ascending=[False, False])
        
        #Ограничить количество рекомендаций
        recommendations = filtered_tracks.head(n_recommendations)
        
        #Возвращаем результат с нужными столбцами
        return recommendations[['name', 'artists', 'popularity', 'similarity']].reset_index(drop=True)
    
    def get_top_popular_similar_tracks(self, playlist_tracks, similarity_threshold=0.7, n_recommendations=10):
        """
        Получение самых популярных треков с учетом схожести
        
        Parameters:
        playlist_tracks: список названий треков или их индексов
        similarity_threshold: порог схожести (0.0 - 1.0)
        n_recommendations: количество рекомендаций
        
        Returns:
        DataFrame с рекомендациями
        """
        recommendations = self.get_similarity_based_recommendations(
            playlist_tracks, 
            similarity_threshold, 
            n_recommendations * 2  #Получаем больше для фильтрации
        )
        
        #Если получили меньше рекомендаций, возвращаем все доступные
        if len(recommendations) < n_recommendations:
            return recommendations
        
        #Сортируем по популярности и возвращаем топ N
        top_recommendations = recommendations.sort_values('popularity', ascending=False).head(n_recommendations)
        
        return top_recommendations[['name', 'artists', 'popularity', 'similarity']].reset_index(drop=True)
    





def find_track_by_name(dataset, track_name):
    """
    Функция поиска трека по названию в датасете
    
    Parameters:
    dataset: DataFrame с данными о треках
    track_name: строка с названием трека для поиска
    
    Returns:
    Информация о найденном треке или список треков с похожими названиями
    """
    
    #Поиск треков, содержащих в названии заданную строку (регистронезависимо)
    matching_tracks = dataset[dataset['name'].str.lower() == track_name.lower()]
    
    if len(matching_tracks) == 0:
        print(f"Трек с названием '{track_name}' не найден в датасете")
        return None
    
    elif len(matching_tracks) == 1:
        #Найден один трек
        track_info = matching_tracks.iloc[0]
        print(f"Найден трек: {track_info['name']}")
        print(f"Исполнитель: {track_info['artists']}")
        print(f"Год: {track_info['year']}")
        return dataset[dataset['id'] == track_info['id']].index[0]
    
    else:
        #Найдено несколько треков с похожими названиями
        print(f"Найдено {len(matching_tracks)} треков с похожими названиями\nПожалуйста, укажите один из следующих:")
        for i, (_, track) in enumerate(matching_tracks.iterrows(), 1):
            print(f"{i}. {track['name']} - {track['artists']}")
        
        #Запрос выбора пользователя
        while True:
            try:
                choice = int(input(f"Выберите трек (1-{len(matching_tracks)}): "))
                if 1 <= choice <= len(matching_tracks):
                    selected_track = matching_tracks.iloc[choice-1]
                    print(f"Выбран трек: {selected_track['name']}")
                    print(f"Исполнитель: {selected_track['artists']}")
                    print(f"Год: {selected_track['year']}")
                    return dataset[dataset['id'] == selected_track['id']].index[0]
                else:
                    print(f"Пожалуйста, введите число от 1 до {len(matching_tracks)}")
            except ValueError:
                print("Пожалуйста, введите корректное число")




#Пример использования системы


#Создание экземпляра системы рекомендаций

recommender_classic = MusicRecommendationSystem()

recommender_popularity = SimilarityBasedPopularityRecommendation()

#Загрузка датасета
df = pd.read_csv(def_path)

#Пример плейлиста пользователя (названия треков)

n = int(input("Введите количество треков в плейлисте: "))

while n < 1 or n > 100:
    print("Рекомендательная система поддерживает от 1 до 100 треков в плейлисте включительно")
    n = int(input("Введите количество треков в плейлисте: "))

user_playlist_input = [input(f"Введите название #{_ + 1} трека:") for _ in range(n)]

user_playlist = []

for i in range(len(user_playlist_input)):
    
    ind = find_track_by_name(df, user_playlist_input[i])

    if ind is None:
        sys.exit("Пожалуйста, измените название песни в плейлисте и попробуйте снова")
    else:
        user_playlist.append(ind)

#Кол-во рекомендаций

number_of_recoms = int(input("Введите количество желаемых рекомендаций: "))
while number_of_recoms < 1 or number_of_recoms > 100:
    print("Для наиболее релевантных результатов предполагается количество от 1 до 100 рекомендаций включительно")
    number_of_recoms = int(input("Введите количество желаемых рекомендаций: "))

print("=== Рекомендательная система музыки ===\n")

#Получение рекомендаций
print("Рекомендации на основе плейлиста:")
recommendations = recommender_classic.get_similar_tracks(user_playlist, n_recommendations=number_of_recoms)

for i, (_, rec) in enumerate(recommendations.iterrows(), 1):
    print(f"{i}. {rec['name']} - {rec['artists']}")
    print(f"Схожесть: {rec['similarity_score']:.3f}\n")

#Пример кластеризации
print("Рекомендации через кластеризацию:")
cluster_recommendations = recommender_classic.get_cluster_recommendations(user_playlist, n_recommendations=number_of_recoms)

for i, (_, rec) in enumerate(cluster_recommendations.iterrows(), 1):
    print(f"{i}. {rec['name']} - {rec['artists']}\n")


print("\n\n=== Рекомендательная система по популярности с параметром схожести ===\n")


#Рекомендации с низким порогом схожести
print("Рекомендации (низкий порог схожести = 0.3):")
recommendations_low = recommender_popularity.get_similarity_based_recommendations(
    user_playlist, 
    similarity_threshold=0.3, 
    n_recommendations=number_of_recoms
)

for i, (_, rec) in enumerate(recommendations_low.iterrows(), 1):
    print(f"{i}. {rec['name']} - {rec['artists']}")
    print(f"Популярность: {rec['popularity']}\n")