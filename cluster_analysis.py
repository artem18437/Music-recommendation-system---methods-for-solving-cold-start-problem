import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv('data.csv')

features = ['valence', 'danceability', 'energy', 'acousticness', 'instrumentalness',
            'liveness', 'loudness', 'speechiness', 'tempo']

X = df[features].dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Метод локтя
def elbow_method(X, max_k=10):

    wcss = []
    k_range = range(1, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)  # inertia_ gives WCSS
    
    return k_range, wcss

#Вычисляет сумму квадратов расстояний от точек до центров своих кластеров, 
#кол-во которых k = 1, 2, ..., 10
k_range, wcss = elbow_method(X_scaled, max_k=10)

# Plot the elbow curve
plt.figure(figsize=(10, 6))
plt.plot(k_range, wcss, 'bo-')
plt.xlabel('Количество кластеров')
plt.ylabel('Сумма квадратов расстояния от точек кластера до его центроида')
plt.title('Метод локтя для оптимального кол-ва кластеров')
plt.grid(True)
plt.show()

#Вычисляет искомую точку методом второй производной
def find_elbow_point(wcss):
    """
    Find elbow point using second derivative method
    """
    # Calculate second differences
    second_diff = np.diff(wcss, 2)
    
    # Find the index where second difference is maximum (elbow point)
    elbow_index = np.argmax(second_diff) + 2  # +2 because of diff(2)
    
    return elbow_index


optimal_k = find_elbow_point(wcss)
print(f"Optimal number of clusters (elbow point): {optimal_k}")

#График
plt.figure(figsize=(10, 6))
plt.plot(k_range, wcss, 'bo-', linewidth=2, markersize=8)
plt.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, 
          label=f'Elbow point: k={optimal_k}')
plt.xlabel('Количество кластеров')
plt.ylabel('Сумма квадратов расстояния от точек кластера до его центроида')
plt.title('Метод локтя для оптимального кол-ва кластеров')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"\nОптимальный k = {optimal_k}:")
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(X_scaled)

df['cluster'] = -1
df.loc[X.index, 'cluster'] = cluster_labels

print("Кластеризация завершена, всего", optimal_k, "кластеров")




#Кластеризация DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(X_scaled)

#Добавление меток кластеров в DataFrame
df['cluster'] = -1  # По умолчанию - шум
df.loc[X.index, 'cluster'] = clusters

#Визуализация: Распределение кластеров
def visualize_cluster_distribution(df):
    plt.figure(figsize=(12, 6))
    
    #Гистограмма распределения кластеров
    plt.subplot(1, 2, 1)
    cluster_counts = df['cluster'].value_counts().sort_index()
    plt.bar(cluster_counts.index, cluster_counts.values)
    plt.xlabel('Номер кластера')
    plt.ylabel('Количество треков')
    plt.title('Распределение треков по кластерам')
    
    #Показать количество шума
    noise_count = len(df[df['cluster'] == -1])
    cluster_count = len(cluster_counts) - 1  # без шума
    
    plt.subplot(1, 2, 2)
    labels = ['Кластеры', 'Шум']
    sizes = [cluster_count, noise_count]
    colors = ['lightblue', 'lightcoral']
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
    plt.title('Соотношение кластеров и шума')
    
    plt.tight_layout()
    plt.show()



#Запуск визуализаций
print("Создание визуализаций...")
visualize_cluster_distribution(df)


print("Количество треков в каждом кластере (value_counts):")
cluster_counts = df['cluster'].value_counts().sort_values(ascending=False)
print(cluster_counts)

#Визуализация статистики кластеров
print("\nСтатистика по кластерам:")
cluster_stats = df.groupby('cluster').agg({
    'valence': ['mean', 'std'],
    'danceability': 'mean',
    'energy': 'mean',
    'acousticness': 'mean'
}).round(3)

print(cluster_stats)
