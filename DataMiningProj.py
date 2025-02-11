import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.manifold import MDS
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import arff
import time
from sklearn.cluster import KMeans
import os

class NBC_Literature:
    def __init__(self, k):
        self.k = k

    def calculate_distance_matrix(self, data):
        # Compute pairwise Euclidean distances
        return cdist(data, data)

    def calculate_kNB_and_R_kNB(self, distance_matrix):
        n = distance_matrix.shape[0]
        kNB = [set() for _ in range(n)]
        R_kNB = [set() for _ in range(n)]
        for i in range(n):
            neighbors = np.argsort(distance_matrix[i])[1:self.k + 1]
            kNB[i].update(neighbors)
            for neighbor in neighbors:
                R_kNB[neighbor].add(i)
        return kNB, R_kNB

    def calculate_NDF(self, kNB, R_kNB):
        n = len(kNB)
        NDF = np.zeros(n)
        for i in range(n):
            if len(kNB[i]) > 0:
                NDF[i] = len(R_kNB[i]) / len(kNB[i])
        return NDF

    def cluster(self, data):
        # Step 1: Calculate distances, kNB, R_kNB, and NDF
        distance_matrix = self.calculate_distance_matrix(data)
        kNB, R_kNB = self.calculate_kNB_and_R_kNB(distance_matrix)
        NDF = self.calculate_NDF(kNB, R_kNB)

        # Step 2: Clustering based on NDF
        n = data.shape[0]
        cluster_labels = [-1] * n  # Initialize all points as unassigned
        cluster_count = 0  # Start cluster count
        noise_set = set()

        for i in range(n):
            if cluster_labels[i] != -1 or NDF[i] < 1:  # Skip non-DP/EP points
                continue

            # Assign a new cluster
            cluster_count += 1
            cluster_labels[i] = cluster_count
            DP_set = set()

            # Expand cluster with kNB
            for neighbor in kNB[i]:
                cluster_labels[neighbor] = cluster_count
                if NDF[neighbor] >= 1:  # DP or EP
                    DP_set.add(neighbor)

            # Process DP set to expand the cluster
            while DP_set:
                current_point = DP_set.pop()
                for neighbor in kNB[current_point]:
                    if cluster_labels[neighbor] == -1:  # Unassigned
                        cluster_labels[neighbor] = cluster_count
                        if NDF[neighbor] >= 1:  # DP or EP
                            DP_set.add(neighbor)

        # Identify noise points
        for i in range(n):
            if cluster_labels[i] == -1:
                noise_set.add(i)

        return cluster_labels, noise_set

def load_arff_file(filepath):
    with open(filepath, 'r') as f:
        dataset = arff.load(f)
    data = np.array(dataset['data'])
    return data

def save_plot(data, labels, method, filename):
    unique_labels = set(labels)
    plt.figure(figsize=(10, 6))
    for label in unique_labels:
        if label == -1:
            plt.scatter(data[np.array(labels) == label, 0], data[np.array(labels) == label, 1], c='k', alpha=0.6, label="Noise")
        else:
            plt.scatter(data[np.array(labels) == label, 0], data[np.array(labels) == label, 1], alpha=0.6, label=f"Cluster {label}")

    plt.title(f"{method} Clustering")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    if not os.path.exists("plots/kmeans"):
        os.makedirs("plots/kmeans")
    plt.savefig(f"plots/kmeans/{filename}")
    plt.close()

def main():
    # Load data
    arff_file = "R15.arff"  # Replace with your ARFF file path
    data = load_arff_file(arff_file)
    data = data[:, :-1]
    
    # Normalize the data
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    # K-Means clustering and NBC clustering comparison
    k_values = range(2, 20)
    results = []

    # NBC clustering
    for k in k_values:
        nbc_lit = NBC_Literature(k=k)
        start_time = time.time()
        labels_nbc, noise_nbc = nbc_lit.cluster(data)
        end_time = time.time()
        elapsed_time_nbc = end_time - start_time
        print(f"Clustering execution time with NBC (k={k}): {elapsed_time_nbc:.4f} seconds")

        # Silhouette Score calculation
        silhouette_avg_nbc = silhouette_score(data, labels_nbc)
        print(f"Silhouette Score for NBC k={k}: {silhouette_avg_nbc:.4f}")
        results.append(('NBC', k, elapsed_time_nbc, silhouette_avg_nbc))

        # MDS for NBC visualization
        print("Running MDS for NBC...")
        distance_matrix = cdist(data, data)
        mds = MDS(n_components=2, random_state=42, dissimilarity='precomputed')
        data_mds = mds.fit_transform(distance_matrix)
        data_mds = np.dot(data_mds, np.array([[0, -1], [1, 0]]))
        unique_labels = set(labels_nbc)
        plt.figure(figsize=(10, 6))
        for label in unique_labels:
            if label == -1:
                plt.scatter(data_mds[np.array(labels_nbc) == label, 0], data_mds[np.array(labels_nbc) == label, 1], c='k', alpha=0.6, label="Noise")
            else:
                plt.scatter(data_mds[np.array(labels_nbc) == label, 0], data_mds[np.array(labels_nbc) == label, 1], alpha=0.6, label=f"Cluster {label}")

        plt.title(f"NBC Clustering (k={k})")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.savefig(f"plots/nbc/NBC_R15_k{k}.png")

    # K-Means clustering
    for k in k_values:
        start_time = time.time()
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels_kmeans = kmeans.fit_predict(data)
        end_time = time.time()
        elapsed_time_kmeans = end_time - start_time
        print(f"Clustering execution time with K-Means (k={k}): {elapsed_time_kmeans:.4f} seconds")

        # Silhouette Score calculation
        silhouette_avg_kmeans = silhouette_score(data, labels_kmeans)
        print(f"Silhouette Score for K-Means k={k}: {silhouette_avg_kmeans:.4f}")
        results.append(('K-Means', k, elapsed_time_kmeans, silhouette_avg_kmeans))

        # MDS for K-Means visualization
        print("Running MDS for K-Means...")
        distance_matrix = cdist(data, data)
        mds = MDS(n_components=2, random_state=42, dissimilarity='precomputed')
        data_mds = mds.fit_transform(distance_matrix)
        data_mds = np.dot(data_mds, np.array([[0, -1], [1, 0]]))
        save_plot(data_mds, labels_kmeans, f"K-Means Clustering (k={k})", f"KMeans_R15_k{k}.png")
    
    # Save results (Silhouette scores and execution times)
    if not os.path.exists("results"):
        os.makedirs("results")
    with open("results/clustering_results_comparison.txt", "w") as f:
        f.write("Method,k,Execution Time (s),Silhouette Score\n")
        for result in results:
            f.write(f"{result[0]},{result[1]},{result[2]:.4f},{result[3]:.4f}\n")


if __name__ == "__main__":
    main()
