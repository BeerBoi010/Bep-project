import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from version_3_feature_matrix_forest import X_train,clf


import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming X_train is your feature matrix with shape (num_samples, num_features)

# Perform PCA to reduce dimensionality
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)

# Determine the number of clusters (you can use different methods to find the optimal number of clusters)
# Here, we're just using a fixed number of clusters for demonstration
num_clusters = 4

# Apply K-means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X_train_pca)
cluster_labels = kmeans.labels_

# Add cluster labels as a new column to X_train
X_train_with_clusters = np.column_stack((X_train, cluster_labels))

# Apply PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_train_with_clusters)

# Visualization using the first two principal components
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], hue=cluster_labels, palette='viridis', alpha=0.7, s=50)
plt.title('Clusters Visualized Using PCA Two-Dimensional Space')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster')
plt.show()


# # Explained variance ratio
# print("Explained variance ratio:", pca.explained_variance_ratio_)

# # Principal components
# print("Principal components:")
# print(pca.components_)

# Calculate cluster counts
cluster_counts = np.bincount(cluster_labels)
print("Cluster Counts:")
print(cluster_counts)

# Calculate cluster-wise feature means
cluster_analysis = np.zeros((num_clusters, X_train.shape[1]))
for cluster_idx in range(num_clusters):
    cluster_data = X_train_with_clusters[X_train_with_clusters[:, -1] == cluster_idx, :-1]
    cluster_mean = np.mean(cluster_data, axis=0)
    cluster_analysis[cluster_idx] = cluster_mean

print("\nCluster Analysis (Mean of Features for Each Cluster):")
print(cluster_analysis)
