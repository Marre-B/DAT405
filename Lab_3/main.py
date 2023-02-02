import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, DBSCAN

# Question 1
def question1(data):

    # Filter data
    psi = data['psi']
    phi = data['phi']

    # Scatter plot
    plt.scatter(psi, phi, s=4, alpha=0.25, c="red", marker=".")
    plt.xlabel('Psi [deg]')
    plt.ylabel('Phi [deg]')
    plt.title('Scatter plot of psi and phi')
    plt.plot()
    plt.figure()

    # 2D histogram
    plt.hist2d(psi, phi, bins=100)
    plt.xlabel('Psi [deg]')
    plt.ylabel('Phi [deg]')
    plt.title('2D Histogram of psi and phi')
    plt.plot()
    plt.show()

# Question 2
def question2(data):

    # Filter data
    psi = data['psi']
    phi = data['phi']

    # Insert data into pairs in an array
    data = []
    for x in range(0, min(len(phi), len(psi))):
        data.append([psi[x], phi[x]])

    # K-means clustering
    kmeans = KMeans(n_clusters=6, n_init=10, algorithm= 'elkan')
    label = kmeans.fit_predict(data)
    data = np.array(data)
    label = np.array(label)

    # Getting centroids
    centroids = kmeans.cluster_centers_
    u_clusters = np.unique(label)

    # Plot clusters
    for cluster in u_clusters:
        plt.scatter(data[label == cluster, 0], data[label == cluster, 1], s=6, alpha=0.25, marker=".")
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=45)
    plt.legend()
    plt.xlabel('Psi [deg]')
    plt.ylabel('Phi [deg]')
    plt.title('K-means cluster scatter plot of psi and phi')
    plt.show()

# Question 3
def question3(data):

    # Filter data
    psi = data['psi']
    phi = data['phi']
    p_type = data['residue name']

    # Plot data using DBSCAN clustering
    labels = dbscanPlot(phi, psi)

    outliers = []
    # Get outliers
    for x in range(0, len(labels)):
        if labels[x] == -1:
            outliers.append(x)
    p_type = np.array(p_type)
    outliers = np.array(outliers)
    outliers = p_type[outliers]
   
    
    # Plot outliers in bar graph
    for i in np.unique(outliers):
        plt.bar(x=i, height=np.count_nonzero(outliers == i))
    plt.title(str(len(outliers)) + ' total outliers in bar graph')
    plt.xlabel('Residue name')
    plt.ylabel('Number of outliers')
    plt.show()
    
# Question 4
def question4(data):

    # Filter data
    PROData = data[data["residue name"] == "PRO"]
    psi = PROData['psi']
    phi = PROData['phi']

    # Reset index
    phi = np.array(phi)
    psi = np.array(psi)

    # Plot data using DBSCAN clustering
    dbscanPlot(phi, psi)

    

# Plot data using DBSCAN clustering
def dbscanPlot(phi, psi):

    # Insert data into pairs in an array
    data = []
    for x in range(0, min(len(phi), len(psi))):
        data.append([psi[x], phi[x]])

    # DBSCAN clustering
    dbscan = DBSCAN(eps=8.5, min_samples=52).fit(data)
    labels = dbscan.labels_
    data = np.array(data)
    
    # Plot clusters
    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[dbscan.core_sample_indices_] = True

    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = labels == k

        xy = data[class_member_mask & core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=9,
        )

        xy = data[class_member_mask & ~core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=4,
        )
    plt.xlabel('Psi [deg]')
    plt.ylabel('Phi [deg]')
    plt.title('DBSCAN cluster scatter plot of psi and phi')
    plt.show()

    return labels

def main():

    # Load data
    data = pd.read_csv('data_assignment3.csv')
    
    question1(data)
    question2(data)
    question3(data)
    question4(data)

if __name__ == "__main__":
    main()