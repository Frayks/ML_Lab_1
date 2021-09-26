import numpy
import statistics
import time
from sklearn.cluster import Birch
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from matplotlib import pyplot


def change_loc(array, shift_x, shift_y):
    for point in array:
        point[0] = point[0] + shift_x
        point[1] = point[1] + shift_y


complex1 = numpy.random.normal(0, 1, size=(1000, 2))
change_loc(complex1, 3, 3)
complex2 = numpy.random.normal(0, 1, size=(1000, 2))
change_loc(complex2, 5, 5)
complex3 = numpy.random.normal(0, 1, size=(1000, 2))
change_loc(complex3, 7, 4)
test_complex = numpy.vstack((complex1, complex2, complex3))

model = Birch(threshold=0.01, n_clusters=3)
# model = KMeans(n_clusters=3)
# model = SpectralClustering(n_clusters=3)

time_start = time.time()
# Визначення належності точок до кластерів
points_clustered_index = model.fit_predict(test_complex)
time_end = time.time()
print("Time spent: {:.3f} c.".format(time_end - time_start))

# Визначення унікальних кластерів
cluster_nums = numpy.unique(points_clustered_index)

clusters = []
# створення діаграми розсіювання для зразків з кожного кластера
for cluster_num in cluster_nums:
    # отримати індекси рядків для зразків із цим кластером
    row_ix = numpy.where(points_clustered_index == cluster_num)
    # сформувати масив кластерів
    clusters.append(test_complex[row_ix])
    # створити розкид цих зразків
    pyplot.scatter(test_complex[row_ix, 0], test_complex[row_ix, 1])

# показати графік
pyplot.title(model.__class__.__name__)
pyplot.show()

print("Mean:")
for index, cluster in enumerate(clusters):
    cluster_x = cluster[:, 0]
    cluster_y = cluster[:, 1]
    print("    Cluser {}:".format(index + 1))
    print("        x: {:.3f}".format(statistics.mean(cluster_x)))
    print("        y: {:.3f}".format(statistics.mean(cluster_y)))

print("Median:")
for index, cluster in enumerate(clusters):
    cluster_x = cluster[:, 0]
    cluster_y = cluster[:, 1]
    print("    Cluser {}:".format(index + 1))
    print("        x: {:.3f}".format(statistics.median(cluster_x)))
    print("        y: {:.3f}".format(statistics.median(cluster_y)))

print("Mode:")
for index, cluster in enumerate(clusters):
    cluster_x = cluster[:, 0]
    cluster_y = cluster[:, 1]
    print("    Cluser {}:".format(index + 1))
    print("        x: {:.3f}".format(statistics.mode(cluster_x)))
    print("        y: {:.3f}".format(statistics.mode(cluster_y)))

print("Stdev:")
for index, cluster in enumerate(clusters):
    cluster_x = cluster[:, 0]
    cluster_y = cluster[:, 1]
    print("    Cluser {}:".format(index + 1))
    print("        x: {:.3f}".format(statistics.stdev(cluster_x)))
    print("        y: {:.3f}".format(statistics.stdev(cluster_y)))
