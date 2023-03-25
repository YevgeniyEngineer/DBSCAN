# DBSCAN

Very fast sequential C++ implementation of the DBSCAN algorithm based on Nanoflann library for nearest neighbor search.

# Benchmarks for 3D DBSCAN:

CPU: 12th Gen Intel® Core™ i7-12700H × 20 | 16.0 GiB RAM

| Number of Points | Average Clustering Time (s) |
| ---------------- | --------------------------- |
| 1,000            | 92.330e-06                  |
| 10,000           | 138.30e-06                  |
| 50,000           | 10.565e-03                  |
| 100,000          | 22.980e-03                  |
| 1,000,000        | 799.613e-03                 |