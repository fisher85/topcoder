# input_file = "c:/topcoder/topcoder/2020-08-27 Tinkoff SRE/Task 5. Minimal paths in graph/input.txt"

dist = []

# n - количество вершин
# m - количество рёбер

# with open(input_file, "r") as fr:

# n, m = map(int, fr.readline().strip().split())
n, m = map(int, input().strip().split())

for i in range (0, n):
    dist.append([])
    for j in range (0, n):
        dist[i].append(999999)
    
for i in range(0, m):
    # x, y = map(int, fr.readline().strip().split())
    x, y = map(int, input().strip().split())
    dist[x - 1][y - 1] = 1
    dist[y - 1][x - 1] = 1

for i in range (0, n):
    dist[i][i] = 0

for k in range(0, n):
    for i in range(0, n):
        for j in range(0, n):
            if dist[i][j] > dist[i][k] + dist[k][j]:
                dist[i][j] = dist[i][k] + dist[k][j]

# q = int(fr.readline())   
q = int(input())

for i in range (0, q):
    # x, y = map(int, fr.readline().strip().split())
    x, y = map(int, input().strip().split())
    if dist[x - 1][y - 1] > 999998:
        print(-1)
    else:
        print(dist[x - 1][y - 1])