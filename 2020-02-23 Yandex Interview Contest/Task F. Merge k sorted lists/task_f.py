# 1: Python 3.4.3 TL - 1.09s 4.32Mb test № 17
# 2, save prev position: Python 3.4.3 TL - 1.09s 4.36Mb test № 17
# 3, sort at startup: NO
# 3, move to Python 2.7: NO DIHO, GUY! JUST CHECK STAT!

k = int(input())
distr = {}
for _ in range(k):
    next_line = input().split()
    for i in next_line[1:]:
        if i in distr:
            distr[i] += 1
        else:
            distr[i] = 1

for i in range(101):
    i = str(i)
    if i in distr:
        print(' '.join([i] * distr[i]), end=' ')

"""
            # Get stat. Too slow, find series!!!
            # for j in numbers[1:]: distr[j] = distr[j] + 1
            series_start = -1
            series_count = 0
            for j in numbers[1:]:
                if (j > series_start): # Start new series
                    # Save previous series
                    if (series_start != -1):
                        distr[series_start] = distr[series_start] + series_count 
                    series_start = j
                    series_count = 1
                else:
                    series_count = series_count + 1
            distr[series_start] = distr[series_start] + series_count

        # Output
        output_name = "c:/topcoder/topcoder/2020-02-23 Yandex Interview Contest/Task F. Merge k sorted lists/output.txt"
        with open(output_name, "w") as fw:
            for key in range(0, 101):
                if (distr[key] > 0):
                    for i in range(0, distr[key]):
                        fw.write('{}'.format(key) + " ")
"""

"""
def diho(arr, element, previous_position):
    if element < arr[0]:
        arr.insert(0, element)
        return 0
    if element > arr[len(arr) - 1]:
        arr.insert(len(arr), element)
        return len(arr) - 1

    left = previous_position
    right = len(arr) - 1

    while (left < right):
        mid = int((right - left)/2 + left)
        if (element > arr[mid]):
            left = mid + 1
        else:
            right = mid

    arr.insert(left, element)
    return left

with open("c:/topcoder/topcoder/2020-02-23 Yandex Interview Contest/Task F. Merge k sorted lists/input.txt", "r") as fr:
    input_lines = fr.readlines()
    k = int(input_lines[0])

    if (k > 0):
        # result = a[0].copy() # Python 3.4.3
        first_line = input_lines[1]
        first_splits = first_line.strip().split(' ')
        result = [int(num) for num in first_splits[1:]]

        for i in range(2, k + 1):
            next_line = input_lines[i]
            next_splits = next_line.strip().split(' ')

            previous_position = 0
            for j in next_splits[1:]:            
                previous_position = diho(result, int(j), previous_position)

        # print(*result, sep=' ') # Python 3.4.3
        output_name = "c:/topcoder/topcoder/2020-02-23 Yandex Interview Contest/Task F. Merge k sorted lists/output.txt"
        with open(output_name, "w") as fw:
            for x in result: 
                fw.write('{}'.format(x) + " ")
"""