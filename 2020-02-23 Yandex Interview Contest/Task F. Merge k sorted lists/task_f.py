# 1: TL - 1.09s 4.32Mb test № 17
# 2, save prev position: 
# 3, sort at startup: 

k = 0
a = []

def diho(arr, element, previous_position):
    if element < arr[0]:
        arr.insert(0, element)
        return 0
    if element > arr[len(arr) - 1]:
        arr.insert(len(arr), element)
        return len(arr) - 1

    left = 0
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
    for i in range(0, k):
        next_line = input_lines[i + 1].strip()
        num = next_line[0]
        b = [int(item) for item in next_line.split(' ')]
        b.pop(0)
        a.append(b)

if (k > 0):
    result = a[0].copy()
    previous_position = 0
    for i in range(1, k):
        for j in a[i]:
            previous_position = diho(result, j, previous_position)

    print(*result, sep=' ')