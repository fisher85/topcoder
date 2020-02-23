result = 1
with open("c:/topcoder/topcoder/2020-02-23 Yandex Interview Contest/Task E. Anagrams/input.txt", "rb") as fr:
    # Get first distribution
    d1 = {}
    while True:
        c = fr.read(1)
        if (ord(c) == 10 or ord(c) == 13):
            break
        if (ord(c) in d1):
            d1[ord(c)] = d1[ord(c)] + 1
        else:
            d1[ord(c)] = 1
    # Compare distributions online 
    while True:
        c = fr.read(1)
        if not c:
            # End of file
            break
        if (ord(c) == 10 or ord(c) == 13):
            continue
        if ((ord(c) in d1) and (d1[ord(c)] > 0)):
            d1[ord(c)] = d1[ord(c)] - 1
        else:
            result = 0
            break
for key in d1:
    if d1[key] != 0:
        result = 0
        break
print(result)