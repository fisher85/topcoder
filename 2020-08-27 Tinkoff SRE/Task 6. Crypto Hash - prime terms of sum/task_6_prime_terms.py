

def is_prime(n):
    i = 2
    j = 0 # флаг
    while i**2 <= n and j != 1:
        if n%i == 0:
            j = 1
        i += 1
    if j == 1:
        return False
    else:
        return True

# print(is_prime(99767))
# print(is_prime(99768))

x = int(input())

if is_prime(x):
    print(1)
else:
    if x % 2 == 0:
        print(2) # Проблема Гольдбаха
    else:
        # Если не простое и нечетное
        probe = x - 2
        if is_prime(probe):
            print(2)
        else:
            print(3)