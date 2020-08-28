n, vx1, vy1 = map(int, input().split())
a = []
for i in range(n):
    x, y = map(int, input().split())
    a.append([x, y])
m, vx2, vy2 = map(int, input().split())
aa = []
for i in range(m):
    x, y = map(int, input().split())
    aa.append([x, y])
b = True
for i in range(n):
    if b:
        for j in range(m):
            if not(aa[j][0]-a[i][0]!=0 and vx1-vx2==0):
                if aa[j][0]-a[i][0]==vx1-vx2==0:
                    if not(aa[j][1]-a[i][1]!=0 and vy1-vy2==0):
                        print('Yes')
                        b = False
                        break   
                else:
                    x=(aa[j][0]-a[i][0])/(vx1-vx2)
                    if not(aa[j][1]-a[i][1]!=0 and vy1-vy2==0) and (aa[j][0]-a[i][0]==vx1-vx2==0 or x==(aa[j][1]-a[i][1])/(vy1-vy2)):
                        print('Yes')
                        b = False
                        break
    else:
        break
    
else:
    print('No')