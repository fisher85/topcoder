class MeanCalculator:
    def __init__(self):
        self.Count = 0.
        self.Mean = 0.

    def Add(self, value, weight = 1.):
        self.Count += weight
        self.Mean += weight * (value - self.Mean) / self.Count

    def Remove(self, value, weight = 1.):
        self.Add(value, -weight)

class SumSquaredErrorsCalculator:
    def __init__(self):
        self.MeanCalculator = MeanCalculator()
        self.SSE = 0.

    def Add(self, value, weight = 1.):
        curDiff = value - self.MeanCalculator.Mean
        self.MeanCalculator.Add(value, weight)
        self.SSE += weight * curDiff * (value - self.MeanCalculator.Mean)

    def Remove(self, value, weight = 1.):
        self.Add(value, -weight)

class Instances:
    def __init__(self):
        self.Items = []
        self.OverAllSSE = SumSquaredErrorsCalculator()

    def Read(self):
        file_name = "2019-05-12 Yandex ML blitz 2018 final/input_A_2.txt"
        #file_name = "stump.in"
        fin = open(file_name)

        for line in fin.readlines()[1:]:
            x, y = map(float, line.strip().split())
            self.Items.append([x, y])
            self.OverAllSSE.Add(y)
        self.Items.sort()

    def BestSplit(self):
        left = SumSquaredErrorsCalculator()
        right = self.OverAllSSE

        bestA = 0
        bestB = right.MeanCalculator.Mean
        bestC = self.Items[0][0]

        bestQ = right.SSE

        for i in range(len(self.Items) - 1):
            item = self.Items[i]
            nextItem = self.Items[i + 1]

            left.Add(item[1])
            right.Remove(item[1])

            if item[0] == nextItem[0]:
                continue

            a = left.MeanCalculator.Mean
            b = right.MeanCalculator.Mean
            c = (item[0] + nextItem[0]) / 2

            q = left.SSE + right.SSE

            if q < bestQ:
                bestA = a
                bestB = b
                bestC = c
                bestQ = q

        return bestA, bestB, bestC

instances = Instances()
instances.Read()
a, b, c = instances.BestSplit()

output_name = "2019-05-12 Yandex ML blitz 2018 final/output_A_12.txt"
#output_name = "stump.out"
with open(output_name, "w") as fw:
    fw.write(str.format("{0:.6f}", a) + " ")
    fw.write(str.format("{0:.6f}", b) + " ")
    fw.write(str.format("{0:.6f}", c))