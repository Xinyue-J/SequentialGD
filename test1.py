from plotDecBoundaries import plotDecBoundaries
import csv
import numpy as np

file = open('synthetic3_train.csv')
length = len(file.readlines())

sum_x1 = 0
sum_x2 = 0
sum_y1 = 0
sum_y2 = 0
class1_cnt = 0
class2_cnt = 0
error_train = 0
error_test = 0
m = 1000

criterion = np.zeros((length, 1))
w = np.zeros((length, 3))
train = np.zeros((length, 3))
test = np.zeros((100, 2))
label_train = np.zeros(length, dtype=np.int)
label_test = np.zeros(100, dtype=np.int)


def zn(label):
    if label == 1:
        return 1
    else:
        return -1


reader = csv.reader(open('synthetic3_train.csv'))  # must read twice
index = 0
for row in reader:
    train[index, 0] = float(row[0])
    train[index, 1] = float(row[1])
    train[index, 2] = float(row[2])
    index += 1

np.random.shuffle(train)
label_train = train[:, 2]
train = np.hstack((train[:, 0].reshape(length, 1), train[:, 1].reshape(length, 1)))

w = np.array([[0.1, 0.1, 0.1]]).T  # two []
x = np.array([[1.0, 0.0, 0.0]]).T
weight = np.zeros((m * length, 3))
halt = 0
j = 0
for m in range(1, m + 1):
    for n in range(0, length):
        i = (m - 1) * length + n
        x[1, 0] = train[n, 0]
        x[2, 0] = train[n, 1]
        if np.dot(w.T, x) * zn(label_train[n]) <= 0:  # if wT*zn*xn<=0
            w = w + zn(label_train[n]) * x
            halt = 0
        else:
            halt += 1
        weight[i, 0] = w[0, 0]
        weight[i, 1] = w[1, 0]
        weight[i, 2] = w[2, 0]

        if halt == length:
            final_w = weight[i]
            print('final:', final_w)
            break

        # calculate J(w) during the last full epoch
        if m == 1000:
            j = 0
            for i in range(0, length):
                x[1, 0] = train[i, 0]
                x[2, 0] = train[i, 1]
                j = j - zn(label_train[i]) * np.dot(w.T, x) * (np.dot(w.T, x) * zn(label_train[i]) <= 0)
            criterion[n] = float(j)
        # print final weight by finding the min J(w)

    if halt == length:
        break

if m == 1000:
    for i in range(0, length):
        if criterion[i] == criterion.min(0):
            final_w = weight[(m - 1) * length + i]
            print('final:', final_w)
            break

x = np.array([[1.0, 0.0, 0.0]]).T

# error rate for train
index = 0
for axis_train in train:
    x[1, 0] = axis_train[0]
    x[2, 0] = axis_train[1]
    if np.dot(final_w.T, x) * zn(label_train[index]) <= 0:  # if g(zn*xn)<=0
        error_train += 1
    index += 1
print('error rate for train:', error_train / length)

# error rate for test
reader = csv.reader(open('synthetic3_test.csv'))
index = 0
for row in reader:
    test[index, 0] = float(row[0])
    test[index, 1] = float(row[1])
    label_test[index] = float(row[2])
    index += 1

index = 0
for axis_test in test:
    x[1, 0] = axis_test[0]
    x[2, 0] = axis_test[1]
    if np.dot(final_w.T, x) * zn(label_test[index]) <= 0:  # if g(zn*xn)<=0
        error_test += 1
    index += 1
print('error rate for test:', error_test / 100)

print(m)
plotDecBoundaries(train, label_train, final_w)
# plotDecBoundaries(test, label_test, final_w)
