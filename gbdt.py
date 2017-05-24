import numpy as np
from scipy import sparse
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd

# getting data
print('loading training data...')
file = open('train_data.txt')
train_X = []
train_y = []
train_row_index = []
train_col_index = []
count = 0

for line in file:
    data_line = str(line).split()
    train_y.append(int(data_line[0]))
    for i in range(1, len(data_line)):
        index_value = data_line[i].split(':')
        index = int(index_value[0])
        value = float(index_value[1])
        # it is said that test example does not contain more than 132 features -- mahua
        if index > 132:
            break
        train_X.append(value)
        train_row_index.append(count)
        train_col_index.append(index)
    count = count + 1

file.close()

train_X = sparse.csr_matrix((train_X, (train_row_index, train_col_index))).todense()
train_y = np.array(train_y)

## feature scaling
print('feature normalizing...')
m, n = train_X.shape
X_mean = []
X_std = []
for i in range(n):
    mean = np.mean(train_X[:, i])
    std = np.std(train_X[:, i])
    X_mean.append(mean)
    X_std.append(std)
X_mean = np.array(X_mean)
X_std = np.array(X_std)

epthelon = 1e-20
for i in range(n):
    train_X[:, i] = (train_X[:, i] - X_mean[i]) / (X_std[i] + epthelon)

## training
print('training...')
clf = GradientBoostingClassifier()
clf.fit(train_X, train_y)

## prediciton
print('loading testing data...')
file = open('test_data.txt')
test_X = []
test_row_index = []
test_col_index = []
count = 0

for line in file:
    data_line = str(line).split()
    id = int(data_line[0])
    if id == count:
        for i in range(1, len(data_line)):
            index_value = data_line[i].split(':')
            index = int(index_value[0])
            value = float(index_value[1])
            test_X.append(value)
            test_row_index.append(count)
            test_col_index.append(index)
        count = count + 1

file.close()

test_X = sparse.csr_matrix((test_X, (test_row_index, test_col_index))).todense()
m, n = test_X.shape
for i in range(n):
    test_X[:, i] = (test_X[:, i] - X_mean[i]) / (X_std[i] + epthelon)
predit = clf.predict(test_X)

print('writting to file...')
df = pd.DataFrame(predit)
df.to_csv('predict.csv', float_format='%.12f', index_label='id', header=['label'])
