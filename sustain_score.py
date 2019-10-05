import random
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

def createRecords(i):
    water = random.uniform(10, 100)
    energy = random.uniform(10, 100)
    plastic = random.uniform(10, 100)
    cars = random.randrange(10, 100)
    trees = random.randrange(10, 100)
    record = list(map(lambda x: round(x,2),[water, energy, plastic, cars, trees]))
    #multipler = random.uniform(0,1)
    score = sum(record)/ len(record)
    return record, score

training_set, training_labels = [], []
for i in range(10000):
    training_record, training_label = createRecords(i)
    training_set.append(training_record)
    training_labels.append(training_label)
X, y = training_set, training_labels

reg = LinearRegression().fit(X, y)
tree_model = DecisionTreeRegressor().fit(X, y)
rf_model = RandomForestRegressor().fit(X, y)

# print(tree_model)
# print(rf_model)
test_set, test_labels = [], []
for i in range(10):
    record, label = createRecords(i)
    test_set.append(record)
    test_labels.append(label)
X, y = test_set, test_labels
print(test_labels)
result = list(map(lambda x: round(x,2),tree_model.predict(test_set)))
print(result)
result = list(map(lambda x: round(x,2),rf_model.predict(test_set)))
print(result)

print(rf_model.predict([[52.85,100, 135, 142, 30]]))