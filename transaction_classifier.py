import random
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

random.seed(4)


class Transaction:
    def __init__(self):
        self.categories = [
            "Restaurants", "Groceries", "Utilities", "Other"
        ]

        self.restaurant_vendors = [
            "Subway", "Chick-Fila", "Chipotle", "Blaze", "Pita Jungle", "In-N-Out Burger",
            "McDonalds", "Qdoba", "Baked Bear", "Starbucks", "Dominoes", "Hungry Howies", "Culinary Dropouts"
        ]

        self.grocery_vendors = [
            "Walmart", "Costco", "Food City", "Safeway", "Target", "Staples", "Whole Food Markets",
            "Sprouts Farmers Market", "Trader Joe's", "Natural Grocers"
        ]

        self.utilities_vendor = [
            "SRP", "APS", "AWS", "T-Mobile", "AT&T", "Sprint", "Cox"
        ]

        self.location_options = [
            "Tempe", "Phoenix", "San Francisco", "New York", "Dallas", "Denver", "Orlando", "Los Angles",
            "Seattle", "Boston", "Washington D.C.", "Raleigh", "San Diego"
        ]

        self.other_vendors = [
            "Lowes", "Home Depot", "Best Buy", "Staples", "H&M", "Hot Topic", "eBay", "Zara", "Rakuten", "Amazon"
        ]
        self.vendors = [self.restaurant_vendors,
                        self.grocery_vendors, self.utilities_vendor, self.other_vendors]

    def createTransaction(self, ID):
        self.index = random.randrange(len(self.vendors))
        return [random.choice(self.location_options), random.randrange(200), random.choice(self.vendors[self.index]),
                self.categories[self.index]]


if __name__ == "__main__":
    transaction = Transaction()
    x = [20000, 40000, 60000, 80000, 100000]
    last = []
    for num in x:
        train_data = []
        train_labels = []
        for i in range(num):
            res = transaction.createTransaction(i)
            train_data.append([res[2]])
            train_labels.append([res[3]])
        # print(train_data)
        # print(data)
        # enc = OneHotEncoder(handle_unknown='ignore')
        # enc.fit(data)
        # data = enc.transform(data).toarray()
        # print(len(data[0]))
        # print(data)
        # model = Model(data)
        # print(model.result())
        enc = OneHotEncoder(handle_unknown='ignore')
        train_data_np = np.asarray(train_data)
        train_labels_np = np.asarray(train_labels)
        enc.fit(train_data_np)
        train_data_hot_enc = enc.transform(train_data_np).toarray()
        enc.fit(train_labels_np)
        train_labels_hot_enc = enc.transform(train_labels_np).toarray()

        clf = MLPClassifier(solver="lbfgs", alpha=1e-5,
                            hidden_layer_sizes=(20, 10), learning_rate="adaptive")
        clf.fit(train_data_hot_enc, train_labels_hot_enc)
        test_data = []
        test_labels = []
        enc.fit(train_data_np)
        for i in range(num):
            res = transaction.createTransaction(i)
            test_data.append([res[2]])
            test_labels.append([res[3]])

        test_data_np = np.asarray(test_data)
        test_labels_np = np.asarray(test_labels)
        # print((test_labels_np))
        test_data_hot_enc = enc.transform(test_data_np).toarray()
        # print(test_data_hot_enc)
        res = clf.predict(test_data_hot_enc)

        enc.fit(train_labels_np)
        result = enc.inverse_transform(res)
        # print(result)
        sample = test_labels_np != result
        # print(np.sum(test_labels_np=='Restaurants'))
        # print(np.sum(result=='Restaurants'))
        # print(np.sum(test_labels_np=='Groceries'))
        # print(np.sum(result=='Groceries'))
        # print(np.sum(test_labels_np=='Utilities'))
        # print(np.sum(result=='Utilities'))
        # print(np.sum(test_labels_np=='Other'))
        # print(np.sum(result=='Other'))

        print("Accuracy - ", np.sum(sample == False)*100/len(sample))
        last.append(np.sum(sample == False)*100/len(sample))
    y = last
    plt.plot(x, y)
    plt.ylabel('Accuracy')
    plt.xlabel('Size of Training Data')
    plt.title('Classification of Transactions')

    plt.show()
