import json
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
import random
import pprint

industries = ["Climate", "Food", "Education", "Local Biz", "Healthcare"]
selection = 3
enc = OneHotEncoder(handle_unknown='ignore')

industry_labels = {
    0: random.sample(industries, k = selection),
    1: ["Food", "Education", "Healthcare"],
    2: random.sample(industries, k = selection),
    3: random.sample(industries, k = selection),
    4: ["Climate", "Education", "Local Biz"]
}

with open('account_data.txt') as json_file:
    data = json.load(json_file)
    
    res = set()
    for p in data:
        if p["zip"] not in res:
            res.add(int(p["zip"]))
    
    X = np.asarray(list(res)).reshape(-1,1)
    kmeans = KMeans(n_clusters=5, random_state=10).fit(X)
    labels = list(kmeans.labels_)
    zip_label = dict()
    zips = list(res)
    for i in range(len(labels)):
        zip_label[zips[i]] = labels[i]

    for p in data:
        if p["zip"] in zip_label:
            p["recommendation"] = industry_labels[zip_label[p["zip"]]]
    
    pprint.pprint(data)
    features = []
    results = []
    for i in range(len(data)):
        results.append(data[i]['recommendation'])
        data[i].pop('recommendation')
    #print(results)
    for i in range(len(data)):
        if data[i]["ficoScore"] == '':
            data[i]["ficoScore"] = 0
        if data[i]["customScore"] == '':
            data[i]["customScore"] = 0
        res= [data[i]["ficoScore"], data[i]["customScore"], data[i]["balanceAvailable"]]
        features.append(res)
    features_np = np.asarray(features)
    results_np = np.asarray(results)
    enc.fit(results_np)
    result = enc.transform(results_np).toarray()
    print(enc.get_feature_names())
    X_train, X_test, y_train, y_test =\
        train_test_split(features_np, result, test_size=0.2, random_state=42)
    
    clf = MLPClassifier(solver="lbfgs", alpha=1e-5,
                            hidden_layer_sizes=(100, 80, 50, 20), learning_rate="adaptive")
    classifier = clf.fit(X_train, y_train)
    #print(classifier)
    fint = enc.inverse_transform(y_test)
    #print(fint)
    fin = clf.predict(X_test)
    finl = enc.inverse_transform(fin)
    #print(finl)

