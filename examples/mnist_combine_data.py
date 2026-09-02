"""
concatenate the mnist csv's from kaggle into one large dataset for surge.
"""
import pandas as pd
mnist_train = pd.read_csv("C:\\Users\\Bipo1\\Downloads\\Kaggle MNIST CSV Data\\mnist_train.csv")
mnist_test = pd.read_csv("C:\\Users\\Bipo1\\Downloads\\Kaggle MNIST CSV Data\\mnist_test.csv")
mnist_full = pd.concat([mnist_train, mnist_test], ignore_index=True)
mnist_full.to_csv("C:\\Users\\Bipo1\\Downloads\\Kaggle MNIST CSV Data\\mnist_full.csv", index=False)

asl_mnist_train = pd.read_csv("C:\\Users\\Bipo1\\Downloads\\Kaggle ASL MNIST Data\\sign_mnist_train.csv")
asl_mnist_test = pd.read_csv("C:\\Users\\Bipo1\\Downloads\\Kaggle ASL MNIST Data\\sign_mnist_test.csv")
asl_mnist_full = pd.concat([asl_mnist_train, asl_mnist_test], ignore_index=True)
asl_mnist_full.to_csv("C:\\Users\\Bipo1\\Downloads\\Kaggle ASL MNIST Data\\sign_mnist_full.csv", index=False)

fashion_train = pd.read_csv("C:\\Users\\Bipo1\\Downloads\\Kaggle Fashion MNIST Data\\fashion-mnist_train.csv")
fashion_test = pd.read_csv("C:\\Users\\Bipo1\\Downloads\\Kaggle Fashion MNIST Data\\fashion-mnist_test.csv")
fashion_full = pd.concat([fashion_train, fashion_test], ignore_index=True)
fashion_full.to_csv("C:\\Users\\Bipo1\\Downloads\\Kaggle Fashion MNIST Data\\fashion_full.csv", index=False)
