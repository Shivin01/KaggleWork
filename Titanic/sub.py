import pandas as pd

train_data = pd.read_csv('train.csv')
# print(train_data.head())
#
test_data = pd.read_csv("test.csv")
# print(test_data.head())

print(train_data.Age)

# features = ["Pclass", "Sex", "SibSp", "Parch"]
# print(f'features : {features}')
# X = pd.get_dummies(train_data[features])
# print(f'X={X}')
# X_test = pd.get_dummies(test_data[features])
# print(f'X_test={X_test}')

