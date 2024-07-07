# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.head()

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head()


women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)


men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)

train_data_updated = train_data.interpolate(method='linear')
test_data_updates = test_data.interpolate(method='linear')

y = train_data_updated['Survived']

features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train_data_updated[features])
X_test = pd.get_dummies(test_data_updates[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)
len(predictions)

output = pd.DataFrame({'PassengerId': test_data_updates.PassengerId, 'Survived': predictions})
print(output)
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")
