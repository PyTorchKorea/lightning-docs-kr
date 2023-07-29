# app.py
# !pip install sklearn xgboost
# !conda install py-xgboost-gpu
import lightning as L
from sklearn import datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

class XGBoostComponent(L.LightningWork):
    def run(self):
        iris = datasets.load_iris()
        X, y = iris.data, iris.target

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        bst = XGBClassifier(tree_method='gpu_hist', gpu_id=0, verbosity=3)
        bst.fit(X_train, y_train)
        preds = bst.predict(X_test)
        print(f'preds: {preds}')

compute = L.CloudCompute('gpu')
app = L.LightningApp(XGBoostComponent(cloud_compute=compute))
