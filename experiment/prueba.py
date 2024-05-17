from naiveautoml import NaiveAutoML
import sklearn.datasets
naml = NaiveAutoML(evaluation_fun="mccv_1", random_state=2)
X, y = sklearn.datasets.load_iris(return_X_y=True)
naml.fit(X, y)
print(naml.chosen_model)