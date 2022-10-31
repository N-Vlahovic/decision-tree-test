from sklearn.datasets import load_breast_cancer  # noqa
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier  # noqa
from sklearn.model_selection import train_test_split  # noqa
from sklearn.metrics import accuracy_score
from sklearn.utils import Bunch

import pandas as pd
from pyspark.sql import DataFrame as SparkDF, Row, functions as F, types as T, SparkSession  # noqa


def create_session(
        name: str = None
) -> SparkSession:
    return SparkSession.builder.appName(name or __file__ or "Some Session").getOrCreate()


def sk_bunch_to_spark_df(spark: SparkSession, ds: Bunch) -> SparkDF:
    col_names = [_.strip().replace(" ", "_") for _ in ds["feature_names"]]
    rows = [Row(**{k: v for k, v in zip(col_names, tuple(x))}) for x in ds["data"]]
    return spark.createDataFrame(rows)


def main() -> None:
    print("Loading data")
    breast_data = load_breast_cancer(as_frame=True)
    df = breast_data.data
    df["target"] = breast_data.target
    print("OK")

    print("Separating train/test")
    X = df.copy()
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    print("OK")

    print("Loading model")
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    print("OK")

    print("Predict")
    y_pred = clf.predict(X_test)
    print("OK")

    print("Check performance")
    print(accuracy_score(y_test, y_pred))


if __name__ == '__main__':
    main()
