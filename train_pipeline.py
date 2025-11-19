import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

SEED = 1

df = pd.read_csv("data/bank-additional-full.csv", sep=";")
TARGET = "y"
df[TARGET] = df[TARGET].map({"yes": 1, "no": 0})

df_train, df_test = train_test_split(df, test_size=0.2, random_state=SEED)
y_train = df_train.pop(TARGET)

mini_features = [
    "cons.price.idx",
    "contact",
    "emp.var.rate",
    "euribor3m",
    "month",
    "pdays",
    "previous",
]

train_dict = df_train[mini_features].to_dict(orient="records")

pipeline = make_pipeline(
    DictVectorizer(sparse=False),
    StandardScaler(),
    RandomForestClassifier(
        random_state=SEED,
        n_jobs=-1,
        bootstrap=True,
        class_weight="balanced",
        max_depth=8,
        max_features=0.5,
        min_samples_leaf=12,
        min_samples_split=13,
        n_estimators=439,
    ),
)


pipeline.fit(train_dict, y_train)


# evaluate
y_test = df_test.pop(TARGET)
test_dict = df_test[mini_features].to_dict(orient="records")
y_test_prob = pipeline.predict_proba(test_dict)[:, 1]
test_auc = roc_auc_score(y_test, y_test_prob)

print(f"Finished training. Test AUC: {test_auc:.3f}")

with open("pipeline_v1.bin", "wb") as file_out:
    pickle.dump(pipeline, file_out)
