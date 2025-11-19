import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

SEED = 1

df = pd.read_csv("data/bank-additional-full.csv", sep=";")
TARGET = "y"
df[TARGET] = df[TARGET].map({"yes": 1, "no": 0})

df_train, df_test = train_test_split(df, test_size=0.2, random_state=SEED)
y_train = df_train.pop(TARGET)

RESTRICTED_FEATURES = [
    "age",
    "pdays",
    "default",
    "campaign",
    "euribor3m",
    "contact",
    "day_of_week",
    "month",
    "cons.price.idx",
    "poutcome",
    "emp.var.rate",
]

rf_best_params = {
    "bootstrap": False,
    "class_weight": None,
    "max_depth": 20,
    "max_features": "sqrt",
    "min_samples_leaf": 20,
    "n_estimators": 800,
}

train_dict = df_train[RESTRICTED_FEATURES].to_dict(orient="records")

pipeline = make_pipeline(
    DictVectorizer(sparse=False),
    StandardScaler(),
    RandomForestClassifier(
        **rf_best_params,
        random_state=SEED,
        n_jobs=-1,
    ),
)


pipeline.fit(train_dict, y_train)


# evaluate
y_test = df_test.pop(TARGET)
test_dict = df_test[RESTRICTED_FEATURES].to_dict(orient="records")
y_test_prob = pipeline.predict_proba(test_dict)[:, 1]
test_auc = average_precision_score(y_test, y_test_prob)

print(f"Finished training. Test AP: {test_auc:.3f}")

with open("pipeline_v1.bin", "wb") as file_out:
    pickle.dump(pipeline, file_out)
