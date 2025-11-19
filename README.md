# Setting

Don’t you just love getting calls from a call centre? Now imagine working for one — as the manager! Each day you have far too many victims customers to choose from, so you need to prioritise. That’s where we come in.

We provide an API that assigns every customer a “success score” estimating how likely they are to convert a call into a shiny new bank loan (oh — did we forget to mention you also work for a bank?). This helps you direct your valued employees’ efforts where they matter most.

# 1. Problem Description

The data is from 
>[Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, In press, http://dx.doi.org/10.1016/j.dss.2014.03.001

We use it to select features and train a binary classification model. The Telemarketing team can then query this model to rank customers by likelihood of success. We are not making a hard yes/no prediction — we want a score suitable for ranking.

The workflow looks like this:
```
EDA + feature selection 
→ fine-tuning logistic regression and random forest models
→ export best model to training script and pickle it
→ create deployment script
→ build & publish container
→ set up cloud interface

```

# 2. Data Description

All data is in 📁 `data`.

An excerpt from `data/bank-additional-names.txt` provides short descriptions of the 20 features:

- **Bank Client Data**:

    1. age : age of client (numeric)
    2. job : type of job (categorical)
    3. marital : marital status (categorical)
    4. education (categorical)
    5. default: has credit in default? (categorical)
    6. housing: has housing loan? (categorical)
    7. loan: has personal loan? (categorical)

- **Last contact of the current campaign**:

    8. contact: contact communication type (categorical) 
    9. month: last contact month of year (categorical)
    10. day_of_week: last contact day of the week (categorical)
    11. duration: last contact duration, in seconds (numeric)

- **Other attributes**:

    12. campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
    13. pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
    14. previous: number of contacts performed before this campaign and for this client (numeric)
    15. poutcome: outcome of the previous marketing campaign (categorical)


- **Social and economic context**:

    16. emp.var.rate: employment variation rate - quarterly indicator (numeric)
    17. cons.price.idx: consumer price index - monthly indicator (numeric)     
    18. cons.conf.idx: consumer confidence index - monthly indicator (numeric)     
    19. euribor3m: euribor 3 month rate - daily indicator (numeric)
    20. nr.employed: number of employees - quarterly indicator (numeric)

> ⚠️ **Important**: For API calls, the “last contact” features (`contact`, `month`, `day_of_week`, `duration`) refer to the call _about to be made_. Since `duration` is obviously unknown before the call, it must be omitted in a predictive model.

# 3. EDA Summary

The dataset is generally clean: no missing values and no strange categories.

There is one inconsistency with `pdays`: its value should be `999` whenever `poutcome` is `nonexistent`, but several rows contradict this. This has no practical impact on AUC-ROC performance.

After a greedy feature-removal process, we retain the following seven features for training:
`cons.price.idx`, `contact`, `emp.var.rate`, `euribor3m`, `month`, `pdays`, `previous`.

# 4. Modelling Approach & Metrics

We keep the metric simple (KISS!) and use AUC-ROC to compare model performance.

We fine-tune both logistic regression and random forest models.
Random forest wins.

# 5. How to run

Dependencies are managed with `uv`.

> _Note_: All commands below assume you are in the project root.

First, create a virtual environment:
```bash
uv venv
```

To run 🗒️ `notebook.ipynb` locally you need to install all dependencies (e.g., for running notebook.ipynb):

```bash
uv sync
```

(I run Jupyter notebooks locally in VS Code and simply select the .venv Python kernel. I don't know about other setups.)


### Run the final Training
```bash
uv run train_pipeline.py
```
This produces a pickled model: `pipeline_v1.bin`  
(already included in the repo, so re-training is not required for deployment).

### Running the Server Locally
```bash
uv run uvicorn server:app --port 8000
```

### Running the Server in Docker

You can either build the Docker image yourself or pull the prebuilt version.

#### A. Build locally

```bash
docker build -t julxi/ml-zoomcamp-midterm:2025 .
```

#### B. Pull from repository

```bash
docker pull julxi/ml-zoomcamp-midterm:2025
```

#### Running the Docker image

Check that the image is available:
```bash
docker images
```

You should see something along these lines:
```bash
REPOSITORY                  TAG     IMAGE ID       CREATED        SIZE
julxi/ml-zoomcamp-midterm   2025    4fe632388077   6 hours ago    432MB
```

Run the container:
```bash
docker run -p 9696:9696 julxi/ml-zoomcamp-midterm:2025
```

# 6. API Usage Example

These examples assume the server or Docker container is running on the ports shown above.

### Server (port 8000)

#### Health
```bash
curl -X GET "http://localhost:8000/health"
```

#### Predict
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "nr.employed": 5008.7,
           "job": "admin.",
           "age": 29,
           "loan": "no",
           "cons.conf.idx": -40.0,
           "campaign": 1,
           "euribor3m": 0.683,
           "previous": 3,
           "cons.price.idx": 93.876,
           "contact": "cellular",
           "pdays": 3,
           "month": "may",
           "emp.var.rate": -1.8
         }'
```

### Docker container (port 9696)
#### Health
```bash
curl -X GET "http://localhost:9696/health"
```

#### Predict
```bash
curl -X POST "http://localhost:9696/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "nr.employed": 5008.7,
           "job": "admin.",
           "age": 29,
           "loan": "no",
           "cons.conf.idx": -40.0,
           "campaign": 1,
           "euribor3m": 0.683,
           "previous": 3,
           "cons.price.idx": 93.876,
           "contact": "cellular",
           "pdays": 3,
           "month": "may",
           "emp.var.rate": -1.8
         }'
```

## 7. Known limitations / next steps
