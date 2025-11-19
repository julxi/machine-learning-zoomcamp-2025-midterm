# Setting

Who doesn’t love receiving calls from a call center? Now, imagine managing one! With countless ~~victims~~ customers to choose from each day, prioritization is key. That’s where we, as ML Engineers, step in. We provide an API for the manager that assigns each customer a “success score,” estimating how likely they are to convert a call into a shiny new bank loan (oh—did I forget to mention this is for a bank?). This will help the manager to direct their team’s efforts where they matter most.

# 1. Problem Description

The dataset is sourced from:

>[Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, In press, http://dx.doi.org/10.1016/j.dss.2014.03.001

I use this data to select features and train a binary classification model. The telemarketing team can query this model to rank customers by their likelihood of success. Instead of making a hard yes/no prediction, it generates a score suitable for ranking.

The workflow is as follows:
```
Exploratory Data Analysis (EDA) + Feature Selection
→ Fine-tuning Logistic Regression and Random Forest Models
→ Training Script with Best Parameters and Pickling the Model
→ Creating a Deployment Script
→ Building and Publishing a Container
```

**Note**: There is no cloud deployment 🚫.

# 2. Data Description

The telemarketing data is located in the `data` folder. The file `data/bank-additional-names.txt` provides descriptions of the 20 features. Here is an overview:

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

> ⚠️ **Important**: For the API calls, the “last contact” features (`contact`, `month`, `day_of_week`, `duration`) refer to the call _about to be made_. Since `duration` is obviously unknown before the call, it must be omitted in a predictive model.

# 3. EDA Summary

The dataset is clean, with no missing values or unusual categories. One inconsistency exists with `pdays`: its value should be `999` whenever `poutcome` is `"nonexistent"`, but some rows contradict this. However, this does not significantly impact model performance.

# 4. Modelling Approach & Metrics

The API ranks clients, so we use [average precision](https://en.wikipedia.org/w/index.php?title=Information_retrieval&oldid=793358396#Average_precision) as the primary metric to compare and fine-tune models.

We consider logistic regression and random forest models.
In the end random forest wins 👑.

# 5. How to run

Dependencies are managed with `uv`.

> _Note_: All commands below assume you are in the project root.

First, create a virtual environment:
```bash
uv venv
```

To run 🗒️ `notebook.ipynb` locally, install all dependencies 
```bash
uv sync
```

_Note_: I run Jupyter notebooks locally in VS Code using the .venv Python kernel. Other setups may vary.


The rest of these commands don't need an explicit `uv sync`.

### Run the final Training
```bash
uv run train_pipeline.py
```
This generates a pickled model: pipeline_v1.bin (already included in the repository, so retraining is not required for deployment).

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
julxi/ml-zoomcamp-midterm   2025    4fe632388077   6 hours ago    520MB
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
           "age": 58,
           "pdays": 6,
           "default": "unknown",
           "campaign": 1,
           "euribor3m": 4.076,
           "contact": "cellular",
           "day_of_week": "thu",
           "month": "nov",
           "cons.price.idx": 93.2,
           "poutcome": "success",
           "emp.var.rate": -0.1
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
           "age": 58,
           "pdays": 6,
           "default": "unknown",
           "campaign": 1,
           "euribor3m": 4.076,
           "contact": "cellular",
           "day_of_week": "thu",
           "month": "nov",
           "cons.price.idx": 93.2,
           "poutcome": "success",
           "emp.var.rate": -0.1
         }'
```

## 7. Known limitations / next steps

- Using a more tailored metric, such as a cost-reward metric or Precision@k
- Explore a wider range of models
