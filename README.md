# Setting

Don't you love having phone calls from a calling center? Imagine working for one – as a manager! You have too many ~~victims~~ customers too choose from each day so you need to prioritize. That's where we come in, we will provide an API given each customer a success score for how likely the call will succeed into a subscription to a loan (Have I forgotten to mention that you are also working for a bank?), helping to prioritize your precious workers efforts.

# 1. Problem description

The data is from 
>[Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, In press, http://dx.doi.org/10.1016/j.dss.2014.03.001

We use it to select features and train a binary classification model, which then can be queried from the Telemarketing team to rank the success likelihood of their next client calls. So we are not looking for a yes/no decision but a score for ranking the customers success rate.

The workflow is like this
```
eda + feature selection -> finetuning logistic regression model and random tree model -> exporting best model to trainings script and pickle it -> create deployment script -> put it into a container -> publish contaier -> setup cloud-interface
```

# 2. Data Description

The data is in 📁 `data`.
Here is an excerpt of `data/bank-additional-names.txt` for shorts descriptions of the 20 features:

- bank client data:

    1. age : age of client (numeric)
    2. job : type of job (categorical)
    3. marital : marital status (categorical)
    4. education (categorical)
    5. default: has credit in default? (categorical)
    6. housing: has housing loan? (categorical)
    7. loan: has personal loan? (categorical)

- related with the last contact of the current campaign:

    8. contact: contact communication type (categorical) 
    9. month: last contact month of year (categorical)
    10. day_of_week: last contact day of the week (categorical)
    11. duration: last contact duration, in seconds (numeric)

- other attributes:

    12. campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
    13. pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
    14. previous: number of contacts performed before this campaign and for this client (numeric)
    15. poutcome: outcome of the previous marketing campaign (categorical)


- social and economic context attributes

    16. emp.var.rate: employment variation rate - quarterly indicator (numeric)
    17. cons.price.idx: consumer price index - monthly indicator (numeric)     
    18. cons.conf.idx: consumer confidence index - monthly indicator (numeric)     
    19. euribor3m: euribor 3 month rate - daily indicator (numeric)
    20. nr.employed: number of employees - quarterly indicator (numeric)

> ⚠️ **Important:** For the API call the features related with the last contact `contact`, `month`, `day_of_week`, and `duration` are for the call that is about to be made. However, the duration is not known before a call is performed. Thus, this input has to be discarded if the intention is to have a realistic predictive model.

# 3. EDA Summary

The data itself looks solid. There are no missing values or strange values for categorical features.

Except one inconsitency with `pdays`. It should be 999 when `poutcome` has the value nonexistent, but there are some rows which contradict this. In the end it doesn't affect the auc-roc scores.

Also after a greedy feature selection (or rather removal) we will use only these 7 features for training:
`cons.price.idx`,`contact`,`emp.var.rate`,`euribor3m`,`month`,`pdays`,`previous`

# 4. Modeling approach & metrics

We KISS the metric and use the simple auc-roc score for comparing model performance.

We do finetuning runs for linear regression models and also for a random forest. The winner is random forest.

# 5. How to run

The dependencies are managed via `uv`. How you install uv is up to you. I have installed it as a global packet manager, but it can also be installed in a virtual environment somehow, but I don't know the details.

To run the `notebook.ipynb` you have to install all dependencies via
```bash
uv sync
```
### Running `train_pipeline.py`
To run the exported training script `train_pipeline.py` you have to run
```bash
uv run train_pipeline.py
```
This creates a pickled `pipeline_v1.bin` (it's also included in this repository, so running the previous file is not mandatory for deployment)

### Run server locally
To run deployment just run
```bash
uvicorn server:app --port 8000
```

### Run server in Docker

To run the server in docker you either have to build the dockerfile or download the prebuild file from my repository.

#### A. Build locally

```bash
docker build -t julxi/ml-zoomcamp-midterm:2025 .
```

#### B. Download from repository

```bash
docker pull julxi/ml-zoomcamp-midterm:2025
```

#### Running the docker image

Make sure that you have downloaded or build the docker image. You should see something like this.
```bash
jx@hope:~/projects/machine-learning-zoomcamp-2025-midterm$ docker images
REPOSITORY                  TAG       IMAGE ID       CREATED        SIZE
julxi/ml-zoomcamp-midterm   2025      4fe632388077   6 hours ago    432MB
```

Then you can run the image like this

```bash
docker run -p 9696:9696 julxi/ml-zoomcamp-midterm:2025

```

# 6. API usage example

These scripts assume that you run the server or docker image on the ports given as above.

Example api call for the server
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"cons.price.idx": 93.918, "contact": "cellular", "emp.var.rate": 1.4, "euribor3m": 4.957, "month": "jul", "pdays": 999, "previous": 0}'
```

For the runnig docker container
```bash
curl -X POST "http://localhost:9696/predict" \
     -H "Content-Type: application/json" \
     -d '{"cons.price.idx": 93.918, "contact": "cellular", "emp.var.rate": 1.4, "euribor3m": 4.957, "month": "jul", "pdays": 999, "previous": 0}'
```