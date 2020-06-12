# Disaster Response Pipeline Project

### Instructions:
1. Install Dependencies
   - pip install -r requirements.txt for python 2.x
                     OR
   - pip3 install -r requirements.txt for python 3.x
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Website:
![1](https://github.com/shauryabit2k18/ETL-pipeline/blob/master/images/1.png)
![2](https://github.com/shauryabit2k18/ETL-pipeline/blob/master/images/2.png)
![3](https://github.com/shauryabit2k18/ETL-pipeline/blob/master/images/3.png)
![4](https://github.com/shauryabit2k18/ETL-pipeline/blob/master/images/4.png)
![5](https://github.com/shauryabit2k18/ETL-pipeline/blob/master/images/5.png)
![6](https://github.com/shauryabit2k18/ETL-pipeline/blob/master/images/6.png)
