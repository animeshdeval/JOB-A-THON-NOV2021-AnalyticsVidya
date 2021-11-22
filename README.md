# JOB-A-THON NOV2021 AnalyticsVidya
 Job-a-thon hosted by AnalyticsVidya to predict employee attrition prediction using Survival Analysis

Problem Statement

You are working as a data scientist with HR Department of a large insurance company focused on sales team attrition. Insurance sales teams help insurance companies generate new business by contacting potential customers and selling one or more types of insurance. The department generally sees high attrition and thus staffing becomes a crucial aspect.

To aid staffing, you are provided with the monthly information for a segment of employees for 2016 and 2017 and tasked to predict whether a current employee will be leaving the organization in the upcoming two quarters (01 Jan 2018 - 01 July 2018) or not, given:

    Demographics of the employee (city, age, gender etc.)
    Tenure information (joining date, Last Date)
    Historical data regarding the performance of the employee (Quarterly rating, Monthly business acquired, designation, salary)

Private Leaderboard Rank - 59
Public Leaderboard Rank - 88

Approach

Formulating the problem into a Machine Learning and transforming the data was the most challenging part

Approach 1:

In this approach, I tried to establish a ground-truth with a typical classification problem. Since the attrition of employee was to be predicted for 2 quarters that is from 1 Jan 2018 to 30 June 2018, the target variable was created for training dataset in such a way that if an employee left the organization between 1 July 2017 - 31 Dec 2017, it was labelled as 1, else 0. Now '0' labelled employees will also contain those who have already churned before 1 July 2017, thus those employees were filtered out from the training dataset. Also, the shift of the data with respect to time was done by creating a function. So effectively, the training dataset independent variables will be filtered from 1 Jan 2016 - 30 June 2017 from Monthly joining data ('MMM-YY') and target variable created based on LastworkingDate column between 1 July 2018-31 Dec 2017.  Further the data will shift by 6 months to predict the future attrition of employees. However, this logic failed to create a ground-truth as using hyperparameters of gradient boosting algorithm, logistic regression, randomforest as well as TPOT, F1 score of not more than 0.4 was observed. Thus a new thought was required and was applied which is described in Approach 2

Approach 2:

In this approach, a new direction of using Survival analysis algorithm was employed. Since, I had to omit all those employees who had churn before 31 June 2017 in the approach 1 was something I was not very comfortable with. Thus I decided to explore Survival Analysis algorithm where I could utilize data for all employees to run the model. In this approach LastworkingDate was used to create target 1 (event) for all employees who have resigned. Number of days of employment (duration) was calculated based on LastworkingDate and JoiningDate. After fitting the model, the prediction for next 183 days i.e. 2 quarters of 2018 was calculated based on the probability of attrition (1-p(survival)). Probability of 0.5 has considered as threshold. The predicted output was validated directly on the AnalyticsVidya platform instead of splitting the dataset in train and test. This approach is submitted as the final submission.

Feature Engineering:

    1. df_demograph: dataframe of unique values for demographic data
    2. df_salary_change: dataframe engineered from 'Salary'to derive the 'increment' column
    3. df_promotion: dataframe engineered from 'Joining Designation' and 'Designation' to derive 'Promotion'
    4. df_total: dataframe engineered to derive the total business-value of the employee
    5. df_average: dataframe engineered to derive the total business-value of the employee
    6. df_working_days: dataframe engineered from the final and joining date
    7. df_reporting: dataframe engineered for the column indicating total reporting for each employee
    8. df_target: dataframe created for the target variable with presence of last working date, labeled as 1.

Model Building:

The model CoxPHFitter from lifelines library was used for prediction. The model requires no multicollinearity between independent variables and thus VIF (threshold of 5) was used to remove columns with multicollinearity, further pandas profiling analysis was done to remove further 2 variables. The model was build on feature engineered data and using two important columns-  duration_col ('number_employment_days') and event_col ('target'). The predictions were uploaded on the AnalyticsVidya platform to finalize the model.
