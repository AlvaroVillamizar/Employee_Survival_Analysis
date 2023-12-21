<style>
  .blue-text {
    color: blue;
    font-family: 'Courier New', monospace;
  }
</style>

# Human Resource Analytics Employee Retention

**Hook and Overview***

Employee retention is defined as the organization's ability to prevent employee turnover, or the loss of organizational talent over a period of time, either voluntary or involuntary.  According to the Society for Human Resource Management (SHRM), the cost to replace just one employee can be three to four times the position's salary. In the US, the overall cost of employee turnover in 2020, was 630 billion dollars. Keeping turnover rates low helps you avoid these expenses and frees up money in the company budget to invest in people's development, benefits, and more.

**Scope and Objectives***

In this project I analyzed and discussed the main causes for employee turnover by performing survival analysis and building a machine learning model, in order to give an answer to the following questions.

- What is the employee's lifetime in the company?
- What are the significant factors that drive employee turnover?
- How each factor contributes to employee turnover?
- What can be done in order to improve retention?
- How can we predict when an employee is likely to left?

**Dataset Introduction***

The dataset used for this analysis can be found in <a href= "https://kaggle.com/datasets/mfaisalqureshi/hr-analytics-and-job-prediction/data">Kaggle</a>, this dataset is also part of the capstone project in the <a href= "https://www.coursera.org/professional-certificates/google-advanced-data-analytics"> Google Advanced Data Analytics Certificate</a>. This dataset contains employees information collected throughout 10 years, the information inside is distribuited in 15.000 rows and 10 columns, the columns are described as follows:


<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-1g7j{background-color:#ffffff;border-color:inherit;color:rgba(0, 0, 0, 0.87);font-weight:bold;text-align:center;
  vertical-align:middle}
.tg .tg-8spe{background-color:#ffffff;border-color:inherit;color:rgba(0, 0, 0, 0.87);text-align:left;vertical-align:middle}
.tg .tg-14gg{background-color:#ffffff;color:#000000;text-align:left;vertical-align:top}
</style>
<table class="tg">
<thead>
  <tr>
    <td class="tg-1g7j"><span style="font-weight:bold">Variable</span></th>
    <td class="tg-1g7j"><span style="font-weight:bold">Description</span></th>
    <td class="tg-14gg"><span style="font-weight:bold">Type</span></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-8spe">satisfaction_level</td>
    <td class="tg-8spe">Employee-reported job satisfaction level [0–1]</td>
    <td class="tg-14gg">float64</td>
  </tr>
  <tr>
    <td class="tg-8spe">last_evaluation</td>
    <td class="tg-8spe">Score of employee's last performance review [0–1]</td>
    <td class="tg-14gg">float64</td>
  </tr>
  <tr>
    <td class="tg-8spe">number_project</td>
    <td class="tg-8spe">Number of projects employee contributes to</td>
    <td class="tg-14gg">int64</td>
  </tr>
  <tr>
    <td class="tg-8spe">average_monthly_hours</td>
    <td class="tg-8spe">Average number of hours employee worked per month</td>
    <td class="tg-14gg">int64</td>
  </tr>
  <tr>
    <td class="tg-8spe">time_spend_company</td>
    <td class="tg-8spe">How long the employee has been with the company (years)</td>
    <td class="tg-14gg">int64</td>
  </tr>
  <tr>
    <td class="tg-8spe">Work_accident</td>
    <td class="tg-8spe">Whether or not the employee experienced an accident while at work</td>
    <td class="tg-14gg">int64</td>
  </tr>
  <tr>
    <td class="tg-8spe">left</td>
    <td class="tg-8spe">Whether or not the employee left the company</td>
    <td class="tg-14gg">int64</td>
  </tr>
  <tr>
    <td class="tg-8spe">promotion_last_5years</td>
    <td class="tg-8spe">Whether or not the employee was promoted in the last 5 years</td>
    <td class="tg-14gg">int64</td>
  </tr>
  <tr>
    <td class="tg-8spe">Department</td>
    <td class="tg-8spe">The employee's department</td>
    <td class="tg-14gg">object</td>
  </tr>
  <tr>
    <td class="tg-8spe">salary</td>
    <td class="tg-8spe">The employee's salary (U.S. dollars)</td>
    <td class="tg-14gg">object</td>
  </tr>
</tbody>
</table>



**Key Variables***

From the Variables in the previous column, we can highlight left as our targer variable, and time_spend_company (or tenure), because this two variable are esential to perform a survival analysis. Therefore, with Survival Analysis techniques we can uncovered relevant features of employee turnover, in the timeline of 10 years.

**Methodology Overview***

Survival Analysis is a technique used to study the amount of time it takes before an event occurs, usually this events can only occur once (e.g death). For this analysis, we are interested in analyze the length of time before an employee decides to leave the company, find which groups are more likely to left, and why is that.

I used Kaplan-Meier Curves to represent the survival rate of employees in the time span of 10 years. To construct this curves, is necessary to define the time interval, in this case 'tenure', and calculate the the survival probabilities within each interval.

I used Cox-Proportional Tests to analyze which factors influence the survival time and to find the predictor variables to employee turnover. And, finished the analysis with a classification machine learning model to predict Whether an employee left the company.

**Data Exploration***

During the Exploratory Data Analysis, the first thing I do was checking the data type of the variables, the results are shown below

<center>
<img src="Variables_info.jpg" width="300" height="auto">
<figcaption> <strong>Figure 1.</strong> Content summary for the HR-Employee dataset. </figcaption>
</center>

We can observed that this dataset does not have Null rows, and most of the variables are numeric, only Department and Salary are categorical, this last variable is classified as Low, Medium, and High.

The next step in the EDA, was to analyze the presence of incorrect data, such as Negative values, outliers, and type errors in the categorical variables. First, a Descriptive statistics of the numeric data is shown below,

<center>
<img src="Description.jpg" width="600" height="auto">
<figcaption> <strong>Figure 2.</strong> Summary Descriptive Statistics of the HR-employee dataset. </figcaption>
</center>

As we can see, the min value of all the variables are non-negative, and the max values seems to have reasonable values, there are no radical presence of outliers in this dataset. An analysis of outliers is going to be done later.

Finally, the last step in the EDA was to check for duplicates in the data, using the .duplicated() function from the Pandas Library, we found the presence of 3008 duplicated entries in the dataset. Shown below,

<center>
<img src="Duplicates.jpg" width="600" height="auto">
<figcaption> <strong>Figure 3.</strong> Duplicated entries in the HR-Employee Dataset. </figcaption>
</center>

The appropiate approach would be to ask the dataset owner to check these entries and find out if these entries are unique and legit people or it was a typo error. However the case, in the current situation the approach was to eliminated them, because duplicated data do not provide new information, but it creates incorrect and skew results. The final verification was checking the unique entries in the categorical variables, to find any typo error in these variables, the results are shown below,

<center>
<img src="Categorical.jpg" width="auto" height="auto">
<figcaption> <strong>Figure 4.</strong> Duplicates in the categorical variables of the HR-Employee dataset. </figcaption>
</center>


**Limitations***

Some of the limitations that are found within the dataset is that the variable 'time_spend_company', which was crucial for the survival analysis, wasn't collected on a monthly basis. Therefore, some of the keys insights can be less accurate because of this, with more information of the moment in which the employees left, some of the action plan can be deployed promply.

- Or find with more precision, the reasons for turnover.

**Audience Appeal***

**Visualizations (Teaser)***

# Project organization

<font color="red"> Explain briefly how the github is going to be structured </font>

## Turnover Survival Analysis

<font color="red"> Explain a little about what survival analysis is, how is going to help for this data set* </font>

<font color="red"> finished with the objectives of this section* </font>

<font color="red"> Introduce Kaplan-Meier Curve for Tenure* </font>

<font color="red"> Explain each findings in Tenure vs Salary, Department, Promotion, Monthly Hours, Last Evaluation, Satisfaction, Work Accidents, and Number of projects.** </font>
