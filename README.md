# Human Resource Analytics Employee Retention


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


<table class="tg">
<thead>
  <tr>
    <td class="tg-1g7j"> <strong> Variable </strong> </th>
    <td class="tg-1g7j"> <strong> Description </strong></th>
    <td class="tg-14gg"> <strong> Type </strong></th>
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

I used Kaplan-Meier Curves to represent the survival rate of employees in the time span of 10 years. To construct this curves, is necessary to define the time interval, in this case 'tenure', and calculate the survival probabilities within each interval.

I used Cox-Proportional Tests to analyze which factors influence the survival time and to find the predictor variables to employee turnover. And, finished the analysis with a classification machine learning model to predict Whether an employee left the company.

**Data Exploration***

During the Exploratory Data Analysis, the first thing I do was checking the data type of the variables, the results are shown below

<figure class="image">
<p align="center">
<img src="https://github.com/AlvaroVillamizar/Employee_Survival_Analysis/blob/main/Images/Results/Variables_info.jpg" width="auto" height="auto">
<figcaption> <strong>Figure 1.</strong> Content summary for the HR-Employee dataset. </figcaption>
</p>
</figure>

We can observed that this dataset does not have Null rows, and most of the variables are numeric, only Department and Salary are categorical, this last variable is classified as Low, Medium, and High.

The next step in the EDA, was to analyze the presence of incorrect data, such as Negative values, outliers, and type errors in the categorical variables. First, a Descriptive statistics of the numeric data is shown below,

<figure class="image">
<p align="center">
<img src="https://github.com/AlvaroVillamizar/Employee_Survival_Analysis/blob/main/Images/Results/Description.jpg" width="auto" height="auto">
<figcaption> <strong>Figure 2.</strong> Summary Descriptive Statistics of the HR-employee dataset. </figcaption>
</p>
</figure>

As we can see, the min value of all the variables are non-negative, and the max values seems to have reasonable values, there are no radical presence of outliers in this dataset. An analysis of outliers is going to be done later.

Finally, the last step in the EDA was to check for duplicates in the data, using the .duplicated() function from the Pandas Library, we found the presence of 3008 duplicated entries in the dataset. Shown below,

<figure class="image">
<p align="center">
<img src="https://github.com/AlvaroVillamizar/Employee_Survival_Analysis/blob/main/Images/Results/Duplicates.jpg" width="auto" height="auto">
<figcaption> <strong>Figure 3.</strong> Duplicated entries in the HR-Employee Dataset. </figcaption>
</p>
</figure>

The appropiate approach would be to ask the dataset owner to check these entries and find out if these entries are unique and legit people or it was a typo error. However the case, in the current situation the approach was to eliminated them, because duplicated data do not provide new information, but it creates incorrect and skew results. The final verification was checking the unique entries in the categorical variables, to find any typo error in these variables, the results are shown below,

<figure class="image">
<p align="center">
<img src="https://github.com/AlvaroVillamizar/Employee_Survival_Analysis/blob/main/Images/Results/Categorical.jpg" width="auto" height="auto">
<figcaption> <strong>Figure 4.</strong> Duplicates in the categorical variables of the HR-Employee dataset. </figcaption>
</p>
</figure>


Moreover, during this exploratory analysis some limitations were evident, one of them are presented in the variable 'time_spend_company', which was crucial for the survival analysis, this variable wasn't collected on a monthly basis. Therefore, some of the keys insights are less accurate, because some problems that can led to turnover can happen earlier than expected. More information of the moment in which the employees left, can improve the reaction time of the action plan to prevent employee turnover.


**Audience Appeal***

**Visualizations (Teaser)***

# Project organization

```
.
├── Images/                             : All plots from the analysis
│    ├── Plots/
│    └── Results/
├── HR-Project.ipynb                    : EDA, Survival Analysis, and ML implementation
└── README.md                           : Report
```

# Turnover Survival Analysis

<font color="red"> Explain a little about what survival analysis is, how is going to help for this data set* </font>





<font color="red"> finished with the objectives of this section* </font>

<font color="red"> Introduce Kaplan-Meier Curve for Tenure* </font>

<font color="red"> Explain each findings in Tenure vs Salary, Department, Promotion, Monthly Hours, Last Evaluation, Satisfaction, Work Accidents, and Number of projects.** </font>
