# Human Resource Analytics Employee Retention

<p align="center">
<img src="https://instaprop.com/blog/img/blog/alex-kotliarskyi-QBpZGqEMsKg-unsplash.jpg" width="400" height="auto">
</p>

Employee retention is defined as the organization's ability to prevent employee turnover, or the loss of organizational talent over a period of time, either voluntary or involuntary. According to the Society for Human Resource Management, the cost to replace just one employee can be three to four times the position's salary. In the US, the overall cost of employee turnover in 2020 was 630 billion dollars. Keeping turnover rates low helps you avoid these expenses and frees up money in the company budget to invest in people's development, benefits, and more.

**Scope and Objectives:** In this project, I analyzed and discussed the main causes for employee turnover by performing survival analysis and building a machine learning model to answer the following questions.

- What is the employee's lifetime in the company?
- What are the significant factors that drive employee turnover?
- How does each factor contributes to employee turnover?
- What can be done to improve retention?
- How can we predict when an employee is likely to leave?

### Dataset Introduction

The dataset used for this analysis can be found <a href= "https://kaggle.com/datasets/mfaisalqureshi/hr-analytics-and-job-prediction/data">here</a>, this dataset is also part of the capstone project in the <a href= "https://www.coursera.org/professional-certificates/google-advanced-data-analytics"> Google Advanced Data Analytics Certificate</a>. The dataset contains employees information collected over 10 years,  distributed across 15.000 rows and 10 columns. The columns are described as follows:


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
    <td class="tg-8spe">work_accident</td>
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
    <td class="tg-8spe">department</td>
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

**Table 1.** Content summary of HR-dataset's variables.

From the previous column we can highlight _left_ as our target variable and _time_spend_company_ (or tenure) because this two variable are essential for performing survival analysis. Therefore, with this analysis technique, we would uncover relevant features of employee turnover over a 10-years timeline.

### Methodology Overview

Survival Analysis is a technique used to study the amount of time it takes before an event occurs, Usually, these events can only occur once (e.g., death). For this project, we are interested in analyzing the length of time before an employee decides to leave the company, finding which groups are more likely to leave, and why.

I used Kaplan-Meier Curves to represent the survival rate of employees over the span of 10 years. To construct these curves, it is necessary to define the time interval, in this case, 'tenure', and calculate the survival probabilities within each time-interval.

Then, I used Cox-Proportional Tests to analyze which factors influence the survival time and find the best predictor variables for employee turnover. Finally, I concluded the analysis with a classification machine learning model to predict if an employee left the company.

### Data Exploration

During the Exploratory Data Analysis, the first thing I did was to check the data type of the variables, the results are shown below

<figure class="image">
<p align="center">
<img src="https://github.com/AlvaroVillamizar/Employee_Survival_Analysis/blob/main/Images/Results/Variables_info.jpg" width="auto" height="auto">
<figcaption> <strong>Figure 1.</strong> Content summary for the HR-Employee dataset. </figcaption>
</p>
</figure>

We observed that this dataset does not have null rows, and most of the variables are numeric. Only Department and Salary are categorical; this last variable is classified as Low, Medium, and High.

The next step in the EDA, was to analyze the presence of incorrect data, such as negative values, outliers, and type errors in the categorical variables. First, a Descriptive statistics of the numeric data is shown below:

<figure class="image">
<p align="center">
<img src="https://github.com/AlvaroVillamizar/Employee_Survival_Analysis/blob/main/Images/Results/Description.jpg" width="auto" height="auto">
<figcaption> <strong>Figure 2.</strong> Summary Descriptive Statistics of the HR-employee dataset. </figcaption>
</p>
</figure>

As we can see, the minimum value of all the variables are non-negative, and the maximum values seems to be reasonable. There are no radical presence of outliers in this dataset.

Then, I checked the presence of duplicates in the data, using the _.duplicated()_ function from the Pandas Library. We found the presence of 3008 duplicated entries in the dataset, shown below

<figure class="image">
<p align="center">
<img src="https://github.com/AlvaroVillamizar/Employee_Survival_Analysis/blob/main/Images/Results/Duplicates.jpg" width="auto" height="auto">
<figcaption> <strong>Figure 3.</strong> First 7 duplicated entries in the HR-Employee Dataset. </figcaption>
</p>
</figure>

The appropriate approach to handle the previous duplicates would be to ask the dataset owner to check these entries and find out if these are either different people or if it was a typo error. However, given the situation, the approach was to eliminated them because duplicated data do not provide new information, and it creates incorrect and skewed results.

Finally, the last verification was to check the entries in the categorical variables to find any typo error. The results are shown below,

<figure class="image">
<p align="center">
<img src="https://github.com/AlvaroVillamizar/Employee_Survival_Analysis/blob/main/Images/Results/Categorical.jpg" width="auto" height="auto">
<figcaption> <strong>Figure 4.</strong> Duplicates in the categorical variables of the HR-Employee dataset. </figcaption>
</p>
</figure>

As we can see, there are no errors in the categorical variables.

Additionally, during the analysis, there was some evident limitations with the dataset. One of them are in the variable _'time_spend_company'_, which was crucial for the survival analysis. This variable wasn't collected on a monthly basis. Therefore, some of the keys insights are less accurate, because with a more refined time variable, the analysis becomes more precise. Moreover, having more information in which the employees left, can improve the reaction time of the action plan to prevent employee turnover.


<p align="center">
<img src="https://github.com/AlvaroVillamizar/Employee_Survival_Analysis/blob/main/Images/Results/HR_Dashboard_page-0001.jpg" width="auto" height="auto">
<figcaption> <strong>Figure 5.</strong> HR Analysis Dashboard. </figcaption>
</p>
</figure>

# Project organization

```
.
├── Data/
│    ├── Cleaning_SQL.sql/              : SQL file with the cleaning process
│    ├── HR_comma_sep.csv/              : Original dataset
│    └── Hr_cleaned.csv/                : Cleaned dataset
├── Images/                             : All plots and tables from the analysis
│    ├── Plots/
│    └── Results/
├── HR-Project.ipynb                    : EDA, Survival Analysis, and ML implementation
├── HR_Dashboard.pbix                   : Power BI Dashboard   
└── README.md                           : Report
```

# Turnover Survival Analysis

The objectives of this analysis is to use Survival Techniques to determine the Employee' lifetime in the span of 10 years to answer the following questions.

- Which situations affect the employee's life time in the company?
- Which actions can be made to prevent this employee loss?
- Which factors are most important for employee turnover?

Our target variable is "_Tenure_" (See **Table 1**). From the **Figure 5** we can see that the number of employees who left the company was 1991, around 17% of the whole crew; this indicates that the out dataset is unbalance, since "Tenure" represent less than 20% of the  whole data.

<figure class="image">
<p align="center">
<img src="https://github.com/AlvaroVillamizar/Employee_Survival_Analysis/blob/main/Images/Plots/Unbalance.png" width="auto" height="auto">
<figcaption> <strong>Figure 6.</strong> Distribution of Employee Turnover. </figcaption>
</p>
</figure>

#

In this analysis each variable (except _Tenure_) was plotted in the y-axis and tenure in the x-axis. Then, their respective Kaplan-Meier Curves was constructed, in order to observed its survival behavior.

### Salary

We know that Compensation is one of the most importat aspect on Employee Retention. In fact, this <a href= "https://www.scirp.org/journal/paperinformation?paperid=126223#:~:text=Studies%20have%20shown%20that%20compensation,to%20stay%20with%20their%20organizations"> study </a> suggest that employees who earn more are more likely to stay with their organizations. In the followed image, the data was grouped according to their respective salary range (Low, Medium, or High), and plot their respective Survival Curve to analyze its relationship.

<figure class="image">
<p align="center">
<img src="https://github.com/AlvaroVillamizar/Employee_Survival_Analysis/blob/main/Images/Plots/Survival_Salary.png" width="auto" height="auto">
<figcaption> <strong>Figure 7.</strong> Survival Curve in Salary groups (left), and number of employees in each group (right). </figcaption>
</p>
</figure>

As we can see, because of the difference between groups, salary plays an important role in Employee retention. The highest number of employee turnover comes from the Low range salary' group, the survival probability was 40%, meaning that 2 out of the 5 employees from this group are likely to stayed beyond the 10 years. and Employees from the Medium group had a survival probability of roughly 55%, meaning that 11 out of 20 employees will stayed beyond the 10 years. Because the majority of the company crew comes from the Low and Medium salary group, this issue could be more detrimental if is not fix promptly.

**Suggestions:** To improve Employee loss because of salary, the organization can start offering competitive compensation packages, either direct or indirect. These packages can be, but isn´t limited to, better health insurance, retirement plans, paid time off, or bonuses.

### Departments

For this analysis, the dataset was group by department in order to find survival patterns related with its departments, then we plot each Kaplan Meier curve to analyze its behavior.

<figure class="image">
<p align="center">
<img src="https://github.com/AlvaroVillamizar/Employee_Survival_Analysis/blob/main/Images/Plots/Survival_Department.png" width="auto" height="auto">
<figcaption> <strong>Figure 8.</strong> Survival Curve in Department groups (left), and number of employees in each group (right). </figcaption>
</p>
</figure>

As we can see from the graph in the left, there exist a survival difference between each department, in fact the departments with the lowest retention expectancy were: Support, Technical and Sales, with a survival probability of roughly 40%, 45%, and 50% respectively. In contrast, the department with the highest retention expectancy was Management, with a survival probability of roughly 75%, meaning that 3 out of 4 employee from this department will stayed in the organization beyond 10 years.

### Promotion

Employees who feel recognized and rewarded tend to work harder and are more likely to stay with their company, but isn't limited, this kind of recognitions boost employees confidence, and drive positive attitudes in the work place (Marschall, 2023). In this analysis, employees were divided in two groups according to whether they were promoted or not, in the last 5 years.

<figure class="image">
<p align="center">
<img src="https://github.com/AlvaroVillamizar/Employee_Survival_Analysis/blob/main/Images/Plots/Survival_Promotion.png" width="auto" height="auto">
<figcaption> <strong>Figure 9.</strong> Survival Curve in Promotion groups (left), and number of employees in each group (right). </figcaption>
</p>
</figure>

The group of employees who didn't received a promotion had a lower life expectancy compared to the other group, after 6 years there was a 50% chance of employee turnover, meaning that halve of this group will stayed in the organization beyond 6 years, and the other halve not. Meanwhile, the other group didn't show any tendency of quitting after 10 years of work. Additionally, the majority of the 1991 employees who left comes from the group who didn't received this recognition.

**Suggestions:** Some expert suggests having a regular date to celebrate employees milestone, this type of events can include, employee of the month, or performance reviews. This kind of activities make employees to perform better for each period of time, or increase the number of promotions in the organization to boost retention.

### Monthly Hours

For this analysis the variable _average_monthly_hours_ was divided into 5 class intervals, in order to crate a new categorical data to compared its survival probability, each group was label as Very Low, Low, Medium, High, and Very High.

<figure class="image">
<p align="center">
<img src="https://github.com/AlvaroVillamizar/Employee_Survival_Analysis/blob/main/Images/Plots/Survival_Hours.png" width="auto" height="auto">
<figcaption> <strong>Figure 10.</strong> Survival Curve in Monthly Hours groups (left), and number of employees in each group (right). </figcaption>
</p>
</figure>

We can observed that employees with "Very High" number of monthly hours tend to quit at the highest rate, in the 10 years span the survival probability of this group was 25%, meaning that 1 out of 4 employees who has very high number of monthly hours will remained in the organization at some point beyond 10 years. This behavior is similar to the employee's group who has a "High" number of monthly hours, in this case the survival probability was 35%, meaning that 7 out of 20 employees will stayed in the organization beyond 10 years. The group with the highest survival probability was the employee's who worked a "medium" number of monthly hours, its survival probability was roughly 80%, meaning that 4 out of 5 employees will stayed in the organization beyond 10 years, this pattern was similar with the remaining two groups. Moreover, we can notice that the critical years in which most employees quit was between the 3rd and 4th year, in this time there was a drop in the survival probability of  27.5% in average.   

### Evaluation

The "last_evaluation" variable followed a similar procedures from the previous variable, this variable was divided into 3 classes: Low, Medium, and High, in order to analyze each group behavior.

<figure class="image">
<p align="center">
<img src="https://github.com/AlvaroVillamizar/Employee_Survival_Analysis/blob/main/Images/Plots/Survival_Evaluation.png" width="auto" height="auto">
<figcaption> <strong>Figure 11.</strong> Survival Curve in Last Evaluation groups (left), and number of employees in each group (right). </figcaption>
</p>
</figure>

In this graph we can observed clearly that the employee's group who had a high score in their last performance review quit at a rapid rate than the others, this could be related to the fact that within the organization there isn't many promotion or incentives for working hard, this group had their biggest drop in the 5th year, with a 35% reduction. Moreover, this group had a survival probability of 30%, meaning that 3 out of 10 employees will stayed in the organization beyond 10 years. The group with the highest probability group correspond to the employees who had a "medium" score in their last performance review, this group had a survival probability of 90%, meaning that 9 out of 10 employees will stayed in the organization beyond 10 years.

### Satisfaction

Similar to the Compensation aspect, many employee's lifetime is linked to its satisfaction within the organization, because  employees who are satisfied with their jobs will build relationships, be productive, and stay with the company (Swofford, 2023). In this analysis, the variable "_satisfaction_level_" was measured continuously, ranging from 0 to 1. Therefore, to analyze it, the data was divided in three 3 classes, each one with the same size in length. Then, each employee was classified as either, Low, Medium or High, according to their respective satisfaction level.  

<figure class="image">
<p align="center">
<img src="https://github.com/AlvaroVillamizar/Employee_Survival_Analysis/blob/main/Images/Plots/Survival_Satisfaction.png" width="auto" height="auto">
<figcaption> <strong>Figure 12.</strong> Survival Curve in Satisfaction groups (left), and number of employees in each group (right). </figcaption>
</p>
</figure>

From the previous graph, we can see that employees with a high satisfaction didn't show any tendency of leaving in the span of 10 years, while the other group had a big decline after 5 years of work, in fact, employees who felt a medium satisfaction were more likely to quit than the group from a low satisfaction. However, the rate at which employees from the low group quit was more rapid in the first 5 years than the medium group.

### Work Accidents

<figure class="image">
<p align="center">
<img src="https://github.com/AlvaroVillamizar/Employee_Survival_Analysis/blob/main/Images/Plots/Survival_Accidents.png" width="auto" height="auto">
<figcaption> <strong>Figure 13.</strong> Survival Curve in Accident groups (left), and number of employees in each group (right). </figcaption>
</p>
</figure>

We can observed that the group who had accidents at work had a lower survival probability with respect the group who didn't, its survival probability was 45%, meaning that 9 out of 20 employees who had accidents at work will stayed in the organization beyond 10 years. Meanwhile, the group who didn't have accidents had a survival probability of roughly 80%, meaning that 4 out of 5 employees will stayed in the organization beyond 10 years.

### Number of Projects

<figure class="image">
<p align="center">
<img src="https://github.com/AlvaroVillamizar/Employee_Survival_Analysis/blob/main/Images/Plots/Survival_Projects.png" width="auto" height="auto">
<figcaption> <strong>Figure 14.</strong> Survival Curve in Projects group (left), and number of employees in each group (right). </figcaption>
</p>
</figure>

We can observed that the group who contributed in a total of 7 projects didn't stay in the organization for more than 5 years, all of them quit at some point during this period. Meanwhile, the groups who contributed in 2, 6 or 5 projects followed similar patterns, this groups had a survival probability of roughly 35%, meaning than 7 out of 20 employees will stayed in the organization beyond 10 years. In contrast, the employee's group who contributed in 3 projects had the highest survival probability, 95%, meaning that 19 out of 20 employees will stayed in the organization beyond 10 years.

**Suggestions:** This contrast suggest that having an excessive and very little amount of work is detrimental for employee retention, according to the results showed in the graphs, the ideal work balance to maintain a healthy work relationship is to let employees contribute in 3 projects. However, having 4 to 5 projects during the first 5 years of work didn't show signs of turnover, but having too much or too little work isn't ideal.

## Cox Model Results

<figure class="image">
<p align="center">
<img src="https://github.com/AlvaroVillamizar/Employee_Survival_Analysis/blob/main/Images/Results/Cox_Model_Results.png" width="auto" height="auto">
<figcaption> <strong>Figure 15.</strong> Cox Proportional Hazard Model Results. </figcaption>
</p>
</figure>

From the previous results we can observed a relationship between the employee's survival time and the predictor variables. The region inside the rectangle red box represent the p-values associated with the z-statistics of the each predictor variable, this score indicates whether or not the evidence suggest that we should reject the null hypothesis for each predictor variable, the Null hypothesis states that: There are no survival trends from the subjects in the different groups.

These scores indicates that salary, promotion, monthly hours, satisfaction level, work accidents, and number of projects had a influence in the survival expectancy. The "exp(coef)", also known as Hazard Ratio, indicates that all the previous variables, except monthly hours had a decreased risk, meaning that for each year, the hazard function change in the respective direction. The next graph shows the change for both the survival probability and cumulative hazard functions.  

<figure class="image">
<p align="center">
<img src="https://github.com/AlvaroVillamizar/Employee_Survival_Analysis/blob/main/Images/Plots/Survival_Hazard.png" width="auto" height="auto">
<figcaption> <strong>Figure 16.</strong> Survival function (left) and Cumulative Hazard function (right). </figcaption>
</p>
</figure>

In the left, we can observed from the Survival function, his behavior is decreasing meaning that through the years the survival probability decrease, and beyond 6 years, the survival probability remain constant with a value of 80%, meaning that there is a 80% chance that an employee will survive beyond this time. Meanwhile, in the right, from the cumulative hazard function we see that its behavior is increasing and beyond 6 years, it becomes constant, with a value of roughly 25%, meaning that beyond 6 years, there is a 25% chance that an employee will leave the company.

# Machine learning Model

In this section, I explore different non-parametric, supervised learning, Tree-based models to predict whether an employee is likely to quit, the chosen models were: Decision Tree, Random Forest and XGBoost. To select the final model I analyzed its results from Precision, Recall, Accuracy, F1 (Harmonic mean) and ROC AUC scores using the validation data, the model who out performs the others is going to be select to make the predictions based on the test data.


### Decision Tree

For our Decision Tree I used GridSearch the set that produces the best hyperparameters' set based on the metrics mentioned above and provide a good balance between bias and variance, this process was divided in 3 stages to prevent overwhelming the machine. The hyperparameters explored were:
- Maximum depth [4, 5, 6, 7, 8, 9, 10, 11, 12, 15]
- Minimum sample leaf [2, 5, 10, 20, 50] and [2, 3, 4, 5, 6, 7]
- Maximum number of features [2, 3, 4, 5, None]
- Maximum Number of leaf nodes [2, 4, 5, 6, 8]
- Criterion [_'entropy'_, _'gini'_]

**Observations:** The model wasn't sensible with the number of features, indicating that there isn't an fixed value for this parameter

The best parameters were:
- Maximum depth [7]
- Minimum sample leaf [2]
- Maximum Number of leaf nodes [8]
- Criterion [_'entropy'_]

The final decision tree is showed below,

<figure class="image">
<p align="center">
<img src="https://github.com/AlvaroVillamizar/Employee_Survival_Analysis/blob/main/Images/Plots/Decision_Tree.png" width="auto" height="auto">
<figcaption> <strong>Figure 17.</strong> Decision Tree with max depth of 2. </figcaption>
</p>
</figure>

This decision tree was plotted until its 2nd level, because the complete tree was too big to analyze. From the previous figure we can observed that based on the _gini_ score, _satisfaction_level_, _last_evaluation_, and  _number_project_, were among the most important features to decided whether an employee was likely to quit.

<figure class="image">
<p align="center">
<img src="https://github.com/AlvaroVillamizar/Employee_Survival_Analysis/blob/main/Images/Plots/DTree_Features.png" width="auto" height="auto">
<figcaption> <strong>Figure 18.</strong> Relevant features to make predictions based on Decision Trees. </figcaption>
</p>
</figure>

In this graph, we can observed the most important features, the importance was computed by the mean of the total reduction of the _gini_ criterion.

### Random Forest

For this model the used of GridSearch was used again, this process was divided in 3 stages to prevent overwhelming the machine. The hyperparameters explored were:

- Number of trees in the forest [65, 70, 75, 85, 95 100, 125, 150],
- Maximum depth [7, 9, 11, 13, 15, None],
- Number of trees in the forest [65, 70, 75, 85, 95 100, 125, 150],
- Minimum Number of samples to split a node [2, 4, 6, 8],
- Criterion [_'entropy'_, _'gini'_]
- Minimum sample leaf [1, 3, 5, 7]

**Observations:** The model wasn't sensible to be affected by min_samples_leaf, min_sample_split, and criterion, indicating that there might be other interactions that affect the significance of the previous parameters.

The best parameters were:
- Number of trees in the forest [95],
- Maximum depth [11],
- Maximum number of features [3]

<figure class="image">
<p align="center">
<img src="https://github.com/AlvaroVillamizar/Employee_Survival_Analysis/blob/main/Images/Plots/RandomF_Features.png" width="auto" height="auto">
<figcaption> <strong>Figure 19.</strong> Relevant features to make predictions based on Random Forests. </figcaption>
</p>
</figure>

This graph shows the most relevant features for Random Forest model, we can compared the relevance of this model with the Decision Tree model, one thing we can notice is that both _number_project_ and _tenure_, were more important features than _last_evaluation_, indicating that work enviroment was more important for employees's turnover.

### XGBoost

A similar process for tunning the parameters was deployed in this model. The hyperparameters explored were:

- Maximum depth [4, 5, 6, 7, 8],
- Learning rate [0.1, 0.2, 0.3],
- Number of boosting stages [75, 100, 125],
- Minimum sum of instance weight in a child [1, 2, 3, 4, 5]

The best parameters were:
- Maximum depth [7],
- Learning rate [0.1],
- Number of boosting stages [100],
- Minimum sum of instance weight in a child [1]

<figure class="image">
<p align="center">
<img src="https://github.com/AlvaroVillamizar/Employee_Survival_Analysis/blob/main/Images/Plots/Final_Features.png" width="auto" height="auto">
<figcaption> <strong>Figure 20.</strong> Relevant features to make predictions based on XGBoosts. </figcaption>
</p>
</figure>

In the previous graph we can observed the most important features for the XGBoost model, the features with the highest contribution in the predictions were: _average_monthly_hours_, _satisfaction_level_, and _last_evaluation_. In contrast, with the other two model, XGBoost considered _average_monthly_hours_ in their most important features.

In the followed graph we can observed the performance of each model with the validation data,

<figure class="image">
<p align="center">
<img src="https://github.com/AlvaroVillamizar/Employee_Survival_Analysis/blob/main/Images/Results/Model_Comparison.png" width="auto" height="auto">
<figcaption> <strong>Figure 21.</strong> XGBoost model Predictions (Left) and ROC curve (Right). </figcaption>
</p>
</figure>

As we observed, the Random Forest model was roughly better than the XGBoost model on Precision, F1, and Accuracy. However, the difference were very little. Because of the results the XGBoost model tend to have on imbalance datasets, the size of the dataset, and the results showed above, I decided to choose the XGBoost model to make our predictions.

### Final Model

In the graph below we observed the confusion matrix from our prediction.

<figure class="image">
<p align="center">
<img src="https://github.com/AlvaroVillamizar/Employee_Survival_Analysis/blob/main/Images/Plots/Results.png" width="auto" height="auto">
<figcaption> <strong>Figure 22.</strong> XGBoost model Predictions (Left) and ROC curve (Right). </figcaption>
</p>
</figure>

Is showed that the model predicted correctly most of the employees who left, because in the right column we have the positive values (both false and true). The model predicted 11 false positive, meaning 11 employees who were categorized as "left the company", but in reality they stayed, and 458 true positives, it means 458 employees who really left the company. In the left column we have the negatives values, the first square indicates the true negatives, our model accurately predicted that 2489 employees stayed in the company, while the remained 40 were categorized as "stayed in the company", but in reality they left. In the left we plotted the ROC AUC graph, this graph indicates the relationship of False positive rate and True positive rate, which was 0.96, and the red dotted line indicates the performance of a random classifier as comparison, which is set as 0.5.


## Reference:

[1] Marschall Amy. (2023). Why Employee Recognition is an Important Part of a Thriving Workplace. Spring Health.  https://www.springhealth.com/blog/why-employee-recognition-is-important#article-heading-2

[2] Meng Kheang Sorn, Adoree R. L. Fienena, Ali, Y., Muhammad Rafay, & Fu, G. (2023). The Effectiveness of Compensation in Maintaining Employee Retention. OAlib, 10(07), 1–14. https://doi.org/10.4236/oalib.1110394
