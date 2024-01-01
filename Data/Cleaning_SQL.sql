SELECT 
	satisfaction_level,
	last_evaluation,
	number_project,
	average_monthly_hours,
	tenure,
	work_accident,
	turnover,
	promotion_last_5years,
	CASE 
		WHEN department = 'sales' THEN 0
		WHEN department = 'accounting' THEN 1
		WHEN department = 'hr' THEN 2
		WHEN department = 'technical' THEN 3
		WHEN department = 'support' THEN 4
		WHEN department = 'management' THEN 5
		WHEN department = 'IT' THEN 6
		WHEN department = 'product_mng' THEN 7
		WHEN department = 'marketing' THEN 8
		ELSE 9
	END AS department,
	CASE
		WHEN salary ='low' THEN 0
		WHEN salary = 'medium' THEN 1
		ELSE 2
	END AS salary
INTO #HR_data
FROM HR_comma_sep

WITH DUPLICATES AS (
    SELECT *, 
	ROW_NUMBER() OVER (PARTITION BY satisfaction_level, last_evaluation,
	number_project, average_monthly_hours, tenure, work_accident, turnover, promotion_last_5years,
	salary, department ORDER BY satisfaction_level) AS RowNum
    FROM #HR_data
)
DELETE FROM DUPLICATES
WHERE 
	RowNum >1;

SELECT
	MIN(average_monthly_hours),
	MAX(average_monthly_hours)
FROM 
	#HR_data


DECLARE @STEP1 FLOAT = (310- 96)/5,
		@STEP2 FLOAT = (1.00 - 0.09)/3,
		@STEP3 FLOAT = (1.00 - 0.36)/3
SELECT
	*,
	CASE
		WHEN (average_monthly_hours < 96 + @STEP1) THEN 'Very Low'

		WHEN (average_monthly_hours >= 96 + @STEP1)
		AND (average_monthly_hours <= 96 + 2*@STEP1) THEN 'Low'

		WHEN (average_monthly_hours >= 96 + 2*@STEP1)
		AND (average_monthly_hours <= 96 + 3*@STEP1) THEN 'Medium'

		WHEN (average_monthly_hours >= 96 + 3*@STEP1)
		AND (average_monthly_hours <= 96 + 4*@STEP1) THEN 'High'

		ELSE 'Very High'
		END AS monthly_hours_cat,
	CASE
		WHEN (satisfaction_level < 0.09 + @STEP2) THEN 'Low'

		AND (satisfaction_level <= 0.09 + 2*@STEP2) THEN 'Medium'

		ELSE 'High'
		END AS satisfaction_cat,
	CASE
		WHEN (last_evaluation < 0.36 + @STEP3) THEN 'Low'

		WHEN (last_evaluation >= 0.36 + @STEP3)
		AND (last_evaluation <= 0.36 + 2*@STEP3) THEN 'Medium'

		ELSE 'High'
		END AS evaluation_cat
FROM #HR_data


DROP TABLE #HR_data
