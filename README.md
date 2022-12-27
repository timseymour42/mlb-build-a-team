## Introduction

	This MLB-Build-a-Team project was originally inspired by my curiosity as a baseball fan - I wanted to know the answer to questions like how much better would a team get if they added Trea Turner in free agency, what would the Yankees record be if Aaron Judge played every outfield position, or would the Mets win every game if Jacob DeGrom were to pitch every inning? Baseball is one of few sports where it is relatively easy to evaluate individual player performance without knowing the strengths and weaknesses of the rest of the team . A pitcher’s dependence on team fielding is the glaring exception to this idea, however modern metrics are good at isolating the outcomes that can be attributed to them. Knowing this, I thought it would be worthwhile to build a model that could predict the success of some creatively designed roster. In doing so, I applied my working knowledge of advanced baseball statistics, investigated the value of different baseball skill sets, and learned how to deploy an application for external users. I now have a functioning app where you can input any combination of player seasons since 1900 and see how many wins my model projects them to earn in a 162 game season! Here’s how I did it.
	
## Data

	In building this model, the first decision to make was what the granularity of a data point should be. My original idea was to take entire seasons and create a regression model that would predict team wins. The problem I quickly found with this approach was the lack of data volume and relevance. Training a model with only a few thousand data points (∼century of baseball with ∼30 team seasons per year) is already not ideal. This is compounded by the fact that season to season data from the early 1900s would not necessarily be representative of how a team with similar statistics could be expected to perform in modern baseball - some of the statistics used in this process are normalized for era, however it stands to reason that a more recent data point is a better data point given that my goal is to compare hypothetical teams to current MLB teams. This approach gives the model more credibility as a forward looking tool for team building. 
	
	Instead of using yearly team data, I decided to use game data going back to 2015. This resulted in a dataset with 30,564 rows of applicable baseball data. I scraped the data from FanGraphs (FanGraphs) because they had a solid variety of advanced statistics that capture performance in all facets of baseball at a game level of detail. The initial untouched dataset can be seen below in figure 1 (Statistics followed by _x are hitting, _y are pitching).
	
![alt-text](https://drive.google.com/file/d/1NVqAG-7kcHiduHC1dBSxi3OWJOeWJaqx/view?usp=share_link)
Figure 1

## Pre-processing
	The win column serves as the binary target variable that will be predicted, so the goal in this portion of the investigation is to decide whether a team will win a game with a given set of advanced statistics capturing their performance. I used sci-kit learn’s StandardScaler to normalize the baseball data and I stored the mean and unit variance for each stat to be able to normalize creatively designed lineups later on. When inputting creatively designed lineups, there are fewer columns than at the time of initial data preprocessing, so the same StandardScaler object could not be repurposed. The data was then split into train, test, and dev sets; the train and dev sets are used in hyperparameter tuning and the test set is used for final model comparison.
	
## Feature Selection

	The feature selection process required a careful examination of how each statistic would help the model generalize to unseen data. Many of the statistics in the original dataset are impacted greatly by luck, but correlate strongly with winning. Weeding out these unreliable stats and identifying the ‘fair’ stats that would be most predictive of winning were my two priorities in this section.
Examining results-oriented statistics impact

	A couple of statistics that stood out as results-oriented were Left On Base Percentage (LOB%) and Batting Average on Balls in Play (BABIP). With LOB%, if you strand fewer players on base, you will be more likely to win a given game. For the most part though, a batter’s likelihood to get a hit is minimally impacted by the presence of baserunners, so LOB% will vary significantly from inning to inning and game to game. With BABIP, a high score will most often mean you are also very likely to win a game. The problem with this stat is that a player has little control over where the ball goes when it leaves the bat - they can only control the quality of contact which is typically measured in launch angle and exit velocity. BABIP penalizes players for a hard-hit out and rewards them for lucky, weak contact. To visualize how these statistics dominate and corrupt a win prediction model as features, I graphed the train and test accuracies of logistic regression models with feature sets as outputted by recursive feature elimination. 
	
Figure 2

	The green and purple lines in Figure 2 represent the model accuracy when LOB% and BABIP were excluded. Including these two results-oriented statistics leads to noticeably better accuracy and the closeness of train and test accuracy would suggest that the model is not overfitting. However, because of the previously discussed luck based nature and high variance of these statistics, past LOB%/BABIP is not predictive of future LOB%/BABIP. The test set in this case has the benefit of hindsight, but if the model were to predict a year into the future using either of these stats, the result would be flawed.

## Feature Importance Analysis

	After removing a few obvious luck-based features from consideration, I chose to assess each feature’s usefulness in prediction using feature importances and correlation to winning. 
	
### Feature Importances
	
	After training Random Forest and AdaBoost ensemble models using the remaining feature set, I obtained feature importances from each as can be seen in figure 3.

Figure 3

For both models, Earned Run Average (ERA) is deemed the most important to win prediction. This stat is also results oriented in nature and will be discussed in greater detail soon. Other noteworthy takeaways are AdaBoost’s high importance for Home Runs allowed per nine innings (HR/9) and its complete suppression of Weighted Runs Created (wRC+) which is a well regarded offensive statistic. AdaBoost seems to ignore repetitive statistics, so this finding is taken with a grain of salt.  




### Correlation

 	The next step was to assess each feature’s correlation with each other feature and with winning. Beginning with feature to feature correlation, I separated the statistics into pitcher data and position player data because they are independent of one another. 
	
Figure 4

	One area of focus from the pitching stats heat map is the strong negative correlation between Fielding Independent Pitching (FIP) and Wins Above Replacement (WAR). FIP is actually used in the calculation of pitching WAR, so this is no surprise, but it does highlight the need to choose between the two for inclusion in the model. 
	
WAR = [[([(League “FIP” – “FIP”) / Pitcher Specific Runs Per Win] + Replacement Level) * (IP/9)] * Leverage Multiplier for Relievers] + League Correction

	Looking at each statistics’s correlation with winning in figure 5, pitching WAR has a stronger correlation with winning than FIP. WRC+, wOBA, and ERA have the strongest correlations with winning of all the statistics. One other consideration in this feature selection process is that it would be preferable to include at least one statistic representing each facet of the game - Defensive Runs Above Average (Def) and Base Running (BsR) are the only defensive and baserunning advanced statistics that are kept on a game to game basis, so I chose to include these two to increase feature diversity. These stats had relatively weaker correlations with winning  (0.13 and .07)


Figure 5

## Deciding on a Pitching Statistic

	From the feature importance analysis and my knowledge of the similarities of these statistics, I chose to move forward with wRC+, SLG, BsR, Def, and HR/9. The last step in the feature selection process was to decide on one more pitching statistic to use between WAR, FIP, and ERA. My method of choosing between the three was to evaluate model performance with each of the three stats using logistic regression as a baseline model. Each model was trained using the five features that were decided upon along with one of the pitching stats.

Figure 6

	As displayed in figure 6, the ERA model is the best performing model across the board in terms of accuracy, precision, recall, F1 Score, and area under the curve (AUC). WAR clearly outperforms FIP as the second best metric. As previously mentioned, ERA is another results oriented stat that is biased by luck and team circumstance. ERA is calculated using the earned runs a pitcher allows, meaning that a run is only attributed to them if it was not caused by an error. A common misconception is that this detail properly accounts for defense. An error will only be credited to a defender when they are expected to make the play; if a defender makes an exceptional play for their pitcher and records an out, it would not have otherwise been recorded as an error. This is why pitchers with higher quality defenses behind them are at a tremendous advantage when it comes to ERA. Per FanGraphs, the Yankees led the league with 129 defensive runs saved (DRS) and the Giants finished last with -53 DRS in 2022. Giants pitchers are put at an extreme disadvantage in ERA because player fielding is not in their control. FIP and WAR, by contrast, only consider outcomes that the pitcher has control over in their calculations (HR, K, BB) so they generalize better to a new defense. Pitching WAR is thus the feature used in the final model. One shortcoming of FIP and WAR worth mentioning is that it cannot account for the tremendous value a catcher adds in framing pitches. 
	
	The finalized app allows the user to display scatter plots visualizing the relationships between each feature used and wins for team seasons going back to 1900. These plots give a general sense of how each statistic has contributed to team effectiveness and the importance of each facet of the game throughout baseball history.
	
## Hyperparameter Tuning

	The models that I chose to train, tune, and compare were sci-kit learn’s Logistic Regression, Naive Bayes algorithm, RandomForestClassifier, AdaBoostClassifier, and Multi-layer Perceptron Classifier. The Naive Bayes and Logistic Regression models did not require any hyperparameter tuning as they do not have hyperparameters. The first step that I took in the hyperparameter tuning process was to assess model performance with different hyperparameter values to understand at what point the models could overfit. Using this information, I chose hyperparameter ranges to run grid search on to maximize each model’s performance.
	
	As an example of my process in the first step, figure 7 gives a clear picture of the values for max depth in the Random Forest Classifier that will lead to overfitting. After a max depth of six, there is a sharp separation between training and testing accuracy indicating that models with larger max depths could be unnecessarily complex. 
	
Figure 7

## Results

	After tuning each model, I trained them on the entire train and dev set and evaluated their performance on the hold-out test set. Every classifier was able to predict the winning team at a greater than 80% test accuracy! The AdaBoost Classifier performed the best with an 85.8% test accuracy and a 85.4% F1 Score. The test accuracy also exceeded the train accuracy which is a positive sign that the algorithm generalizes well to unseen data. The MLP, Random Forest, and Logistic Regression classifiers were all within 2.5% of AdaBoost’s accuracy, so there is a clear best predictor, but each classifier is reliable. 

Figure 8

## Modifying the Models to Fit my Task

	As you will remember, the goal of this investigation is to be able to predict a team’s win total across an entire season rather than just whether or not they will win one game. With this in mind, my intuition was to use the predict_proba function for each model to see if these values would be representative of season win percentages. This function outputs the probability that the model classifies a data point as a win instead of a 0 or 1 concrete value. To do this, I loaded team data from FanGraphs for seasons since 2015 and scaled the data to be predicted upon by my models. For this task, I trained each model on the entire dataset and predicted upon the new team season data points, multiplying each output by 162 to mimic a full season.
	
Figure 9

	Both the Naive Bayes and Logistic Regression models appear to have linear relationships for their predict_proba predictions. The MLP Classifier behaved curiously in this analysis: this model has a large cluster of data points near the y-axis. It is possible that the model prioritized correctly predicting extreme positive examples meaning any given team’s season long statistical averages would not often meet the criteria for a win in a single game. Notably, the 2020 Dodgers that were on a 116 win pace were given a 2.5% chance to win a game. The Random Forest Classifier win predictions had an inconsistent relationship with wins making it a poor fit for my task. The AdaBoost Classifier requires a closer look as the predictions fall between 78 and 82 wins, so it is hard to tell the relationship from figure 9.
	
Figure 10

	There is clearly not a very strong relationship between win prediction probability and total wins for the AdaBoostClassifier upon a closer look in figure 10. 

Figure 11

	The clear choice for predicting wins across a full season is the Logistic Regression model. The next step in my process was to create a linear function that modifies the Logistic Regression prediction to minimize the error in win prediction. To accomplish this, I use a single feature linear regression. Making predictions on predictions is not an ideal data science practice, however the linear relationship between logistic regression win prediction season totals and actual wins is incredibly strong with a correlation coefficient greater than .9 as per figure 11. For the 2015-2022 seasons, the average error in win total is 4.38 and the median error is 3.52. These values represent a huge success for the win prediction task as my model can be expected to make a prediction within four wins of an actual win total.

## User Interface

	With an effective win prediction model in hand, I wanted to create a user interface that any MLB fan could use to explore their own curiosities. Specifically, I wanted a fan to be able to choose any combination of players' seasons throughout baseball history and get an estimate of how many wins that team would have. For organizational purposes and for simpler aggregation of statistics, I built Hitter and Pitcher class objects. 

Figure 12

	Included in figure 12 is a code snippet of how a Pitcher object is initialized. The Hitter and Pitcher class also have getters, setters, and methods to scale a player’s production to a user-selected games played or innings pitched contribution. This structure allows the model to consume any number of games played or innings pitched for each player as it can assign a proportion to their impact on a hypothetical team. 
	
	When a user interacts with the Dash app, they must populate their lineup and rotation by selecting players, seasons played for those players, and games played or innings pitched for the hypothetical team. You have to use all 1458 games and all 1458 innings in your lineup and rotation to be able to submit your team. Once you submit, you will see a projected win total for your team and there will be a data point on the PlotlyExpress scatter plot for your team to see how they compare to any era or subset of teams.
	
	As an example, figure 13 displays what it would look like if 2022 Aaron Judge played all nine positions and 2021 Corbin Burnes threw every inning for a team. My model predicted that this team would finish 129-33. Clearly, there are no positional requirements when building out your lineup because the Def statistic is aggregated assuming the player will play each position as well as his primary position.

Figure 13

	There is also a search feature at the bottom of the app where you can look through a player’s career statistics before adding them to your squad.
	
## Ensuring Sustainability and Limiting FactorsEnsuring Sustainability and Limiting Factors

	In creating a sustainable, user-friendly application I faced a variety of challenges before coming up with a final product. As a means of ensuring that the data in my app would refresh throughout an MLB season, I built out a local MySQL database and a script that would scrape and update the database (entitled Automated_Data_Collection.py in my GitHub). I was able to schedule the database update using the Windows Task Manager, although the local MySQL database could not be read from when I wanted to deploy the app for exterior use. I explored the possibility of replicating my database in AWS, however I quickly exceeded the free data usage and opted to read from static data for the time being. 
	
	Another challenge that I faced was the speed that my app loaded. The first main obstacle was loading player data into a dash_table object. My original intention was to include all player seasons since 1900 in the table and allow the user to filter down to find the information that they were seeking. With over 88,000 player seasons, however, the Render app encountered an error loading the layout. This amount of data was already testing the limits of what a dash data table could handle, so I chose to require users to add players they wanted to see instead of letting them scroll through pre-populated records. Even after this change, the app still ran very slowly with the amount of data that was stored in each player dropdown. To ensure a more user-friendly experience, I created a quick load version of the app to go along with the full version.

