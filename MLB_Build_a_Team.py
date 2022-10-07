#!/usr/bin/env python
# coding: utf-8

# # Data and Analysis Plan: MLB Build a Team
# 
# ## Team 4
# - Tim Seymour (seymour.ti@northeastern.edu)
# 
# ## Project Goal:
# The goal of this project is to be able to predict how many wins any combination of players would have over a 162 game season. To do this, I build a classification model to estimate whether a team will win a given game. To generate a new data point, I will aggregate the statistics for a given lineup, rotation, and bullpen. For this new data point, I will use predict_proba to determine how likely that team is to win a given game - multiplying this value by 162 will give an estimate of how many wins that team would have assuming full health.
# - My original idea was to build a regression model with team win totals over a season as the target, but this severely limits the size of the data set. My preference was to use data from as recently as possible because baseball has changed dramatically throughout its history, so including data from past eras could water down the significance of home run rate and strikeout rate in predicting the winner of a single game.

# <a id='data'></a>
# 
# ## Data
# I will obtain statistics from FanGraphs including:
# 
# 'HR', 'R', 'RBI', 'SB', 'BB%',
#        'K%', 'ISO', 'BABIP_x', 'AVG', 'OBP', 'SLG', 'wOBA', 'xwOBA', 'wRC+',
#        'BsR', 'Off', 'Def', 'WAR_x' (Hitter's war), 'SV',
#        'K/9', 'BB/9', 'HR/9', 'BABIP_y', 'LOB%', 'GB%', 'HR/FB', 'vFA (pi)',
#        'ERA', 'xERA', 'FIP', 'xFIP', 'WAR_y' (Pitcher's war)
# 
# There are considerations to be made about the inclusion of several of these statistics with many relying on luck which should not be projected over a season sample size. This will have a large effect on my feature selection process - stats used should be under team and player control. For example, BABIP or Batting Average on Balls in Play is a metric that will vary game to game and likely has more to do with luck than skill. While exit velocity and defensive alignment will have an impact, it is more practical to use a different statistic for generalization purposes. 
# 
# A stat like Saves also implies a win, so this would lead to overfitting. I will analyze the choices that I make in the feature selection process as I go. One other consideration is that it would be preferable to include at least one statistic representing each facet of the game - Def and BsR are the only defensive and baserunning advanced statistics that are kept on a game to game basis, so provided they have even a 2-3 game impact over a season they will be included in the model.
# 
# As for the inclusion of pitching and hitting statistics, I will make it a priority to include statistics that do not correlate strongly with each other (avoiding autocorrelation) to ensure the model has diverse features.

# In[1]:


from bs4 import BeautifulSoup
import datetime
import pandas as pd
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import plotly
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from jupyter_dash import JupyterDash


# In[2]:


pd.set_option('display.max_columns', None)


# # Data Collection

# Scraping team data from 2015 and later

# In[2]:


def date_to_str(date):
    '''
    Args:
        date (datetime): datetime object for the day of the season

    Returns:
        str: string representation of the given date

    '''
    month = str(date.month)
    day = str(date.day)
    if date.day <= 9:
        day = str(0) + day
    if date.month <= 9:
        month = str(0) + month
    return str(date.year) + '-' + month + '-' + day


# To include advanced stats on a game by game basis, I found FanGraphs to have the best combinations of metrics that would be helpful in predicting the winner of a single game.

# Scraping process takes ~20 minutes

# In[5]:


def collect_team_data():
    
    '''
    Scrapes FanGraphs data from each day between April 1, 2015 and today's date
    Returns:
        hit (pd.DataFrame) contains hitting stats with each record representing one game for a team
        pit (pd.DataFrame) contains pitching stats with each record representing one game for a team
    '''
    # beginning of sample is 2015
    first_date = datetime.datetime(year = 2015, month = 4, day = 1)
    # When date reaches last date, date resets to first_date (plus one year)
    last_date = datetime.datetime(year = 2015, month = 10, day = 3)
    date = datetime.datetime(year = 2015, month = 4, day = 1)
    # collects team hitting stats for each day
    hit = pd.DataFrame()
    # collects team pitching stats for each day
    pit = pd.DataFrame()
    # sustainable way of changing year without change in code
    while (date < datetime.datetime.now()):
        date_str = date_to_str(date)
        print(date_str)
        # scrape hitting data
        hit_df = pd.read_html(f'https://www.fangraphs.com/leaders.aspx?pos=all&stats=bat&lg=all&qual=0&type=8&season={date.year}&month=1000&season1={date.year}&ind=0&team=0%2Cts&rost=0&age=0&filter=&players=0&startdate={date_str}&enddate={date_str}')
        # getting rid of the final row with non-numeric data
        hit_df = hit_df[16][:-1]
        hit_df[('temp', 'Date')] = date_str
        hit_df.columns = hit_df.columns.droplevel(0)
        if len(hit_df['#']) > 1:
            hit = hit.append(hit_df)
        # scrape pitching data
        pit_df = pd.read_html(f'https://www.fangraphs.com/leaders.aspx?pos=all&stats=pit&lg=all&qual=0&type=8&season={date.year}&month=1000&season1={date.year}&ind=0&team=0%2Cts&rost=0&age=0&filter=&players=0&startdate={date_str}&enddate={date_str}')
        # getting rid of the final row with non-numeric data
        pit_df = pit_df[16][:-1]
        pit_df[('temp', 'Date')] = date_str
        pit_df.columns = pit_df.columns.droplevel(0)
        if len(pit_df['#']) > 1:
            pit = pit.append(pit_df)
        if (date < last_date):
            date += datetime.timedelta(days = 1)
        else:
            print(date.year)
            last_date = datetime.datetime(year = last_date.year + 1, month = last_date.month, day = last_date.day)
            first_date = datetime.datetime(year = first_date.year + 1, month = first_date.month, day = first_date.day)
            date = first_date
    return hit, pit


# In[ ]:


hit, pit = collect_team_data()


# In[ ]:


pit.drop(columns = ['G'], inplace = True)
# Joining hitting and pitching dataframes on team and date
all_stats = pd.merge(hit, pit, left_on = ['Team', 'Date'], right_on = ['Team', 'Date'], how = 'inner')
# Excludes data from days where team played a double header
all_stats = all_stats[all_stats.GS == '1']
all_stats.to_csv('daily_stats.csv') 
files.download('daily_stats.csv')


# # Shortened Data Collection

# In[3]:


from google.colab import drive 
drive.mount('/content/gdrive')


# In[4]:


all_stats = pd.read_csv("gdrive/MyDrive/Colab Notebooks/daily_stats.csv", encoding="utf-8")


# # Cleaning

# The data cleaning process consisted of changing the data type of each numerical column from a string to a float. I also dropped columns that did not have values for most game logs ('xwOBA', 'xERA'), columns that do not have anything to do with whether a game is won ('Team', 'G', 'Date', 'GS'), and columns that were disqualified as valid predictors ('PA', 'R', 'L', 'SV', 'IP', 'RBI')

# In[6]:


# These columns have only null values for single games
all_stats.drop(columns = ['xwOBA', 'xERA'], inplace = True)


# In[6]:


# turning every value in the dataframe into a float
def string_to_num(string):
    if(string == 'NA'):
        return 'NA'
    elif(type(string) == str):
        if('%' in string):
            string = string.replace('%', '')
    return float(string)


# In[8]:


# applying the function to each column to ensure all data points are numerical
for col in all_stats.columns:
    if col not in ['Team', 'Date', 'GB']:
        all_stats[col] = all_stats[col].apply(string_to_num)


# ### Subjective decision made to exclude RBI
# RBI is a statistic that is often outside of player control as it is highly dependent on how often runners are in scoring position for their at bats. It also could lead to overfitting as almost all runs result in an RBI and comparing RBIs to ERA would make for too obvious of a prediction. I have not excluded ERA to this point because, while it is a bit more within a player's control and is therefore more representative of their performance.

# In[9]:


all_stats = all_stats.drop(columns = ['#_x', 'Team', 'G', 'PA', 'R',
       'Date', '#_y', 'L', 'SV', 'GS', 'IP', 'RBI', 'Unnamed: 0'])
# Only ~100 columns with null values
all_stats.dropna(inplace = True)


# # Data Pre-Processing

# The first step of preprocessing was to separate the data into X and y with 'W' (win) being the target variable. The next step is to scale X using StandardScaler to normalize the dataset. I then store the factors used in scaling each feature in a dictionary to be used in scaling aggregated player data for prediction. Finally, I split the data randomly into train, test, and validation sets with 15% used for validation.

# In[10]:


X = all_stats.drop(columns = ['W'])
cols = X.columns
y = all_stats['W']


# In[11]:


#Scaling each column to be 
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
X = pd.DataFrame(X, columns = cols)


# In[12]:


#Storing values used to scale each feature for manual normalization in later step
feat_names = np.append(scaler.get_feature_names_out()[:-1], 'WAR')
scales = pd.DataFrame({'Feature': feat_names, 'Unit Variance': scaler.scale_, 'Mean': scaler.mean_})
scales.set_index('Feature', inplace = True)


# In[13]:


X


# In[14]:


X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.15, random_state=100)
#Saving testing set for final model testing
X_train, X_test, y_train, y_test = train_test_split(X_tr, y_tr, test_size=0.25, random_state=100)


# # Feature Selection

# My approach to feature selection is to use recursive feature elimination with a logistic regression model to determine the optimal number of features. Once the results of each model is recorded, I plot the train and test accuracy against the number of features to evaluate for overfitting.

# In[ ]:


# selects the X most important features to be used for the model
def feature_selection(x_train, x_test, y_train, num_feats, print_bool):
    '''
    Args:
          x_train (pd.DataFrame): training set
          x_test (pd.DataFrame): testing set
          y_train (pd.DataFrame): target variable
          num_feats (int): number of features to select
          print_bool (boolean): decides whether or not to print selected features

    Returns:
          X_train_selected (np.array) contains new training set with optimal features selected
          X_test_selected (np.array) contains new testing set with optimal features selected

  '''
    # instantiate
    select = RFE(DecisionTreeRegressor(random_state = 300), n_features_to_select = num_feats)
    
    # fit the RFE selector to the training data
    select.fit(x_train, y_train)
    
    # transform training and testing sets so only the selected features are retained
    X_train_selected = select.transform(x_train)
    X_test_selected = select.transform(x_test)
    
    if print_bool:
      # prints selected features/Sample Output
      selected_features = [feature for feature, status in zip(x_train, select.get_support()) if status == True]
      
      print('Selected features:')
        for feature in selected_features:
            print(feature)

    # returns selected features
    return X_train_selected, X_test_selected;


# In[ ]:


def calc_metrics(y_pred, y_actual, test_or_train, print_bool):
    '''
    input empty string for test_or_train if not relevant
    otherwise put train or test with trailing space

    Args:
        y_pred (pd.DataFrame): values predicted by classifier
        y_actual (pd.DataFrame): actual values
        test_or_train (str): string that will precede metrics to distinguish training, testing measurements
        print_bool (boolean): decides whether or not to print selected features

    Returns:
        results (Dict) contains metrics in format to be added to DataFrame
    '''
    a = accuracy_score(y_actual, y_pred)
    f1 = f1_score(y_actual, y_pred)
    prec = precision_score(y_actual, y_pred)
    rec = recall_score(y_actual, y_pred)
    auc = metrics.roc_auc_score(y_actual, y_pred)
    results = {f'{test_or_train}Accuracy': a, f'{test_or_train}Error': 1-a, f'{test_or_train}Precision': prec, f'{test_or_train}Recall': rec,
          f'{test_or_train}F1 Score': f1, f'{test_or_train}AUC': auc}
    if(print_bool):
        print(f'{test_or_train}Error: {1 - a}')
        print(f'{test_or_train}Accuracy: {a}')
        print(f'{test_or_train}Precision: {prec}')
        print(f'{test_or_train}Recall: {rec}')
        print(f'{test_or_train}F1: {f1}')
        print(f'{test_or_train}AUC: {auc}')
    return results


# Evaluating Logistic Regression Performance with different amounts of features

# In[ ]:


#list of dictionaries to make a DataFrame
dicts = []
# Evaluating models with 1 - 15 features
for num_feat in range(1, 16, 1):
    train, test = feature_selection(X_train, X_test, y_train, num_feat, False)
    LR = LogisticRegression()
    LR = LR.fit(train, y_train)
    # Training predictions
    y_pred_tr = LR.predict(train)
    # Testing predictions
    y_pred_test = LR.predict(test)
    train_results = calc_metrics(y_pred_tr, y_train, 'Train ', False)
    test_results = calc_metrics(y_pred_test, y_test, 'Test ', False)
    train_results.update(test_results)
    train_results['Num Features'] = num_feat
    dicts.append(train_results)
df = pd.DataFrame(dicts)
#Repeating the process excluding result-oriented statistics (LOB%, BABIP)
#list of dictionaries to make a DataFrame
dicts = []
# Evaluating models with 1 - 15 features
for num_feat in range(1, 16, 1):
    train, test = feature_selection(X_train.drop(columns = ['LOB%', 'BABIP_x', 'BABIP_y']), X_test.drop(columns = ['LOB%', 'BABIP_x', 'BABIP_y']), y_train, num_feat, False)
    LR = LogisticRegression()
    LR = LR.fit(train, y_train)
    # Training predictions
    y_pred_tr = LR.predict(train)
    # Testing predictions
    y_pred_test = LR.predict(test)
    train_results = calc_metrics(y_pred_tr, y_train, 'Train ', False)
    test_results = calc_metrics(y_pred_test, y_test, 'Test ', False)
    train_results.update(test_results)
    train_results['Num Features'] = num_feat
    dicts.append(train_results)
df_excluding = pd.DataFrame(dicts)
# Renaming Columns for graphing purposes
for col in df_excluding.columns:
    df_excluding.rename(columns = {col: col + ' excluding LOB%, BABIP'}, inplace = True)
df = pd.concat([df, df_excluding], axis = 1)


# In[ ]:


df


# In[ ]:


px.line(df, x = 'Num Features', y = ['Train Accuracy', 'Test Accuracy', 'Train Accuracy excluding LOB%, BABIP', 'Test Accuracy excluding LOB%, BABIP'], title="Logistic Regression Accuracy with Different Numbers of Features",
            labels={ # replaces default labels by column name,
                'value': "Accuracy", 'variable': 'Accuracy Type'}, range_y = [.88, .93])


# There is not much accuracy to be gained from using more than six or seven statistics 

# ### Feature selection favors result-oriented statistics, unsurprisingly
# - LOB% signifies the percentage of baserunners that did not come around to score; this is a testament to the value of clutch hitting, although for this model this should not be factored in - timely hitting is more likely to be a result of luck
# - BABIP_y is the percentage of balls put in play that result in hits which has an element of luck and is not as representative of the strength of a team or player as batting average itself
# 
# I have decided against using these metrics for the reasons I outline above and because as is seen in the graph, there is minimal gain in accuracy from their use.

# In[ ]:


feature_selection(X_train.drop(columns = ['LOB%', 'BABIP_x', 'BABIP_y']), X_test.drop(columns = ['LOB%', 'BABIP_x', 'BABIP_y']), y_train, 10, True);


# In[ ]:


X_train = X_train.drop(columns = ['LOB%', 'BABIP_x', 'BABIP_y'])
X_test = X_test.drop(columns = ['LOB%', 'BABIP_x', 'BABIP_y'])


# The next means of feature selection used was to use to train random forest and adaboost classifiers to access the feature importance attribute. ERA was the most important feature as was expected because most runs are earned runs, so it is nearly one to one with runs given up.

# In[ ]:


RFC = RandomForestClassifier() 
RFC.fit(X_train, y_train)

feature_importances = RFC.feature_importances_
feature_importances


# In[ ]:


ADA = AdaBoostClassifier() 
ADA.fit(X_train, y_train)

feature_importances_ada = ADA.feature_importances_
feature_importances_ada


# In[ ]:


sorted = np.argsort(feature_importances)[::-1]

plt.title('Random Forest Feature Importances')
plt.bar(range(X_train.shape[1]), feature_importances[sorted], align='center')
plt.xticks(range(X_train.shape[1]), X_train.columns[sorted], rotation=90)
plt.tight_layout()
plt.show()


# In[ ]:


sorted = np.argsort(feature_importances_ada)[::-1]

plt.title('Feature Importances AdaBoost')
plt.bar(range(X_train.shape[1]), feature_importances_ada[sorted], align='center')
plt.xticks(range(X_train.shape[1]), X_train.columns[sorted], rotation=90)
plt.tight_layout()
plt.show()


# Here, I visualize each feature's correlation with every other feature, and then isolate each feauture's correlation with win percentage. Stats like WAR, FIP, and HR/9 stood out as statistics that could be best predictive of winning.

# In[ ]:


corr_to_win = all_stats.corr()[['W']].sort_values('W', ascending = False)


# In[ ]:


# Interesting to note how offensive statistics have higher correlations with one another than pitching statistics
sns.heatmap(all_stats.corr())


# In[ ]:


corr_to_win


# In[ ]:


fig = px.scatter(corr_to_win)
# make y-axis invisible in plot
fig.update_yaxes(title = 'Correlation to W-L%', visible = True, showticklabels = True)
fig.update_xaxes(title = 'Statistic', visible = True, showticklabels = True)

fig


# ERA is dominant in terms of AdaBoost and Random Forest feature importance, although it is not regarded as the best measurement of inidividual pitching performance because it is known to be dependent on the quality of the pitcher's defense. For the purposes of my investigation, I would like to be able to assume a pitcher's defense will be whatever the inputted lineup will offer them, so despite this stat's effectiveness in win prediction, my next step is to look into the tradeoff of using FIP or pitching WAR in its place.

# ## Deciding which pitching statistics to use with Logistic Regression as a baseline model
# FIP, ERA, and WAR_y have many common factors in their calculations, so I have decided to choose only one of the three so as to have diverse features that will not contradict each other in prediction
# - Note: Pitching WAR is calculated using the following formula: 
#   
#   WAR = [[([(League “FIP” – “FIP”) / Pitcher Specific Runs Per Win] + Replacement Level) * (IP/9)] * Leverage Multiplier for Relievers] + League Correction

# In[ ]:


X_train = X_train[['wRC+', 'HR/9', 'BsR', 'FIP', 'WAR_y', 'ERA', 'Def', 'SLG']]
X_test = X_test[['wRC+', 'HR/9', 'BsR', 'FIP', 'WAR_y', 'ERA', 'Def', 'SLG']]


# In[ ]:


list(pitch_stats[0].values())


# In[ ]:


dicts = []
pitch_stats = [{'ERA': ['FIP', 'WAR_y']}, {'WAR_y': ['FIP', 'ERA']}, {'FIP': ['ERA', 'WAR_y']}]
for drop_stats in pitch_stats:
    drop = list(drop_stats.values())
    drop = drop[0]
    stat = list(drop_stats.keys())
    stat = stat[0]
    xt = X_train.drop(columns = drop)
    xte = X_test.drop(columns = drop)
    LR = LogisticRegression()
    LR = LR.fit(xt, y_train)
    # Training predictions
    y_pred_tr = LR.predict(xt)
    # Testing predictions
    y_pred_test = LR.predict(xte)
    train_results = calc_metrics(y_pred_tr, y_train, 'Train ', False)
    test_results = calc_metrics(y_pred_test, y_test, 'Test ', False)
    train_results.update(test_results)
    train_results['Feature Used'] = stat
    dicts.append(train_results)
df = pd.DataFrame(dicts)


# In[ ]:


df


# In[ ]:


px.bar(df, x = 'Feature Used', y = df.drop(columns = ['Feature Used', 'Test Error', 'Train Error']).columns, title="Logistic Regression Accuracy with Different Pitching Stats",
            labels={ # replaces default labels by column name,
                'value': "Accuracy", 'variable': 'Accuracy Type'}, barmode = 'group', range_y = [.77, .92])


# Despite the ERA model's better performance across the board, I will use the WAR model because it is a better representation of a single pitcher's performance rather than the defense and pitching combined of a team. FIP is part of the calculation of WAR, so the WAR model is the obvious choice considering it is a better indicator of team success (which is expected as it is normalized for competition)

# # Finalizing Training and Testing Sets

# In[15]:


X_train = X_train[['wRC+', 'HR/9', 'BsR', 'WAR_y', 'Def', 'SLG']]
X_test = X_test[['wRC+', 'HR/9', 'BsR', 'WAR_y', 'Def', 'SLG']]
X_tr = X_tr[['wRC+', 'HR/9', 'BsR', 'WAR_y', 'Def', 'SLG']]
X_te = X_te[['wRC+', 'HR/9', 'BsR', 'WAR_y', 'Def', 'SLG']]
X = X[['wRC+', 'HR/9', 'BsR', 'WAR_y', 'Def', 'SLG']]


# <a id='visualization_2'></a>
# 
# # VISUALIZATION

# In[ ]:


import plotly.express as px

px.scatter(team_data, x='BsR', y='W-L%', hover_data = ['Team'], color = 'Season')


# In[ ]:


px.scatter(team_data, 'wRC+', 'W-L%', hover_data = ['Team'], color = 'Season')


# In[ ]:


px.scatter(team_data, 'FIP', 'W-L%', hover_data = ['Team'], color = 'Season')


# In[ ]:


px.scatter(team_data, 'WAR_y', 'W-L%', hover_data = ['Team'], color = 'Season', log_y=True)


# In[ ]:


px.scatter(team_data, 'Def_x', 'W-L%', hover_data = ['Team'], color = 'Season', log_y=True)


# # Tuning Models

# To tune the models, I build a function that will graph model performance (train and test accuracy) for different hyperparameter values, and I will run GridSearch before deciding which values will perform the best.

# ## Graphing hyperparameter tuning function

# In[ ]:


import matplotlib.pyplot as plt

def graph_hyper(model, val_list, val_name, arguments):
  '''
  Args:
    model: Sklearn object with global variable representing hyperparameter value
    val_list: values for hyperparameter to choose between
    val_name: hyperparameter name for naming axes, titling

  Purpose:
    Graph the testing vs training accuracy for different values of hyperparameters
  '''

    values = {}
    metrics_output = []

    for threshold in val_list:
        arguments[val_name] = threshold
        clf = model(**arguments)
        clf = clf.fit(X_train, y_train)
        y_train_prediction = clf.predict(X_train)
        y_test_prediction = clf.predict(X_test)
        training_accuracy = accuracy_score(y_train, y_train_prediction)        
        testing_accuracy = accuracy_score(y_test, y_test_prediction)

        values = {val_name : threshold,
                  'Training Accuracy' : training_accuracy,
                  'Testing Accuracy' : testing_accuracy}
        metrics_output.append(values)

    return px.line(metrics_output, x= val_name, y = ['Training Accuracy', 'Testing Accuracy'], title=f'Accuracy for Various {val_name} Values', 
                 labels={'variable': 'Accuracy Type', 'value': 'Accuracy'})


# ## Tuning RandomForestClassifier

# In[ ]:


graph_hyper(RandomForestClassifier, [10, 20, 30, 40, 50], 'n_estimators', {'n_estimators': 0, 'criterion': 'entropy', 'random_state': 100})


# From this plot, it is clear there is not much accuracy to be gained from having more than 30 estimators:
# 
# - Testing Accuracy goes from .849 to .851 when using 30 and 40 estimators

# In[ ]:


graph_hyper(RandomForestClassifier, range(1, 25, 1), 'max_depth', {'n_estimators': 30, 'max_depth': 0, 'criterion': 'entropy', 'random_state': 100})


# From this plot, it is clear there is not much accuracy to be gained from having more than 6 as the max depth and the model clearly overfits after this threshold:
# 
# - Testing Accuracy goes from .851 to .861 when using 6 and 10 as the max depth and the gap between training and testing accuracy balloons from .0018 to .0252

# ### Using GridSearch to inform best hyperparameters based on graph output

# In[ ]:


from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
rfc = RandomForestClassifier()
# define the grid of values to search
grid = dict()
grid['n_estimators'] = [20, 30, 40]
grid['max_depth'] = [6, 7, 8]
# define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define the grid search procedure
grid_search = GridSearchCV(estimator=rfc, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy')
# execute the grid search
grid_result = grid_search.fit(X_train, y_train)
# summarize the best score and configuration
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


# To avoid overfitting, I will use a max depth of 6 and 30 for n_estimators

# ## Tuning AdaBoost hyperparameters

# In[ ]:


graph_hyper(AdaBoostClassifier, [10, 20, 30, 50, 60, 70, 80, 90, 100], 'n_estimators', {'n_estimators': 0})


# Testing accuracy actually exceeds training accuracy until 60 estimators were used. 80 estimators had practically the same training as testing accuracy and .004 better testing accuracy than the model with 50 estimators

# In[ ]:


graph_hyper(AdaBoostClassifier, [0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.75, 1.0], 'learning_rate', {'n_estimators': 50, 'learning_rate': 0})


# Testing accuracy actually exceeds training accuracy for the model with a learning rate of 1 and it has the highest accuracy of any model, so a learning rate of 1 is the obvious choice

# ### Using GridSearch to inform best hyperparameters based on graph output

# In[ ]:


ada = AdaBoostClassifier()
# define the grid of values to search
grid = dict()
grid['n_estimators'] = [60, 70, 80, 100]
grid['learning_rate'] = [0.0001, 0.001, 0.01, 0.1, 1.0]
# define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define the grid search procedure
grid_search = GridSearchCV(estimator=ada, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy')
# execute the grid search
grid_result = grid_search.fit(X_train, y_train)
# summarize the best score and configuration
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


# I will use a learning rate of 1 and 80 for n_estimators

# ## MLP Hyperparameter tuning

# In[ ]:


from sklearn.model_selection import GridSearchCV
grid = {'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant','adaptive']}

mlp = MLPClassifier()
# define the grid search procedure
grid_search = GridSearchCV(estimator=mlp, param_grid=grid, n_jobs=-1, cv=3, scoring='accuracy', verbose = 10)
# execute the grid search
grid_result = grid_search.fit(X_train, y_train)
# summarize the best score and configuration
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


# In[ ]:


parameter_space = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}


# ## Gradient Booster Hyperparameter tuning

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
    max_depth=1, random_state=0).fit(X_train, y_train)
clf.score(X_test, y_test)


# ## Comparing tuned model performances

# With hyperparameter values set, this section collects metrics for each model using the final testing set to determine the best performing models.

# In[ ]:


def print_metrics(y_pred, y_actual, test_or_train, print_bool):
    '''
    input empty string for test_or_train if not relevant
    otherwise put train or test with trailing space
    '''
    a = accuracy_score(y_actual, y_pred)
    f1 = f1_score(y_actual, y_pred)
    prec = precision_score(y_actual, y_pred)
    rec = recall_score(y_actual, y_pred)
    auc = metrics.roc_auc_score(y_actual, y_pred)

    if(print_bool):
        print(f'{test_or_train}Error: {1 - a}')
        print(f'{test_or_train}Accuracy: {a}')
        print(f'{test_or_train}Precision: {prec}')
        print(f'{test_or_train}Recall: {rec}')
        print(f'{test_or_train}F1: {f1}')
        print(f'{test_or_train}AUC: {auc}')
    return {f'{test_or_train}Accuracy': a, f'{test_or_train}Error': 1-a, f'{test_or_train}Precision': prec, f'{test_or_train}Recall': rec,
          f'{test_or_train}F1 Score': f1, f'{test_or_train}AUC': auc}


# In[ ]:


for model in [GaussianNB(), LogisticRegression(), RandomForestClassifier(n_estimators = 30, max_depth = 6), AdaBoostClassifier(learning_rate = 1, n_estimators = 80),
              MLPClassifier({'activation': 'relu', 'alpha': 0.05, 'hidden_layer_sizes': 100, 'learning_rate': 'constant', 'solver': 'adam'})]:
    print(model)
    model = model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    print_metrics(y_pred, y_te, '', True)


# In[ ]:


final_results = list()
for model in [GaussianNB(), LogisticRegression(), RandomForestClassifier(n_estimators = 30, max_depth = 8), AdaBoostClassifier(learning_rate = .3, n_estimators = 30),
              MLPClassifier(activation= 'relu', alpha=0.05, hidden_layer_sizes= (100,), learning_rate= 'constant', solver= 'adam')]:
    model.fit(X, y)
    #final y predictions
    y_pred_fin = model.predict(X_test)
    #final y train predictions
    y_pred_fin_tr = model.predict(X)
    #Train results
    tr = print_metrics(y_pred_fin_tr, y, 'Train ', False)
    #Test results
    te = print_metrics(y_pred_fin, y_test, 'Test ', False)
    te.update(tr)
    final_results.append(te)
final = pd.DataFrame(final_results)
  


# In[ ]:


first_column = ['Naiver Bayes', 'Logistic Regression', 'Random Forest Classifier', 'AdaBoostClassifier', 'MLP Classifier']

final.insert(0, 'Model', first_column)
final


# In[ ]:


px.bar(final, x = 'Model', y = ['Train Accuracy', 'Test Accuracy'], barmode = 'group', title='Accuracy for each Model', 
                 labels={'variable': 'Accuracy Type', 'value': 'Accuracy'}, range_y = [.72, .76])


# Each model performs very similarly in terms of test accuracy. The MLP Classifier does achieve the best test accuracy and F1 score, indicating that it was the best performing model by a small margin.

# # Scraping Player Data

# This section includes code that was originally used in scraping data. I now have a seperate script entitled Automated_Data_Collection that refreshes the data in a local MySQL database. The scraping process is time consuming, so I took this step to allow for a better user experience in the app I have created using this model.

# In[16]:


# beginning of sample is 2015
year = 2021
wrc = pd.DataFrame()
pitch = pd.DataFrame()
field = pd.DataFrame()
# sustainable way of changing year without change in code
while year < datetime.datetime.now().year + 1:
    for num in range(int(pd.read_html(f'https://www.fangraphs.com/leaders.aspx?pos=all&stats=bat&lg=all&qual=0&type=8&season={year}&month=0&season1={year}&ind=0&team=0&rost=0&age=0&filter=&players=0&startdate=&enddate=&page=1_50')[16].columns[0][0][-8:-6].strip())):
        # scrape hitting data
        if (num < 1):
            temp = pd.read_html(f'https://www.fangraphs.com/leaders.aspx?pos=all&stats=bat&lg=all&qual=0&type=8&season={year}&month=0&season1={year}&ind=0&team=0&rost=0&age=0&filter=&players=0&startdate=&enddate=&page={str(num + 1)}_50')[16][:-1]   
            temp.columns = temp.columns.droplevel(0)
            wrc_df = temp
        else:
            temp = (pd.read_html(f'https://www.fangraphs.com/leaders.aspx?pos=all&stats=bat&lg=all&qual=0&type=8&season={year}&month=0&season1={year}&ind=0&team=0&rost=0&age=0&filter=&players=0&startdate=&enddate=&page={str(num + 1)}_50')[16][:-1])
            temp.columns = temp.columns.droplevel(0)
            wrc_df = wrc_df.append(temp)
        # getting rid of the final row with non-numeric data above
    wrc_df['Season'] = year
    wrc = wrc.append(wrc_df)
    # scrape pitching data
    for num in range(int(pd.read_html(f'https://www.fangraphs.com/leaders.aspx?pos=all&stats=pit&lg=all&qual=0&type=8&season={year}&month=0&season1={year}&ind=0&team=0&rost=0&age=0&filter=&players=0&startdate={year}-01-01&enddate={year}-12-31&sort=21,d&page=1_50')[16].columns[0][0][-8:-6].strip())):
        # scrape hitting data
        if (num < 1):
            temp = pd.read_html(f'https://www.fangraphs.com/leaders.aspx?pos=all&stats=pit&lg=all&qual=0&type=8&season={year}&month=0&season1={year}&ind=0&team=0&rost=0&age=0&filter=&players=0&startdate={year}-01-01&enddate={year}-12-31&sort=21,d&page={str(num + 1)}_50')[16][:-1]   
            temp.columns = temp.columns.droplevel(0)
            pitch_df = temp
        else:
            temp = (pd.read_html(f'https://www.fangraphs.com/leaders.aspx?pos=all&stats=pit&lg=all&qual=0&type=8&season={year}&month=0&season1={year}&ind=0&team=0&rost=0&age=0&filter=&players=0&startdate={year}-01-01&enddate={year}-12-31&sort=21,d&page={str(num + 1)}_50')[16][:-1])
            temp.columns = temp.columns.droplevel(0)
            pitch_df = pitch_df.append(temp)

        # getting rid of the final row with non-numeric data above
    pitch_df['Season'] = year
    pitch = pitch.append(pitch_df)
    print(year)
    year+=1


# In[9]:


def clean_player_data(hit_df, pitch_df):
    '''
    function intended to make statistics numerical, manually calculate statistics, and set the indices to Name and Season

    Args:
    wrc (pd.DataFrame) contains individual player data by season
    pitch (pd.DataFrame) contains individual pitcher data by season

    Returns wrc, pitch as clean datasets for use in App'''

    # applying the function to each column to ensure all data points are numerical
    for col in hit_df.columns:
        if col not in ['Name', 'Team', 'Season', 'GB', 'Pos']:
            hit_df[col] = hit_df[col].apply(string_to_num)
    for col in pitch_df.columns:
        if col not in ['Name', 'Team', 'Season', 'GB']:
            pitch_df[col] = pitch_df[col].apply(string_to_num)
    #Determining home runs allowed for each player for easier calculation
    pitch_df['HR'] = pitch_df['HR/9'] * pitch_df['IP'] * 9
    #Determining total bases for each player for more accurate slugging percentage calculation
    # First must find at bats by subtracting walks using walk percentage
    # Calculation ignores HBP
    hit_df['AB'] = hit_df['PA'] * (1 - (hit_df['BB%'] * .01))
    # Calculation necessary for determining slugging percentage over multiple seasons
    hit_df['TB'] = hit_df['SLG'] * hit_df['AB']
    pitch_df.set_index(['Name', 'Season'], inplace = True)
    hit_df.set_index(['Name', 'Season'], inplace = True)
    return hit_df, pitch_df


# In[19]:


hit_df, pitch_df = clean_player_data(wrc, pitch)


# # Aggregating statistics and generating predictions for customized lineups

# Here, I create class objects for pitchers in hitters, so that information about different types of players can be processed more easily and in a more organized manner. 

# In[7]:


class pitcher:
    hr = 0
    war = 0
    # parameterized constructor
    def __init__(self, name, seasons, ip):
        pitch_df = ui_pitch_df.set_index(['Name', 'Season'])
        pitcher = pitch_df.loc[name]
        #eliminating seasons that the player did not play in the given range
        played = list()
        for seas in seasons:
            if seas in pitcher.index:
                played.append(seas)
        pitcher = pitcher.loc[played]
        #finds pitcher HR allowed per inning pitched
        self.hr = pitcher['HR'].sum() / pitcher['IP'].sum()
        #finds pitcher WAR per inning pitched
        self.war = pitcher['WAR'].sum() / pitcher['IP'].sum()
        #Number of innings to be pitched by player (hypothetically)
        self.ip = ip
        self.name = name
    
    def display(self):
        print("HR/9: " + str(self.hr / 9))
        print("WAR/9 " + str(self.war / 9))

    def setIP(self, ip):
        self.ip = ip

    def scale(self, innings):
        self.hr = self.hr * innings
        self.war = self.war * innings

    def getHR(self):
        return self.hr

    def getWAR(self):
        return self.war


# In[8]:


class hitter:
    wrcp = 0
    bsr = 0
    defn = 0
    slg = 0
    # parameterized constructor
    def __init__(self, name, seasons, games):
        hit_df = ui_hit_df.set_index(['Name', 'Season'])
        hitter = hit_df.loc[name]
        #eliminating seasons that the player did not play in the given range
        played = list()
        for seas in seasons:
            if seas in hitter.index:
                played.append(seas)
        hitter = hitter.loc[played]
        #wRC+ is normalized season by season, so average is taken across inputted season range; drawback is smaller sample sizes may have greater effect than hoped
        #Users will be able to see season by season stats for the player, so they can use their own intuition to evaluate validity of using a given season for a player
        self.wrcp = hitter['wRC+'].mean()
        #finds slugging percentage
        self.slg = hitter['TB'].sum() / hitter['AB'].sum()
        #finds defense per game
        self.defn = hitter['Def'].sum() / hitter['G'].sum()
        #finds baserunning per game
        self.bsr = hitter['BsR'].sum() / hitter['G'].sum()
        #Number of games to be played by player (hypothetically)
        self.games = games
        self.name = name
    
    def display(self):
        # displays statistics at a 162 game pace
        print("wRC+: " + str(self.wrcp))
        print("BsR: " + str(self.bsr * 162))
        print("Def: " + str(self.defn * 162))
        print("SLG: " + str(self.slg))

    def setGames(self, games):
        self.games = games

    def scale(self, games):
        self.wrcp = self.wrcp * games
        self.bsr = self.bsr * games
        self.defn = self.defn * games
        self.slg = self.slg * games

    def getWRC(self):
        return self.wrcp

    def getBsR(self):
        return self.bsr

    def getDef(self):
        return self.defn

    def getSLG(self):
        return self.slg


# In[22]:


judge = (hitter('Fernando Tatis Jr.', list(range(2000, 2025)), 10))
judge.display()


# In[21]:


def pitcher_df(rotation):
    #Accumulating dictionaries representing players into a list
    ps = list()
    for p in rotation:
        row = {'Name': p.name, 'IP': p.ip, 'HR': p.hr, 'WAR': p.war}
        ps.append(row)
    return pd.DataFrame(ps)

def hitter_df(lineup):
    #Accumulating dictionaries representing players into a list
    hs = list()
    for batter in lineup:
        row = {'Name': batter.name, 'G': batter.games, 'wRC+': batter.wrcp, 'BsR': batter.bsr, 'Def': batter.defn, 'SLG': batter.slg}
        hs.append(row)
    return pd.DataFrame(hs)

def wins_for_team(lineup, rotation, model='standard'):
    '''
    lineup (list) consists of hitter objects
    pitchers (list) consists of pitcher objects
    model (Sklearn object) model trained on team data to be used to classify customized team (using predict_proba)
    '''

    #Scaling each players statistics to have their contribution correspond to their designated innings pitched or games played
    for batter in lineup:
        batter.display()
        #multiplies each batting statistic by the games they play
        batter.scale(batter.games)
        batter.display()
    for p in rotation:
        p.display()
        #multiplies each pitching statistic by the innings they pitch
        p.scale(p.ip)
        p.display()

    #Creating dataframes for simpler aggregation
    lineup = hitter_df(lineup)
    rotation = pitcher_df(rotation)

    #ensuring that there are 1458 games played by position players and innings thrown by pitchers
    games = lineup['G'].sum()
    if games != 1458:
        raise Exception(f'Total Games inputted: {games}, must be 1458')
    ip = rotation['IP'].sum()
    if ip != 1458:
        raise Exception(f'Total IP inputted: {ip}, must be 1458')
    
    stats = dict()
    #when scaled wrc is multiplied by games played for each player to ensure proportionate contribution
    stats['wRC+'] = lineup['wRC+'].sum() / 1458
    #when scaled slg is multiplied by games played for each player to ensure proportionate contribution
    stats['SLG'] = lineup['SLG'].sum() / 1458
    #The equivalent of a single team's defense metric is the sum of their entire lineup's Def (which is why the denominator is 162)
    stats['Def'] = lineup['Def'].sum() / 162
    stats['BsR'] = lineup['BsR'].sum() / 162
    stats['HR/9'] = rotation['HR'].sum() / (9 * 1458)
    stats['WAR'] = rotation['WAR'].sum() / (162)
    reg_stats = stats.copy()
    #Stored normalization factors from team data
    metrics = ['wRC+', 'HR/9', 'BsR', 'WAR', 'Def', 'SLG']
    for stat in metrics:
        stats[stat] = (stats[stat] - scales.loc[stat]['Mean']) / scales.loc[stat]['Unit Variance']
    if (type(model) == str):
        logReg = LogisticRegression().fit(X, y)
        adaBoost = AdaBoostClassifier(learning_rate = .3, n_estimators = 30).fit(X, y)
        wins = logReg.predict_proba([[stats['wRC+'], stats['HR/9'], stats['BsR'], stats['WAR'], stats['Def'], stats['SLG']]])[0][1] * 2
        wins += adaBoost.predict_proba([[stats['wRC+'], stats['HR/9'], stats['BsR'], stats['WAR'], stats['Def'], stats['SLG']]])[0][1]
        wins /= 3
        wins *= 162
        return wins, reg_stats
    else:
        return model.predict_proba([[stats['wRC+'], stats['HR/9'], stats['BsR'], stats['WAR'], stats['Def'], stats['SLG']]])[0][1] * 162, reg_stats


# Each model performs very similarly, but has drastically different predictions; the next step is to use actual seasons for teams and choose the algorithm that minimizes the difference between predicted and actual win total.
# 
# With each model being relatively the same in terms of accuracy, I am looking find the best combination of algorithms in mimicking season long win totals, historically.

# # Collecting team data to compare model predictions to actual full season win totals

# In[31]:


def collect_team_data_yearly(year):

  '''
  Args:
    year (integer): year to start collecting data from
  Collecting team data to use as testing data
  '''
  
  wrc = pd.DataFrame()
  pitch = pd.DataFrame()
  field = pd.DataFrame()
  # sustainable way of changing year without change in code
  while year < datetime.datetime.now().year + 1:
      # scrape hitting data
      wrc_df = pd.read_html(f'https://www.fangraphs.com/leaders.aspx?pos=all&stats=bat&lg=all&qual=0&type=8&season={year}&month=0&season1={year}&ind=0&team=0,ts&rost=0&age=0&filter=&players=0&startdate=&enddate=')
      # getting rid of the final row with non-numeric data
      wrc_df = wrc_df[16][:-1]
      wrc_df[('temp', 'Season')] = year
      wrc_df.columns = wrc_df.columns.droplevel(0)
      wrc = pd.concat([wrc, wrc_df], axis = 0)
      # scrape pitching data
      pitch_df = pd.read_html(f'https://www.fangraphs.com/leaders.aspx?pos=all&stats=pit&lg=all&qual=0&type=8&season={year}&month=0&season1={year}&ind=0&team=0,ts&rost=0&age=0&filter=&players=0&startdate=&enddate=')
      # getting rid of the final row with non-numeric data
      pitch_df = pitch_df[16][:-1]
      pitch_df[('temp', 'Season')] = year
      pitch_df.columns = pitch_df.columns.droplevel(0)
      pitch = pd.concat([pitch, pitch_df], axis = 0)
      year += 1
  return wrc, pitch

w, p = collect_team_data_yearly(1900)


# In[32]:


team_data = pd.merge(w, p, left_on = ['Season', 'Team'], right_on = ['Season', 'Team'], how = 'outer')


# Changing each column to be numerical

# In[33]:


# applying the function to each column to ensure all data points are numerical
for col in team_data.columns:
    if col not in ['Team', 'Season', 'GB']:
        team_data[col] = team_data[col].apply(string_to_num)


# In[34]:


team_data['W'] = team_data['W'] * (162 / team_data['GS'])
# Saving a copy of the scraped data 
saved_team_data = team_data.copy()
team_data = saved_team_data


# In[ ]:


team_data = team_data.rename(columns = {'Def_x': 'Def'})
wins = team_data['W'].reset_index(drop=True)
team_data = team_data[['Team', 'Season', 'wRC+', 'HR/9', 'BsR', 'WAR_y', 'Def', 'SLG']]
team_data
team_data['BsR'] = team_data['BsR'] / 162
team_data['WAR_y'] = team_data['WAR_y'] / 162
team_data['Def'] = team_data['Def'] / 162


# In[ ]:


#Renaming for agreement with scales dataframe
team_data.rename(columns = {'WAR_y': 'WAR'}, inplace = True)
metrics = ['wRC+', 'HR/9', 'BsR', 'WAR', 'Def', 'SLG']
for stat in metrics:
  team_data[stat] = (team_data[stat] - scales.loc[stat]['Mean']) / scales.loc[stat]['Unit Variance']
#Changing WAR back for agreement with model
team_data.rename(columns = {'WAR': 'WAR_y'}, inplace = True)


# In[ ]:


team_data


# In[ ]:


win_preds = pd.DataFrame()
names = ['Naive Bayes', 'Log Reg', 'Random Forest', 'AdaBoost', 'MLP Classifier', 'GBoost']
idx = 0
for model in [GaussianNB(), LogisticRegression(), RandomForestClassifier(n_estimators = 30, max_depth = 8), AdaBoostClassifier(learning_rate = .3, n_estimators = 30),
              MLPClassifier(activation= 'relu', alpha=0.05, hidden_layer_sizes= (100,), learning_rate= 'constant', solver= 'adam'), GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
    max_depth=1, random_state=0)]:
    model.fit(X, y)
    pred = model.predict_proba(team_data.drop(columns=['Team', 'Season'])) * 162
    pred = pred[:, 1]
    win_preds[names[idx]] = pred
    idx+=1
team_data = team_data.reset_index(drop=True)
final = pd.concat([team_data, win_preds], axis = 1)
final['Actual Wins'] = wins
final


# Logistic Regression is the strongest candidate to use for win prediction because of how it minimizes error with reasonable variance. AdaBoost has very minimal variance in its predictions, which might help to offset some of the error at the extremes of win totals (upon statistical investigation, teams with very high win totals tend to be predicted to perform even better and teams with low win totals tend to be predicted to perform even worse)

# In[ ]:


final['LGAD5050'] = (final['Log Reg'] + final['AdaBoost']) / 2
final['LGAD21'] = (2*final['Log Reg'] + final['AdaBoost']) / 3
final['LGAD32'] = (3*final['Log Reg'] + 2*final['AdaBoost']) / 5
final['LGAD12'] = (final['Log Reg'] + 2*final['AdaBoost']) / 3
final['LGAD23'] = (2*final['Log Reg'] + 3*final['AdaBoost']) / 5


# In[ ]:


for col in ['Naive Bayes', 'Log Reg', 'Random Forest', 'AdaBoost', 'MLP Classifier', 'GBoost', 'LGAD5050', 'LGAD21', 'LGAD32', 'LGAD12', 'LGAD23']:
    new_name = col + ' Diff'
    final[new_name] = final[col] - final['Actual Wins']


# In[ ]:


for col in ['Naive Bayes', 'Log Reg', 'Random Forest', 'AdaBoost', 'MLP Classifier', 'GBoost', 'LGAD5050', 'LGAD21', 'LGAD32', 'LGAD12', 'LGAD23']:
    new_name = col + ' Absolute Diff'
    final[new_name] = abs(final[col] - final['Actual Wins'])


# In[ ]:


final.sort_values(by = 'Actual Wins', ascending = False)


# I am checking to see which periods in history are too different to be included in the analysis
# 
# - HR/9, Defense, and SLG are all too different in the years before 1962 to rationalize using these years

# In[ ]:


early = final.loc[final.Season < 1962]
middle = final.loc[(final.Season > 1962) & (final.Season < 2000)]
recent = final.loc[(final.Season > 2000)]
pd.DataFrame({'pre-1962': early.mean(), 'pre-2000, post-1962': middle.mean(), 'post-2000': recent.mean()})


# Excluding seasons before 1962

# In[ ]:


final = final.loc[final.Season > 1962]


# Revisit and ensure statistics are aggregated correclty

# Visualizing the difference between actual win totals and the predictions of each model
# 
# - This graph will be used to determine which model's predict_proba function translates best to an entire season

# In[ ]:


px.scatter(final, x='Actual Wins', y = ['Naive Bayes Diff', 'Log Reg Diff', 'Random Forest Diff', 'AdaBoost Diff', 'MLP Classifier Diff', 'GBoost Diff'])


# In[ ]:


px.bar(pd.DataFrame(final.mean()).loc[['Naive Bayes Absolute Diff', 'Log Reg Absolute Diff',
       'Random Forest Absolute Diff', 'AdaBoost Absolute Diff',
       'MLP Classifier Absolute Diff', 'GBoost Absolute Diff']], title = 'Average Absolute Error in Win Prediction', labels = {'index': 'model', 'value': 'Error (in wins)'}).update_layout(showlegend=False)


# In[ ]:


px.scatter(final, x='Actual Wins', y = ['Naive Bayes Diff', 'Log Reg Diff', 'MLP Classifier Diff'], hover_data=('Team', 'Season', 'wRC+', 'WAR_y'), labels={'value': 'Error (in wins)', 'variable': 'Algorithm'},
           trendline = 'ols', trendline_scope = 'trace')


# In[ ]:


px.bar(pd.DataFrame(final.mean()).loc[['Naive Bayes Diff', 'Log Reg Diff',
       'Random Forest Diff', 'AdaBoost Diff', 'MLP Classifier Diff',
       'GBoost Diff']], title = 'Average Error in Win Prediction', labels = {'index': 'model', 'value': 'Error (in wins)'}).update_layout(showlegend=False)


# In[ ]:


px.histogram(final, x='Actual Wins', y = ['LGAD5050 Diff', 'LGAD21 Diff', 'LGAD32 Diff'], barmode = 'group', hover_data=('Team', 'Season', 'wRC+', 'WAR_y'), labels={'value': 'Error (in wins)', 'variable': 'Algorithm'}, histfunc = 'avg', nbins = 20)


# In[ ]:


px.scatter(final, x='Actual Wins', y = ['LGAD21 Diff'], hover_data=('Team', 'Season', 'wRC+', 'WAR_y'), labels={'value': 'Error (in wins)', 'variable': 'Algorithm'},
           trendline = 'ols', trendline_scope = 'trace')


# In[ ]:


fin = final.loc[final['Actual Wins'] > 81]


# # Final model selection
# 
# - Using two parts logistic regression and one part AdaBoostClassifier yields the smallest error when considering the last 60 years. These algorithms balance each other out - the AdaBoostClassifier is more certain in its predictions (predict_proba values closer to 0 and 1), while the LogisticRegression's predictions are not as absolute.

# In[10]:


get_ipython().system('pip install jupyter-dash')


# need table to view player statistics with sorting capabilities
# - default state contains all seasons
# - seasons picker will filter out seasons, groupby player - (single season check box to allow no aggregation)
#   - Submit button so that it does not update based on season until start and end are confirmed
# - team filter
# - player filter
# 
# need graph to compare customized lineup to other teams historically
# - shows wins on one axis, chosen statistic (wrc default) on the other (may need to store customized team stats prior to normalization
# - ideally, hovering over team data point would show that team's stats in table broken down by player
# 
# need drop down for players with games played slider defaulted to 162, submit button
# - Games to be allocated left text should automatically adjust
# - Start Season Dropdown, end season dropdown
# - Upon submission, player is added to table

# In[38]:


saved_hit_df = hit_df


# In[39]:


saved_pitch_df = pitch_df


# In[40]:


ui_hit_df = hit_df.reset_index()


# In[41]:


ui_pitch_df = pitch_df.reset_index()


# In[42]:


ui_hit_df = ui_hit_df.head()


# In[43]:


ui_pitch_df = ui_pitch_df.head()


# To do next:
# - submit button for completed lineup/rotation
# - organize model so that prediction can be generated for inputted lineups
# - organize layout
# - create scatter plot that shows win total (y) and drop-down for each stat (x) to compare your lineup to other teams historically (need some way for build-a-team to be easily seen)
# - scrape data and store it in a database
# - configure database to automatically store new data day by day
# - writeup for medium
# - organize code for post to github/linkedin

# In[11]:


import plotly.express as px
from jupyter_dash import JupyterDash
from dash import dcc
from dash import html
from dash import dash_table
from dash.dependencies import Input, Output, State
import json
from dash.exceptions import PreventUpdate


# In[45]:


#hover_data=('Team', 'Season', 'wRC+', 'WAR_y')
team_history = saved_team_data[['Team', 'Season', 'wRC+', 'HR/9', 'BsR', 'WAR_y', 'Def', 'SLG', 'W']]


# ## PLAN: isolate the dcc.store inputs to identify what is going so wrong

# ***Chained Callbacks and dcc.Store

# In[12]:


import MySQLdb
db = MySQLdb.connect(host='127.0.0.1', user='root', passwd='', db='mlb_db')
tblchk = db.cursor()
# The year of the latest record in the data table
sql_game_data = pd.read_sql('SELECT * FROM game_data', con = db)
sql_team_data = pd.read_sql('SELECT * FROM team_data', con = db)
sql_hitter_data = pd.read_sql('SELECT * FROM hitter_data', con = db)
sql_pitcher_data = pd.read_sql('SELECT * FROM pitcher_data', con = db)


# In[13]:


sql_col_mapping = {'BB%': 'BB_pct', 'K%': 'K_pct', 'wRC+': 'wRC_plus', 'K/9': 'K_per_9',
       'BB/9': 'BB_per_9', 'HR/9': 'HR_per_9', 'LOB%': 'LOB_pct', 'GB%': 'GB_pct', 'HR/FB': 'HR_per_FB', 'vFA (pi)': 'vFA'}
python_col_mapping = {v: k for k, v in sql_col_mapping.items()}
sql_game_data.rename(columns = python_col_mapping, inplace = True)
sql_team_data.rename(columns = python_col_mapping, inplace = True)
sql_hitter_data.rename(columns = python_col_mapping, inplace = True)
sql_pitcher_data.rename(columns = python_col_mapping, inplace = True)


# In[14]:


def clean_game_data(all_stats):
    all_stats = all_stats[all_stats.GS == '1']
    # These columns have only null values for single games
    all_stats.drop(columns = ['xwOBA', 'xERA', 'vFA (pi)'], inplace = True)
    # applying the function to each column to ensure all data points are numerical
    for col in all_stats.columns:
        if col not in ['Team', 'Date', 'GB']:
            all_stats[col] = all_stats[col].apply(string_to_num)
    all_stats = all_stats.drop(columns = ['Team', 'G', 'PA', 'R',
           'Date', 'L', 'SV', 'GS', 'IP', 'RBI'])
    # Only ~100 columns with null values
    all_stats.dropna(inplace = True)
    X = all_stats.drop(columns = ['W'])
    X = X[['wRC+', 'HR/9', 'BsR', 'WAR_y', 'Def', 'SLG']]
    cols = X.columns
    y = all_stats['W']
    #Scaling each column to be 
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X = pd.DataFrame(X, columns = cols)
    #Storing values used to scale each feature for manual normalization in later step
    feat_names = ['wRC+', 'HR/9', 'BsR', 'WAR', 'Def', 'SLG']
    scales = pd.DataFrame({'Feature': feat_names, 'Unit Variance': scaler.scale_, 'Mean': scaler.mean_})
    scales.set_index('Feature', inplace = True)
    return X, y, scales


# Identify what is going wrong with win calculation in database, fix those records with invalid win totals

# In[15]:


def clean_team_data(team_data):
    # applying the function to each column to ensure all data points are numerical
    for col in team_data.columns:
        if col not in ['Team', 'Season', 'GB']:
            team_data[col] = team_data[col].apply(string_to_num)
    # Saving a copy of the scraped data 
    return team_data


# In[16]:


def clean_player_data(hit_df, pitch_df):
    '''
    function intended to make statistics numerical, manually calculate statistics, and set the indices to Name and Season

    Args:
    wrc (pd.DataFrame) contains individual player data by season
    pitch (pd.DataFrame) contains individual pitcher data by season

    Returns wrc, pitch as clean datasets for use in App'''
    
    hit_df = hit_df[hit_df['wRC+'] != None]
    pitch_df.dropna(inplace=True)
    # applying the function to each column to ensure all data points are numerical
    for col in hit_df.columns:
        if col not in ['Name', 'Team', 'GB', 'Pos']:
            hit_df[col] = hit_df[col].apply(string_to_num)
    for col in pitch_df.columns:
        if col not in ['Name', 'Team', 'GB']:
            pitch_df[col] = pitch_df[col].apply(string_to_num)
    #Determining home runs allowed for each player for easier calculation
    pitch_df['HR'] = pitch_df['HR/9'] * pitch_df['IP'] * 9
    #Determining total bases for each player for more accurate slugging percentage calculation
    # First must find at bats by subtracting walks using walk percentage
    # Calculation ignores HBP
    hit_df['AB'] = hit_df['PA'] * (1 - (hit_df['BB%'] * .01))
    # Calculation necessary for determining slugging percentage over multiple seasons
    hit_df['TB'] = hit_df['SLG'] * hit_df['AB']
    return hit_df, pitch_df


# In[17]:


X, y, scales = clean_game_data(sql_game_data)
ui_hit_df, ui_pitch_df = clean_player_data(sql_hitter_data, sql_pitcher_data)
team_history = clean_team_data(sql_team_data)
team_history = team_history[['Team', 'Season', 'wRC+', 'HR/9', 'BsR', 'WAR_y', 'Def', 'SLG', 'W']]


# In[18]:


ui_hit_df.reset_index(inplace=True)
ui_pitch_df.reset_index(inplace=True)


# In[19]:


ui_hit_df = ui_hit_df.head()
ui_pitch_df = ui_pitch_df.head()


# In[72]:


get_ipython().system('pip install jupyter-dash')


# In[22]:


app = JupyterDash(__name__)

def generate_table(dataframe, id):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns]) ] +
        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(len(dataframe))], id = id
    )
#hitters selected
hit_sel = pd.DataFrame(columns = ['Name', 'Years', 'Games'])
#pitchers selected
pit_sel = pd.DataFrame(columns = ['Name', 'Years', 'Innings'])
#current year
curr_year = datetime.datetime.now().year
first_year = int(ui_hit_df['Season'].min())
games = 1458
innings = 1458


app.layout = html.Div(children=[
    #Storing the hitters selected in a df
    dcc.Store(id = 'sel_tbl', data = [], storage_type = 'memory'),
    #Storing the remaining number of games
    dcc.Store(id = 'games_rem', data = [], storage_type = 'memory'),
    #Storing the pitchers selected in a df
    dcc.Store(id = 'psel_tbl', data = [], storage_type = 'memory'),
    #Storing the remaining number of innings
    dcc.Store(id = 'inn_rem', data = [], storage_type = 'memory'), 
    #Storing team stats to be displayed in scatter plot  
    dcc.Store(id = 'team_stats', data = [], storage_type = 'memory'),
    html.Div(html.Label("MLB Build a Team"), style = {'text-align': 'center', 'font-size': '25px', 'vertical-align': 'top',
                                           'align': 'center', 'width': '100%', 'margin-top': '40px'}),              
    #PLAYER SELECTION
    #Multi DropDown for hitters
    html.Div(children = [
        
        html.Label([
            "Hitter",
            dcc.Dropdown(
                #Dropdown with players to be inputted into algo
                id='hitter-dd-calc', clearable=True,
                multi = False,
                value=[], options=[
                    {'label': c, 'value': c}
                    for c in ui_hit_df['Name'].unique()
                ])
        ]),
        #Start year
        html.Label(['Start Year',
        dcc.Dropdown(
                id='start-year-dropdown', clearable=False,
                value=[curr_year], options=[
                    {'label': c, 'value': c}
                    for c in range(first_year, curr_year + 1, 1)
                ])]), 
        #End year
        html.Label(['End Year',
        dcc.Dropdown(
                id='end-year-dropdown', clearable=False,
                value=[curr_year], options=[
                    {'label': c, 'value': c}
                    for c in range(first_year, curr_year + 1, 1)
        ])]),
        #Add Player Button
        html.Button('Submit', id='submit-hitter', n_clicks=None, type = 'submit'),
        #Clear Player info Button
        html.Button('Clear Player Info', id='clear-player', n_clicks=None),
        #Input Box for games
        html.Label(['Games', dcc.Input(id='game_input', type='number', min=1, max=games, step=1)]),
        #Label for Games Remaining
        #HTML Table populated by DropDown; (Player, Years, Games)
        html.Div(children = [f'Hitters Selected; Games Remaining: {games}'], id = 'game'),
        html.Div(children = [generate_table(hit_sel, 'hit_sel')], id = 'hit_sel_tbl'),
        #Clear lineup button
        html.Button('Clear Lineup', id='clear-lineup', n_clicks=None, style = {'text-align': 'center'}),
        #Multi DropDown for pitchers
        html.Label([
            "Pitcher",
            dcc.Dropdown(
                id='pitcher-dd-calc', clearable=True,
                multi = False,
                value=[], options=[
                    {'label': c, 'value': c}
                    for c in ui_pitch_df['Name'].unique()
                ])
        ]),
        #Start year
        html.Label([ 'Start Year',
        dcc.Dropdown(
                id='start-year-dropdown-p', clearable=False,
                value=[curr_year], options=[
                    {'label': c, 'value': c}
                    for c in range(first_year, curr_year + 1, 1)
                ])]), 
        #End year
        html.Label(['End Year',
        dcc.Dropdown(
                id='end-year-dropdown-p', clearable=False,
                value=[curr_year], options=[
                    {'label': c, 'value': c}
                    for c in range(first_year, curr_year + 1, 1)
        ])]),
        #Add Pitcher Button
        html.Button('Submit', id='submit-pitcher', n_clicks=None),
        #Clear Pitcher info Button
        html.Button('Clear Pitcher Info', id='clear-pitcher', n_clicks=None),
        #Input Box for innings
        html.Label(['Innings', dcc.Input(id='inn_input', type='number', min=1, max=innings, step=1)]),
        #Label for Innings Remaining
        #HTML Table populated by DropDown; (Player, Years, Innings)
        html.Div(children = [f'Pitchers Selected; Games Remaining: {innings}'], id = 'inn'),
        html.Div(children = [generate_table(pit_sel, 'pit_sel')], id = 'pit_sel_tbl'),
        #Clear rotation button
        html.Button('Clear Rotation', id='clear-rotation', n_clicks=None, style = {'text-align': 'center'}),
        #Submit Buttom that is only clickable when innings and games remaining are 0
        html.Button('Submit Team', id='sub-team', n_clicks=None, style = {'margin-left': '55px'}),
        html.Div(children = ['Wins: '], id = 'team-wins-prediction', style={'margin-top': '20px', 'margin-left': '10px'}),
        ], style = {'display': 'inline-block', 'margin-top': '100px', 'vertical-align': 'top'}),
    #Creative Lineup Comparison Graph
    html.Div(children = [
        
        dcc.Graph(id = 'team-wins', style={'width': '90vh', 'height': '90vh', 'text-align': 'center'}),
        #Dropdowns for querying team wins graph
        #Start year
        html.Div(children = [
        html.Div(
            html.Label(['Start Year', 
            dcc.Dropdown(
                    id='start-year-dropdown-g', clearable=False,
                    value=[], options=[
                        {'label': c, 'value': c}
                        for c in range(first_year, curr_year + 1, 1)
                    ])]), style = {'display': 'inline-block', 'width': '20%', 'margin-left': '55px'}), 
        #End year
        html.Div(
            html.Label(['End Year',
            dcc.Dropdown(
                    id='end-year-dropdown-g', clearable=False,
                    value=[], options=[
                        {'label': c, 'value': c}
                        for c in range(first_year, curr_year + 1, 1)
            ])]), style = {'display': 'inline-block', "margin-left": "15px", 'width': '20%'}),
        html.Div(
            html.Label([
                "Team",
                dcc.Dropdown(
                    id='team-graph', clearable=True,
                    multi=True,
                    value=[], options=[
                        {'label': c, 'value': c}
                        for c in ui_hit_df['Team'].unique()
                    ])
            ]), style = {'display': 'inline-block', "margin-left": "15px", 'width': '20%'}),
        html.Div(
            html.Label([ 'Stat',
            dcc.Dropdown(
                    id='stat-dd', clearable=False,
                    value=[], options=[
                        'wRC+', 'HR/9', 'BsR', 'WAR_y', 'Def', 'SLG'
                    ])]), style = {'display': 'inline-block', "margin-left": "15px", 'width': '20%'}),
        ], style = {'display': 'inline-block', 'width': '100%'})

        ], style = {'display': 'inline-block', "margin-left": "50px"}),

    #HITTER SECTION
    # Team Dropdown for hitter table
    html.Div(children = [
    html.Label([
        "Team",
        dcc.Dropdown(
            id='team-hit', clearable=True,
            multi=True,
            value=[], options=[
                {'label': c, 'value': c}
                for c in ui_hit_df['Team'].unique()
            ], style = {'width': '25%'})
    ]),
    # Hitter Dropdown
    html.Label([
        "Hitter",
        dcc.Dropdown(
            id='hitter-dropdown', clearable=True,
            multi = True,
            value=[], options=[
                {'label': c, 'value': c}
                for c in ui_hit_df['Name'].unique()
            ], style = {'width': '25%'})
    ]),
    #hitter research table
    dash_table.DataTable(
       data=ui_hit_df.to_dict('records'), ####### inserted line
       columns = [{'id': c, 'name': c} for c in ui_hit_df.columns], ####### inserted line
        id='htable',
        filter_action='native',
        row_selectable='single',
        editable=False,
        sort_action="native",
        sort_mode="multi",
        column_selectable="single",
        row_deletable=True,
        selected_columns=[],
        selected_rows=[],
        page_action="native",
        page_current= 0,
        page_size= 10,
        hidden_columns = ['AB', 'TB']
    )], style = {'display': 'inline-block'}),
    #PITCHER SECTION
    # Team Dropdown for pitcher table
    html.Label([
        "Team",
        dcc.Dropdown(
            id='team-pitch', clearable=True,
            multi=True,
            value=[], options=[
                {'label': c, 'value': c}
                for c in ui_pitch_df['Team'].unique()
            ], style = {'width': '25%'})
    ]),
    # Pitcher Dropdown
    html.Label([
        "Pitcher",
        dcc.Dropdown(
            id='pitcher-dropdown', clearable=True,
            multi = True,
            value=[], options=[
                {'label': c, 'value': c}
                for c in ui_pitch_df['Name'].unique()
            ], style = {'width': '25%'})
    ]),
    #pitcher research table
    dash_table.DataTable(
       data=ui_pitch_df.to_dict('records'), ####### inserted line
       columns = [{'id': c, 'name': c} for c in ui_pitch_df.columns], ####### inserted line
        id='ptable',
        filter_action='native',
        row_selectable='single',
        editable=False,
        sort_action="native",
        sort_mode="multi",
        column_selectable="single",
        row_deletable=True,
        selected_columns=[],
        selected_rows=[],
        page_action="native",
        page_current= 0,
        page_size= 10,
        hidden_columns = ['HR']
    )
    
])

#HITTER RESEARCH SECTION CALLBACKS
@app.callback(Output('htable', 'columns'), [Input('team-hit', 'value'), 
                                           Input('hitter-dropdown', 'value')])
def update_columns(teams, hitters):
    return [{"name": i, "id": i} for i in ui_hit_df.columns]
    
@app.callback(Output('htable', 'data'), [Input('team-hit', 'value'),
                                        Input('hitter-dropdown', 'value')])
def update_data(teams, hitters):
  '''
  Args: 
    teams: selected teams
    htiters: selected hitters
  '''
  if teams and hitters:
    a= hit_df.loc[(ui_hit_df.Team.isin(teams)) & (ui_hit_df.Name.isin(hitters))]
    return a.to_dict('records')
  elif teams:
    a= hit_df.loc[(ui_hit_df.Team.isin(teams))]
    return a.to_dict('records')
  elif hitters:
    a= hit_df.loc[(ui_hit_df.Name.isin(hitters))]
    return a.to_dict('records')
  return ui_hit_df.to_dict('records')

#PITCHER RESEARCH SECTION CALLBACKS
@app.callback(Output('ptable', 'columns'), [Input('team-pitch', 'value'), 
              Input('pitcher-dropdown', 'value')])
def update_columns(teams, pitchers):
    return [{"name": i, "id": i} for i in ui_pitch_df.columns]
    
@app.callback(Output('ptable', 'data'), [Input('team-pitch', 'value'),
                                        Input('pitcher-dropdown', 'value')])
def update_data(teams, pitchers):
  '''
  Args: 
    teams: selected teams
    htiters: selected hitters
  '''
  if teams and pitchers:
    a= ui_pitch_df.loc[(ui_pitch_df.Team.isin(teams)) & (ui_pitch_df.Name.isin(pitchers))]
    return a.to_dict('records')
  elif teams:
    a= ui_pitch_df.loc[(ui_pitch_df.Team.isin(teams))]
    return a.to_dict('records')
  elif pitchers:
    a= ui_pitch_df.loc[(ui_pitch_df.Name.isin(pitchers))]
    return a.to_dict('records')
  return ui_pitch_df.to_dict('records')

#CALLBACK FOR PLAYER SUBMISSION
@app.callback([Output('hit_sel_tbl', 'children'), Output('game', 'children'),
               Output('sel_tbl', 'data'), Output('games_rem', 'data'),
               Output('submit-hitter', 'n_clicks'), Output('clear-lineup', 'n_clicks'), 
               Output('game_input', 'max')],
              [Input('hitter-dd-calc', 'value'), Input('start-year-dropdown', 'value'),
              Input('end-year-dropdown', 'value'), Input('game_input', 'value'),
              Input('submit-hitter', 'n_clicks'), Input('clear-lineup', 'n_clicks'),
              State('sel_tbl', 'data'), State('games_rem', 'data')])
def update_lineup(hitter, start_year, end_year, game_input, button, cl_button, sel_tbl, gs):
    #clearing the lineup
    if (cl_button):
        hitters = pd.DataFrame(columns = ['Name', 'Years', 'Games'])
        return generate_table(hitters, 'hit_sel'), 'Hitters Selected; Games Remaining: 1458', [], 1458, None, None, 1458 
    if len(sel_tbl) == 0:
        hitters = pd.DataFrame(columns = ['Name', 'Years', 'Games'])
    else:
        hitters = pd.DataFrame(sel_tbl['data-frame'])
    if type(gs) == list:
        gms = 1458
    else:
        gms = gs
    if (hitter and start_year and end_year and game_input and button and (gms - game_input >= 0)):
        years = f'{start_year} - {end_year}'
        hitters = hitters.append({'Name': hitter, 'Years': years, 'Games': game_input}, ignore_index = True)
        gms = gms - game_input
    table = generate_table(hitters, 'hit_sel')
    new_text = 'Hitters Selected; Games Remaining: ' + str(gms)
    df = {'data-frame': hitters.to_dict('records')}
    return (table, new_text, df, gms, None, None, gms)


#CLEARING DROPDOWNS UPON PLAYER SUBMISSION
@app.callback([Output('hitter-dd-calc', 'value'), Output('start-year-dropdown', 'value'),
              Output('end-year-dropdown', 'value'), Output('game_input', 'value')],
              [Input('clear-player', 'n_clicks')])
def reset_dropdowns(button):
    return None, None, None, None  

#PITCHER SELECTION

#CALLBACK FOR PLAYER SUBMISSION
@app.callback([Output('pit_sel_tbl', 'children'), Output('inn', 'children'),
               Output('psel_tbl', 'data'), Output('inn_rem', 'data'),
               Output('submit-pitcher', 'n_clicks'), Output('clear-rotation', 'n_clicks'), 
               Output('inn_input', 'max')],
              [Input('pitcher-dd-calc', 'value'), Input('start-year-dropdown-p', 'value'),
              Input('end-year-dropdown-p', 'value'), Input('inn_input', 'value'),
              Input('submit-pitcher', 'n_clicks'), Input('clear-rotation', 'n_clicks'),
              State('psel_tbl', 'data'), State('inn_rem', 'data')])
def update_rotation(pitcher, start_year, end_year, inn_input, button, cl_button, psel_tbl, inn):
    #clearing the rotation
    if (cl_button):
        pitchers = pd.DataFrame(columns = ['Name', 'Years', 'Innings'])
        return generate_table(pitchers, 'hit_sel'), 'Pitchers Selected; Innings Remaining: 1458', [], 1458, None, None, 1458 
    if len(psel_tbl) == 0:
        pitchers = pd.DataFrame(columns = ['Name', 'Years', 'Innings'])
    else:
        pitchers = pd.DataFrame(psel_tbl['data-frame'])
    if type(inn) == list:
        inns = 1458
    else:
        inns = inn
    if (pitcher and start_year and end_year and inn_input and button and (inns - inn_input >= 0)):
        years = f'{start_year} - {end_year}'
        pitchers = pitchers.append({'Name': pitcher, 'Years': years, 'Innings': inn_input}, ignore_index = True)
        inns = inns - inn_input
    table = generate_table(pitchers, 'pit_sel')
    new_text = 'Pitchers Selected; Innings Remaining: ' + str(inns)
    df = {'data-frame': pitchers.to_dict('records')}
    return (table, new_text, df, inns, None, None, inns)


#CLEARING DROPDOWNS UPON PLAYER SUBMISSION
@app.callback([Output('pitcher-dd-calc', 'value'), Output('start-year-dropdown-p', 'value'),
              Output('end-year-dropdown-p', 'value'), Output('inn_input', 'value'),
              Output('clear-pitcher', 'n_clicks')],
              [Input('clear-pitcher', 'n_clicks')])
def reset_p_dropdowns(button):
      return None, None, None, None, None 

#UPDATING THE GRAPH BASED ON WHICH STAT TO DISPLAY, WHICH TEAM, AND WHICH YEAR
@app.callback(Output('team-wins', 'figure'),
              [Input('team-graph', 'value'), Input('start-year-dropdown-g', 'value'),
              Input('end-year-dropdown-g', 'value'), Input('stat-dd', 'value'),
              Input('team_stats', 'data'), Input('sub-team', 'n_clicks')])
def update_figure(team, sy, ey, stat, team_stats, sub_team):
    a = team_history.copy()
    s = 'wRC+'
    if team and sy and ey:
        a = a.loc[(a.Team.isin(team)) & (a.Season >= sy) & (a.Season <= ey)]
    elif sy and ey:
        a = a.loc[(a.Season >= sy) & (a.Season <= ey)]
    elif team:
        a = a.loc[(a.Team.isin(team))]
    if stat:
        s = stat
    a['my_team'] = False
    if sub_team and team_stats:
        team_stats['my_team'] = True
        a = a.append(team_stats, ignore_index = True)
    
    fig = px.scatter(a, x = s, y = 'W', hover_data=('Team', 'Season', 'wRC+', 'HR/9', 'BsR', 'WAR_y', 'Def', 'SLG'), 
                    color = 'my_team')
    return fig

#SUBMITTING A LINEUP
@app.callback(Output('team_stats', 'data'), Output('team-wins-prediction', 'children'),
              Input('sub-team', 'n_clicks'), State('psel_tbl', 'data'),
              State('sel_tbl', 'data'), State('games_rem', 'data'),
              State('inn_rem', 'data')
)
def submit_team(submit, pit_sel_tbl, hit_sel_tbl, gs, inn):
    if submit is None:
        raise PreventUpdate
    reg_stats = {}
    wins = ''
    if submit and (gs == 0) and (inn == 0):
        # convert hitters in hitter objects
        hitters = np.array([])
        for h in hit_sel_tbl['data-frame']:
            # Parsing seasons string from hit_sel_tbl
            seasons = h['Years']
            first = int(seasons[:4])
            second = int(seasons[-4:])
            ##########
            # checking to see that first is less than second
            if (first <= second):
                yr_range = list(range(first, second + 1, 1))
            else:
                yr_range = list(range(second, first + 1, 1))
            player = hitter(h['Name'], yr_range, h['Games'])
            hitters = np.append(hitters, player)
        pitchers = np.array([])
        for p in pit_sel_tbl['data-frame']:
            # Parsing seasons string from hit_sel_tbl
            seasons = p['Years']
            first = int(seasons[:4])
            second = int(seasons[-4:])
            # checking to see that first is less than second
            if (first <= second):
                yr_range = list(range(first, second + 1, 1))
            else:
                yr_range = list(range(second, first + 1, 1))
            player = pitcher(p['Name'], yr_range, p['Innings'])
            pitchers = np.append(pitchers, player)

        # hitters and pitchers should be full and contain enough info to make predictions
        # make predictions
        wins, reg_stats = wins_for_team(hitters, pitchers)
        reg_stats['Team'] = 'my_team'
        reg_stats['Season'] = 2022
        reg_stats['W'] = wins
        reg_stats['WAR_y'] = reg_stats.pop('WAR') * 162
        reg_stats['Def'] = reg_stats['Def'] * 162
        reg_stats['BsR'] = reg_stats['BsR'] * 162
    return reg_stats, f'Wins: {wins}'

if __name__ == '__main__':
    app.run_server(debug=True)


# In[49]:


if __name__ == '__main__':
    app.run_server(debug=True)


# Barriers to progress: No traceback errors, no way to find value of local variables

# In[ ]:


px.scatter(team_history, x = 'wRC+', y = 'W', hover_data=('Team', 'Season', 'wRC+', 'HR/9', 'BsR', 'WAR_y', 'Def', 'SLG'))


# 1. Add comments explaining decision making process///
# 2. Update code to read from SQL
# 3. Make app usable for others
# 4. Write out article

# # Messing Around

# In[ ]:


get_ipython().system(' pip install pyngrok')


# In[ ]:


get_ipython().system(" ngrok authtoken '2A8APmoPyytFrFIVVtjLj1yjTgZ_6NMzSg8LeyUQvVNKxBXRr'")


# In[ ]:


from pyngrok import ngrok
public_url = ngrok.connect(port = '8050')#check port
ssh_url = ngrok.connect(22, 'tcp')

