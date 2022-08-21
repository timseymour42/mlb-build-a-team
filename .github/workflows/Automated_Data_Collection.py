import datetime
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
from sqlalchemy import create_engine
import MySQLdb


# In[5]:


#Changing column names of python df to be able to insert into SQL smoothly
sql_col_mapping = {'BB%': 'BB_pct', 'K%': 'K_pct', 'wRC+': 'wRC_plus', 'K/9': 'K_per_9',
       'BB/9': 'BB_per_9', 'HR/9': 'HR_per_9', 'LOB%': 'LOB_pct', 'GB%': 'GB_pct', 'HR/FB': 'HR_per_FB', 'vFA (pi)': 'vFA'}


# In[6]:


#To change column names of SQL Table for agreement with Python
python_col_mapping = {v: k for k, v in sql_col_mapping.items()}


# # Creating CSV with all games from 2015+

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


# Scraping process takes ~20 minutes; CSV stored for convenience

# In[3]:


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


# In[4]:


# CODE USED FOR INITIAL SCRAPING

# hit, pit = collect_team_data()


# In[4]:


def collect_new_team_data(df):
    '''
    Scrapes FanGraphs data from each day the most recent record scraped and today's date

    Returns:
        hit (pd.DataFrame) contains hitting stats with each record representing one game for a team
        pit (pd.DataFrame) contains pitching stats with each record representing one game for a team
    '''
    recent_record = datetime.datetime.strptime(df['Date'].max(), '%Y-%m-%d')
    # beginning of sample is the most recent day data was collected
    date = recent_record + datetime.timedelta(days = 1)
    # collects team hitting stats for each day
    hit = pd.DataFrame()
    # collects team pitching stats for each day
    pit = pd.DataFrame()
    # sustainable way of changing year without change in code
    while (date < datetime.datetime.now()):
        date_str = date_to_str(date)
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
        date += datetime.timedelta(days = 1)
    return hit, pit


# In[6]:


#CODE USED FOR INITIAL SCRAPING

# pit.drop(columns = ['G'], inplace = True)
# # Joining hitting and pitching dataframes on team and date
# all_stats = pd.merge(hit, pit, left_on = ['Team', 'Date'], right_on = ['Team', 'Date'], how = 'inner')
# # Excludes data from days where team played a double header
# all_stats = all_stats[all_stats.GS == '1']
# all_stats.to_csv('daily_game_stats.csv') 
# files.download('daily_game_stats.csv')


# In[100]:


# import MySQLdb
# db = MySQLdb.connect("localhost", 'root', 'P@ssw0rd', 'mlb_db')
# tblchk = db.cursor()

# #CREATING TABLE
# tblchk.execute('Drop table if exists game_data')
# sql_query = '''create table game_data(gm_id int auto_increment primary key, Team varchar(255), G varchar(255), PA varchar(255), HR varchar(255), R varchar(255), RBI varchar(255), SB varchar(255),
#  BB_pct varchar(255), K_pct varchar(255), ISO varchar(255), BABIP_x varchar(255), AVG varchar(255), OBP varchar(255), SLG varchar(255), wOBA varchar(255), xwOBA varchar(255), wRC_plus varchar(255), 
#  BsR varchar(255), Off varchar(255), Def varchar(255), WAR_x varchar(255), Date varchar(255), W varchar(255), L varchar(255), SV varchar(255), GS varchar(255), 
#  IP varchar(255), K_per_9 varchar(255), BB_per_9 varchar(255), HR_per_9 varchar(255), BABIP_y varchar(255), LOB_pct varchar(255), GB_pct varchar(255), HR_per_FB varchar(255), vFA varchar(255), 
#  ERA varchar(255), xERA varchar(255), FIP varchar(255), xFIP varchar(255), WAR_y varchar(255))'''
# tblchk.execute(sql_query)


# Method of adding to SQL game_data Table

# In[14]:


# create sqlalchemy engine
# engine = create_engine("mysql+pymysql://root:P@ssw0rd@localhost/mlb_db"
#                        .format(user="root",
#                                pw="P@ssw0rd",
#                                db="mlb_db"))


# In[9]:


# CODE FOR ADDING INITIAL DATA PREVIOUSLY SCRAPED

# new_stats = pd.read_csv('https://github.com/timseymour42/MLB-Build-a-Team/blob/a3774339cb04887ba2026cab07c2923b27422b60/daily_stats%20(2).csv?raw=true', header = 0, index_col = 0)
# new_stats.rename(columns = sql_col_mapping, inplace=True)
# new_stats.fillna('NA', inplace = True)
# #Dropping #_x, #_y
# new_stats.drop(columns = ['#_x', '#_y'], inplace = True)
# # Insert whole DataFrame into MySQL
# new_stats.to_sql('game_data', con = engine, if_exists = 'append', chunksize = 1000, index = False)


# In[7]:


def update_game_data(sql_col_mapping):
    db = MySQLdb.connect(host='127.0.0.1', user='root', passwd='', db='mlb_db', port=8888)
    # Need to load in CSV identify the most recent date, scrape from most recent date to today, append
    game_data = pd.read_sql('SELECT * FROM game_data', con = db)
    hit, pit = collect_new_team_data(game_data)
    if len(pit) > 0:
        pit.drop(columns = ['G'], inplace = True)
        # Joining hitting and pitching dataframes on team and date
        new_stats = pd.merge(hit, pit, left_on = ['Team', 'Date'], right_on = ['Team', 'Date'], how = 'inner')
        new_stats.rename(columns = sql_col_mapping, inplace=True)
        new_stats.fillna('NA', inplace = True)
        #Dropping #_x, #_y
        new_stats.drop(columns = ['#_x', '#_y'], inplace = True)
        # Insert whole DataFrame into MySQL
        new_stats.to_sql('game_data', con = engine, if_exists = 'append', chunksize = 1000, index = False)


# In[13]:


#update_game_data(sql_col_mapping)


# Creating a string with SQL table columns

# # Scraping Player Data

# In[8]:


def scrape_player_data():
    # beginning of sample is 1900
    year = 1900
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
        year+=1
    return wrc, pitch


# In[9]:


def string_to_num(string):
    if(type(string) == str):
        if('%' in string):
            string = string.replace('%', '')
    return float(string)


# In[10]:


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
    print(pitch_df.columns)
    print(hit_df.columns)
    return hit_df, pitch_df


# In[22]:


def add_new_player_data(hit_df, pit_df):
    # Setting up current CSV data to be appended to
    year = int(hit_df['Season'].max())
    # Excluding current year for freshly scraped aggregates
    wrc = pd.DataFrame()
    pitch = pd.DataFrame()

    while year <= datetime.datetime.now().year:
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
        year+=1
    hit_df, pitch_df = clean_player_data(wrc, pitch)
    return hit_df, pitch_df


# ### Loading player data from CSVs into SQL Tables

# In[12]:


# import MySQLdb
# db = MySQLdb.connect("localhost", 'root', 'P@ssw0rd', 'mlb_db')
# tblchk = db.cursor()

# #CREATING HITTER TABLE
# tblchk.execute('Drop table if exists hitter_data')
# sql_query = '''create table hitter_data(hitter_id int auto_increment primary key, Name varchar(255), Season varchar(255),
# Team varchar(255), G varchar(255), PA varchar(255), HR varchar(255), R varchar(255), RBI varchar(255), 
# SB varchar(255), BB_pct varchar(255), K_pct varchar(255), ISO varchar(255), BABIP varchar(255), AVG varchar(255),
# OBP varchar(255), SLG varchar(255), wOBA varchar(255), xwOBA varchar(255), wRC_plus varchar(255), BsR varchar(255), 
# Off varchar(255), Def varchar(255), WAR varchar(255), AB varchar(255), TB varchar(255))'''
# tblchk.execute(sql_query)

# #CREATING PITCHER TABLE
# tblchk.execute('Drop table if exists pitcher_data')
# sql_query = '''create table pitcher_data(pitcher_id int auto_increment primary key, Name varchar(255),
# Season varchar(255), Team varchar(255), W varchar(255), L varchar(255), SV varchar(255), 
# G varchar(255), GS varchar(255), IP varchar(255), K_per_9 varchar(255), BB_per_9 varchar(255), HR_per_9 varchar(255), 
# BABIP varchar(255), LOB_pct varchar(255), GB_pct varchar(255), HR_per_FB varchar(255), vFA varchar(255), 
# ERA varchar(255), xERA varchar(255), FIP varchar(255), xFIP varchar(255), WAR varchar(255), HR varchar(255))'''
# tblchk.execute(sql_query)


# In[15]:


# #CODE FOR ADDING INITIAL DATA PREVIOUSLY SCRAPED
# engine = create_engine("mysql+pymysql://root:P@ssw0rd@localhost/mlb_db"
#                        .format(user="root",
#                                pw="P@ssw0rd",
#                                db="mlb_db"))
# new_hit_df = pd.read_csv('https://github.com/timseymour42/MLB-Build-a-Team/blob/5548a60b92575ee19b159c791934630cbd9f72d3/hitters_yearly.csv?raw=true', header = 0)
# new_hit_df.rename(columns = sql_col_mapping, inplace=True)
# new_hit_df.fillna('NA', inplace = True)
# #Dropping #
# new_hit_df.drop(columns = ['#'], inplace = True)
# # Insert whole DataFrame into MySQL
# new_hit_df.to_sql('hitter_data', con = engine, if_exists = 'append', chunksize = 1000, index = False)


# In[16]:


# #CODE FOR ADDING INITIAL DATA PREVIOUSLY SCRAPED
# engine = create_engine("mysql+pymysql://root:P@ssw0rd@localhost/mlb_db"
#                        .format(user="root",
#                                pw="P@ssw0rd",
#                                db="mlb_db"))
# new_pit_df = pd.read_csv('https://github.com/timseymour42/MLB-Build-a-Team/blob/5548a60b92575ee19b159c791934630cbd9f72d3/pitchers_yearly.csv?raw=true', header = 0)
# new_pit_df.rename(columns = sql_col_mapping, inplace=True)
# new_pit_df.fillna('NA', inplace = True)
# #Dropping #
# new_pit_df.drop(columns = ['#'], inplace = True)
# # Insert whole DataFrame into MySQL
# new_pit_df.to_sql('pitcher_data', con = engine, if_exists = 'append', chunksize = 1000, index = False)


# In[17]:


def update_players_data(sql_col_mapping):
    db = MySQLdb.connect(host='127.0.0.1', user='root', passwd='', db='mlb_db', port=8888)
    tblchk = db.cursor()
    hit_df_ = pd.read_sql('SELECT * FROM hitter_data', con = db)
    pit_df_ = pd.read_sql('SELECT * FROM pitcher_data', con = db)
    hit, pit = add_new_player_data(hit_df_, pit_df_)
    if len(pit) > 0:
        pit.drop(columns = ['G'], inplace = True)
        # Joining hitting and pitching dataframes on team and date
        hit.rename(columns = sql_col_mapping, inplace=True)
        pit.rename(columns = sql_col_mapping, inplace=True)
        hit.fillna('NA', inplace = True)
        pit.fillna('NA', inplace = True)
        # resetting the indices
        hit.reset_index(inplace = True)
        pit.reset_index(inplace = True)
        # Dropping #_x, #_y
        hit.drop(columns = ['#'], inplace = True)
        pit.drop(columns = ['#'], inplace = True)
        # deleting current year records
        max_year = hit['Season'].max()
        sql_query = f'''DELETE FROM hitter_data hd 
                        WHERE hd.Season >= {max_year};
                        COMMIT;'''
        tblchk.execute(sql_query)
        sql_query = f'''DELETE FROM pitcher_data pd 
                        WHERE pd.Season >= {max_year};
                        COMMIT;'''
        tblchk.execute(sql_query)
        engine = create_engine("mysql+pymysql://root:P@ssw0rd@localhost/mlb_db"
                       .format(user="root",
                               pw="P@ssw0rd",
                               db="mlb_db"))
        # adding data from newest year
        hit.to_sql('hitter_data', con = engine, if_exists = 'append', chunksize = 1000, index = False)
        pit.to_sql('pitcher_data', con = engine, if_exists = 'append', chunksize = 1000, index = False)


# # Collecting team data to compare model predictions to actual full season win totals
# 
# - key question is what model or combination of models minimizes error in predicting team success historically

# In[35]:


# import MySQLdb
# db = MySQLdb.connect("localhost", 'root', 'P@ssw0rd', 'mlb_db')
# tblchk = db.cursor()

# #CREATING TEAM DATA TABLE
# tblchk.execute('Drop table if exists team_data')
# sql_query = '''create table team_data(team_id int auto_increment primary key, Team varchar(255), G_x varchar(255),
# PA varchar(255), HR varchar(255), R varchar(255), RBI varchar(255), SB varchar(255), BB_pct varchar(255), 
# K_pct varchar(255), ISO varchar(255), BABIP_x varchar(255), AVG varchar(255), OBP varchar(255), SLG varchar(255),
# wOBA varchar(255), xwOBA varchar(255), wRC_plus varchar(255), BsR varchar(255), Off varchar(255), Def varchar(255),
# WAR_x varchar(255), Season varchar(255), W varchar(255), L varchar(255), SV varchar(255),
# G_y varchar(255), GS varchar(255), IP varchar(255), K_per_9 varchar(255), BB_per_9 varchar(255), HR_per_9 varchar(255),
# BABIP_y varchar(255), LOB_pct varchar(255), GB_pct varchar(255), HR_per_FB varchar(255), vFA varchar(255),
# ERA varchar(255), xERA varchar(255), FIP varchar(255), xFIP varchar(255), WAR_y varchar(255))'''
# tblchk.execute(sql_query)


# In[70]:


#CODE FOR ADDING INITIAL DATA PREVIOUSLY SCRAPED

# team_data = pd.read_csv('https://github.com/timseymour42/MLB-Build-a-Team/blob/8d552de89f4daf8a9aa27edde95179f3bb192258/team_yearly_data.csv?raw=true', header = 0)
# team_data.rename(columns = sql_col_mapping, inplace=True)
# team_data.fillna('NA', inplace = True)
# #Dropping unnecessary columns
# print(team_data.columns)
# team_data.drop(columns = ['#_x', '#_y', 'Unnamed: 0'], inplace = True)
# # Insert whole DataFrame into MySQL
# team_data.to_sql('team_data', con = engine, if_exists = 'append', chunksize = 1000, index = False)


# In[61]:


def collect_team_data_yearly(year):

    '''
    Args:
    year (integer): year to start collecting data from
    Collecting team data to use as testing data
    '''
    year = int(year)
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


# In[ ]:


# CODE USED FOR INITIAL SCRAPING

# team_data.to_csv('team_yearly_data.csv')
# files.download('team_yearly_data.csv')


# In[68]:

def update_team_data(sql_col_mapping):
    db = MySQLdb.connect(host='127.0.0.1', user='root', passwd='', db='mlb_db', port=8888)
    tblchk = db.cursor()
    # The year of the latest record in the data table
    sql_team_data = pd.read_sql('SELECT * FROM team_data', con = db)
    max_year = sql_team_data['Season'].max()
    sql_query = f'''DELETE FROM team_data td 
                    WHERE td.Season >= {max_year};
                    
                    COMMIT;'''
    tblchk.execute(sql_query)
    
    #collecting team data from most recent year to the present
    h, p = collect_team_data_yearly(max_year)
    team_data = pd.merge(h, p, left_on = ['Season', 'Team'], right_on = ['Season', 'Team'], how = 'outer')
    for col in team_data.columns:
        if col not in ['Team', 'Season', 'GB']:
            team_data[col] = team_data[col].apply(string_to_num)
    team_data['W'] = team_data['W'] * (162 / team_data['GS'])
    
    team_data.rename(columns = sql_col_mapping, inplace=True)
    team_data.fillna('NA', inplace = True)
    #Dropping unnecessary columns
    team_data.drop(columns = ['#_x', '#_y'], inplace = True)
    #Creating engine to append dataframe to database
    engine = create_engine("mysql+pymysql://root:P@ssw0rd@localhost/mlb_db"
                       .format(user="root",
                               pw="P@ssw0rd",
                               db="mlb_db"))
    # adding data from newest year
    team_data.to_sql('team_data', con = engine, if_exists = 'append', chunksize = 1000, index = False)


# In[71]:


def main():
    update_team_data(sql_col_mapping)
    update_game_data(sql_col_mapping)
    update_players_data(sql_col_mapping)


# In[ ]:


if __name__ == '__main__':
    main()

