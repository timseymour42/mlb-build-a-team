# MLB-Build-a-Team

# Data and Analysis Project: MLB Build a Team
- Tim Seymour (seymour.ti@northeastern.edu)

## Project Goal:
The goal is to be able to predict how many wins any combination of players would have over a 162 game season. As a baseball fanatic, I myself have always wanted a tool to see how much a team can expect to improve from a free agency move or a trade, so I decided to create something to do just that. My hope for this project is that baseball fans like me will be able to determine what a player might do for their team given their statistics from the most recent season.

<a id='data'></a>

## Data
I will obtain statcast statistics such as:

wrc+, BsR, DRS/162, FIP, K-BB%

# Hypotheses

The statistics that will be most useful in predicting MLB win totals will be advanced statistics
 - In many cases these control for ballpark, strength of opponent, and other factors that eliminate luck

The model will have an R squared value greater than .6
 - Baseball statistics may be better evaluations of individual player performance than stats in any other sport because there are fewer players connected to each play (an At-Bat in this case).
 - A priority of mine in developing the win prediction model is making sure only to include statistics that can be aggregated given a set of players from different teams.

Baserunning will not have much weight in the Linear Regression Model
 - Although baserunning is an aspect of the game that can give one team an advantage over another, it is something that is often sacrificed in roster construction in the interest of prioritizing hitting. 
 - Teams like the Rays with top-of-the-line analytics departments dominate baserunning, however, so in the coming years I expect this to shift

# Conclusion/Revisiting Hypotheses
The statistics that will be most useful in predicting MLB win totals will be advanced statistics:
 - Though slightly better representative than counting stats, there was not as extreme of a difference as I had expected. Recursive Feature Elimination even selected homeruns over WRC+ which account for the value of a homerun

The model will have an R squared value greater than .6
 - The R squared value was about .73 which I found to be outstanding results. Like no other sport, baseball has easily aggregated advanced statistics that are actually representative of their performance.

Baserunning will not have much weight in the Linear Regression Model
 - Baserunning had limited correlation with win percentage as was expected

# Limitations/Areas in need of improvement
- This model does not take into account the fact that you need more than 9 hitters and 10 pitchers - depth is an incredibly important part of building a winning team. One potential improvement to fix this limitation would be to build a function that allows the user to input how many games each player will play and adjust the aggregation function accordingly.
