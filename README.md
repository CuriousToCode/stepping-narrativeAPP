# stepping-narrativeAPP

This project is built using the Fitbit API and the git repository of orcasgit/python-fitbit -- https://github.com/orcasgit/python-fitbit.

Purpose:
To create a data pipline (hosted on a local server) for which behavorial insights related to physical (step) activity to improve physical activity and QoL for myself and others. 

Stepping Narrative:
- speaks to the story of one indivudals stepping 'patterns' [volume x timing] (steps/minute) over the lifespan of wear-time (hrs, days, months, yrs, etc.)
- Total step volume is useful, knowing when in the day you were able to manufacture such steps adds another layer of insights that may very well increase the liklihood of meeting Canada's reccomended physical activity guidlines. 
- Personalized reports sent in a timely manner and in a localized fashion (i.e. email account)

Stage: 

cron scheduler {

(1) Pull = direct communicaiton with Fitbit API

(2) Store = client data is housed within DBSM - postgres sql on a local drive

(3) Email = daily emails sent containing personalized supportive texts along with 4 updated subplots describing a stepping narrative attached as a single .jpeg file. 
                } 

Timeline and near-term goals:
- The project has been active for approximatly 4 months...
- Complete functionality (with 'single-user') for fall 2019
- pilot with friends, family then design RCT style intervention through YorkU REB
- Additonal help is welcomed! Ideally I'd like to take the concept and move it from a local server to online platform. 
