import streamlit as st
import pickle  
import pandas as pd

st.title('IPL win predictor')

teams=['Rajasthan Royals', 'Royal Challengers Bangalore',
'Sunrisers Hyderabad', 'Delhi Capitals', 'Chennai Super Kings',
'Gujarat Titans', 'Lucknow Super Giants', 'Kolkata Knight Riders',
'Punjab Kings', 'Mumbai Indians',]

cities=['Hyderabad', 'Delhi', 'Chennai', 'Ahmedabad', 'Jaipur', 'Kolkata',
        'Mumbai', 'Dubai', 'Bengaluru', 'Kimberley', 'Abu Dhabi', 'Pune',
        'Johannesburg', 'Port Elizabeth', 'Navi Mumbai', 'Visakhapatnam',
        'Ranchi', 'Durban', 'Sharjah', 'Cape Town', 'Centurion', 'Nagpur',
        'Cuttack', 'Raipur', 'East London', 'Bloemfontein']

pipe=pickle.load(open("D:\Programs\Python\ML_notebooks\IPL_prediction\pipe.pkl",'rb'))

col1, col2= st.columns(2)

with col1:
    batting_team=st.selectbox('Select the batting team',sorted(teams))

with col2:
    bowling_team=st.selectbox('Select the bowling team',sorted(teams))

selected_city=st.selectbox(('Select Venue city'),sorted(cities))

target=st.number_input('Target')

col3, col4= st.columns(2)

with col3:
    cur_score=st.number_input("Current Score")

with col4:
    wickets=st.number_input('Wickets out')

col5, col6= st.columns(2)

with col5:
    overs=st.number_input('Overs Completed')
with col6:
    balls=st.number_input('Balls completed in current over')

if st.button('Predict Probability'):
    runs_left = target-cur_score
    balls_left=120-(overs*6+balls)
    wickets= 10-wickets
    crr=cur_score*6/(120-balls_left)
    rrr=runs_left*6/balls_left

    input_df=pd.DataFrame({'BattingTeam':[batting_team],'BowlingTeam':[bowling_team],'City':[selected_city],'Runs_left':[runs_left],
                        'Balls_left':[balls_left],'Wickets_left':[wickets],'total_run_x':[target],'CRR':[crr],'RRR':[rrr]})

    result=pipe.predict_proba(input_df)
    
    loss=result[0][0]
    win=result[0][1]
    st.header(batting_team + "-" + str(round(win*100))+"%")
    st.header(bowling_team + "-" + str(round(loss*100))+"%")

