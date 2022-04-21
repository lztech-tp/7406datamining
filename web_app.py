"""The main file to predict soccer world cup predictions

The code consists of 3 main parts:
1. Assigning each team data from the csv files to eact team name, and build a dataframe similar to the one the model is trained on.

2. Loading the model and the Preprocess pipeline to process the data including (one hot encoding for categorical variables and Making
PCA for the numerical values for dimensionality reduction

3.Making a web app front end frame work using Streamlit library to deploy the model and put it into production
"""

#Importing the libraries
import pandas as pd
import numpy as np
from numpy import loadtxt
import pickle
import matplotlib.pyplot as plt
import streamlit as st
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, precision_score , accuracy_score ,f1_score


class DataFrameSelector(BaseEstimator,TransformerMixin):
    def __init__(self,attribute_names):
        self.attribute_names = attribute_names
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        return X[self.attribute_names]



#loading in the model and the pipeline files to predict on the data for the second model
pickle_in4 = open('model2_xgb.pkl', 'rb')
model2 = pickle.load(pickle_in4)
classes2 = model2.classes_

pickle_in5 = open('pipeline2.pkl', 'rb')
pipeline2 = pickle.load(pickle_in5)

pickle_in6 = open('model_goals_xgb.pkl', 'rb')
model3 = pickle.load(pickle_in6)
classes3 = model3.classes_

xgb_preds = loadtxt('xgb_preds.csv', delimiter=',')
ytest = loadtxt('xgb_ytest.csv', delimiter=',')


#create choose list for second model including the teams names from the trained data
team1_list2 = ['Algeria', 'Argentina', 'Australia', 'Belgium', 'Brazil', 'Cameroon',
       'Canada','Chile', 'Colombia', 'Costa Rica', 'Croatia', 'Denmark', 'Ecuador',
       'Egypt', 'England', 'France', 'Germany', 'Ghana', 'Greece', 'Honduras',
       'Iceland', 'Iran', 'Italy', 'Japan','Mexico', 'Morocco', 'Netherlands',
       'New Zealand', 'Nigeria', 'Panama', 'Paraguay', 'Peru', 'Poland',
       'Portugal', 'Qatar','Russia', 'Saudi Arabia', 'Scotland','Senegal', 'Serbia', 'Slovakia',
       'Slovenia', 'South Africa', 'South Korea','Spain', 'Sweden', 'Switzerland', 'Tunisia',
       'Uruguay', 'USA', 'Ukraine', 'United Arab Emirates','Wales']

team2_list2 = team1_list2.copy()


#read the meta data for both home and away teams to assign the data
#based on the choosen team for the second model
df_home = pd.read_csv('df_home_all2.csv',index_col=0)
df_away = pd.read_csv('df_away_all2.csv',index_col=0)
                
def welcome():
	return 'welcome all'

# this is the main function in which we define our webpage
def main():
	# giving the webpage a title
	
	# here we define some of the front end elements of the web page like
	# the font and background color, the padding and the text to be displayed
	html_temp = """
	<div style ="background-color:yellow;padding:13px">
	<h2 style ="color:black;text-align:center;">World Cup match Prediction App </h2>
	</div>
	"""

	choices = ['Match Result Prediction','Model Performance']
	ticker = st.sidebar.selectbox('Choose a Page',choices)
	st.markdown(html_temp, unsafe_allow_html = True)

	if (ticker=='Match Result Prediction'):
            # this line allows us to display a drop list to choose team 1 and team 2 
            st.header('Match Prediction Page')
            team_3 = st.selectbox('Team 1', np.array(team1_list2))
            team_4 = st.selectbox('Team 2', np.array(team2_list2))


            # the below line ensures that when the button called 'Predict' is clicked,
            # the prediction function defined above is called to make the prediction
            # and store it in the variable result
            results_df2 = pd.DataFrame()

            # CSS to inject contained in a string
            hide_table_row_index = """
            <style>
            tbody th {display:none}
            .blank {display:none}
            </style>
            """
            
            if st.button("Predict "):
                if (team_3 == team_4):
                    st.text('Please select different teams')
                else:
                    
                    results_df2 = predict_match_result2(team_3 , team_4)

                    # Inject CSS with Markdown
                    st.markdown(hide_table_row_index, unsafe_allow_html=True)

                    #st.dataframe(results_df2)
                    st.table(results_df2.style.format("{:.3f}").hide_index())
                    #this step to preduict the match final result and display the highest results propabilities
                    draw_df , home_w_df , away_w_df = predict_match_result_goals(team_3 , team_4)

                    
                    st.subheader('Match Result prediction')

                    
                    #add three dataframes of the match results in case of Draw, Win , Lose
                    col1, col2 ,col3 = st.columns(3)
                    col1.markdown("Draw Results")
                    col1.markdown(hide_table_row_index, unsafe_allow_html=True)
                    col1.table(((pd.DataFrame(draw_df.loc[0].nlargest(3)).T)*(1/(draw_df.loc[0].nlargest(3).values.sum()))).style.format("{:.3f}"))
                    col2.markdown("Team 1 win Results")
                    col2.markdown(hide_table_row_index, unsafe_allow_html=True)
                    col2.table(((pd.DataFrame(home_w_df.loc[0].nlargest(3)).T)*(1/(home_w_df.loc[0].nlargest(3).values.sum()))).style.format("{:.3f}"))
                    col3.markdown("Team 2 win Results")
                    col3.markdown(hide_table_row_index, unsafe_allow_html=True)
                    col3.table(((pd.DataFrame(away_w_df.loc[0].nlargest(3)).T)*(1/(away_w_df.loc[0].nlargest(3).values.sum()))).style.format("{:.3f}"))


	else:
		st.header('Match Winner Prediction Model Performance')
		st.subheader('Performance Metrics')
		score = 'The accuracy score :' + str(np.round(accuracy_score(ytest, xgb_preds),3))
		st.text(score)

		
		score2 = 'The precision score :' + str(np.round(precision_score(ytest, xgb_preds,average='weighted'),3))
		st.text(score2)

		score3 = 'The recall score :' + str(np.round(recall_score(ytest, xgb_preds,average='weighted'),3))
		st.text(score3)

		st.subheader('Confusion Matrix')
		st.pyplot(plot_confusion_matrix(ytest,xgb_preds))

		

#functions for model2 data assignment
#Assign values from the dataframe to the team name and retuen a dataframe with all team1 data
def assign_values_to_team3(team):
    
    if team in df_home.index :
        team1_data =  df_home.loc[team].reset_index()
        team1_data = team1_data.groupby('index').mean().reset_index().rename(columns={'index':'home_team.name'}).iloc[0]
        return team1_data

#Assign values from the dataframe to the team name and retuen a dataframe with all team2 data
def assign_values_to_team4(team):
    
    if team in df_away.index :
        team2_data =  df_away.loc[team].reset_index()
        team2_data = team2_data.groupby('index').mean().reset_index().rename(columns={'index':'away_team.name'}).iloc[0]
        return team2_data

#run the assign values functions and concat the resultiung 2 dataframes into one dataframe for the model input
def map_inputs_to_data2(team1,team2):

    team_3z = assign_values_to_team3(team1)
               
    team_4z = assign_values_to_team4(team2)

    input_data = pd.concat([team_3z,team_4z])
    return input_data

#get the input data and preprocess the data using the loaded data processing Pipeline,
#and predict the match result probabilities using predict_proba function, and return a dataframe with the probabilites.
def predict_match_result2(team3 ,team4):

    input_d = map_inputs_to_data2(team3 , team4)
    input_processed = pipeline2.transform(pd.DataFrame(input_d).T)
    preds_test = model2.predict_proba(input_processed)
      
    results_df = pd.DataFrame(columns=classes2,data=np.round(preds_test,3))
    results_df.rename(columns={0:'Draw Probability',1:'{} wins Probability'.format(team3),2:'{} wins Probability'.format(team4)},inplace=True)
    return results_df

#Predict function for the final match result prediction
def predict_match_result_goals(team3 ,team4):

    input_d = map_inputs_to_data2(team3 , team4)
    input_processed = pipeline2.transform(pd.DataFrame(input_d).T)
    preds_test = model3.predict_proba(input_processed)
      
    results_df = pd.DataFrame(columns=classes3,data=np.round(preds_test,4))
    draw_df = results_df[[x for x in results_df.columns if (int(x[0]) == int(x[2]))]]
    home_w_df = results_df[[x for x in results_df.columns if (int(x[0]) > int(x[2]))]]
    away_w_df = results_df[[x for x in results_df.columns if (int(x[0]) < int(x[2]))]]

    return draw_df,home_w_df,away_w_df

#function to display confusion matrix
def plot_confusion_matrix(y_test,preds):
    fig, ax = plt.subplots(figsize=(6, 6))
    conf_matrix = confusion_matrix(y_test,preds)
    
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='large')
     
    plt.xlabel('Predictions', fontsize=15)
    plt.ylabel('Actuals', fontsize=15)
    ticks = ['Draw','Team1 Win','Team2 Win']
    labels= [0,1,2]
    plt.xticks(labels,ticks)
    plt.yticks(labels,ticks)
    plt.title('Confusion Matrix', fontsize=16)
    
    return fig
    
    
	
if __name__=='__main__':
	main()

