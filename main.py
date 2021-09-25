import streamlit as st
import evalml 
from evalml import AutoMLSearch
from evalml.demos.churn import load_churn
from evalml.objectives import CostBenefitMatrix
from evalml import AutoMLSearch
import pandas as pd
import lux
from evalml.model_understanding.graphs import graph_confusion_matrix
from evalml.model_understanding.graphs import graph_roc_curve
import woodwork as ww

# Set up the sidebar to display#
#----------------------------------------------------------------------------------------------------#
x,y_ = load_churn()

this_side_bar = st.sidebar.title('Values for the Cost-Benefit Matrix')
this_side_bar.subheader('Select values to optimize the model')

true_positive = st.sidebar.slider('True positive value',0,500,400)
true_negative = st.sidebar.slider('True negative value',-100,100,0)
false_positive = st.sidebar.slider('False positive value',-500,0,-100)
false_negative = st.sidebar.slider('False negative value',-500,0,-200)

#Define ML models #
    
#-------------------------------------------------------------------------------------------------#
x.ww.set_types({'PaymentMethod':'Categorical'},{'Contract':'Categorical'})
cost_benefit_matrix = CostBenefitMatrix(true_positive = true_positive,
                                        true_negative = true_negative,
                                        false_positive = false_positive,
                                        false_negative = false_negative)

#@st.cache(hash_funcs={builtins.dict: my_hash_func})
def model_builder( data_train, y_train):
    
    automl_churn_model = AutoMLSearch(data_train,y_train,problem_type='binary',objective=cost_benefit_matrix)

    return automl_churn_model
     
x_train, x_test, y_train, y_test = evalml.preprocessing.split_data(x,y_,test_size= 0.2,problem_type='binary',random_seed = 0) 


automl_churn = model_builder(x_train,y_train )
    
show_model =automl_churn.search()
    
describe_pipe = automl_churn.describe_pipeline(automl_churn.rankings.iloc[0]['id'], return_dict = True)
    

    
#Define containers to show data #

#--------------------------------------------------------------------#
header = st.container()

dataset = st.container()

model = st.container()

plots = st.container()

more_plots = st.container()

#-----------------------------------------------------------------------------------------------#

with header:
    st.title('Classification: Churn Optimization')
    """
    ### This is an example of implementing a custom function as a metric for a Machine Learning classification problem, in this case, the function is the confussion matrix, in particular the Cost Benefit Matrix case.
    
    ### Correctly classifying a costumer will result in a net profit, because it allows us to intervene, incentivize the       customer to stay, and sign a new contract. 

    ### Incorrectly classifying customers who were not going to churn as customers who will churn (false positive case) will    generate a cost to represent the marketing and effort used to try to retain the user. 

    ### Not identifying customers who will churn (false negative case) will cost us some amount, to represent the lost in        revenue from losing a customer. 


    ### Finally, correctly identifying customers who will not churn (true negative case) will not cost us anything.


    ### Using the CostBenefitMatrix as an objective instead of a metric like log loss improve the total profit.
    
    """

#---------------------------------------------------------------------------------------------------#  
with dataset:
    st.header('Dataset')
    
    showdata = pd.DataFrame(x)
    showdata['Type'] = y_
    showdata
    
    st.caption('Source: [Telco Customer dataset](https://www.kaggle.com/blastchar/telco-customer-churn?select=WA_Fn-UseC_-Telco-Customer-Churn.csv)')
#---------------------------------------------------------------------------------------------------#    

with model:
    st.header('Models fitted with corresponding preprocessing transforms')
    
    st.write(automl_churn.rankings)
    
    
#----------------------------------------------------------------------------------------------------#    
    
with plots:
    st.header('The pipelines with better performance to optimize based on the Cost-Benefit Matrix')
    
    to_sort_ = pd.DataFrame(automl_churn.rankings)
    to_sort = to_sort_['id'].astype(str).str[0]
    to_sort = to_sort.astype(int)
    to_sort = to_sort.iloc[0:5]
    
    trained_pipes = automl_churn.train_pipelines([automl_churn.get_pipeline(i) for i in to_sort.values])
    trained_pipes
    
    st.write('Scoring the pipelines')
    
    pipe_score = automl_churn.score_pipelines([trained_pipes[name] for name in trained_pipes.keys()],
                                             x_test, y_test, [cost_benefit_matrix])
    
    scored_pipes = pd.DataFrame(pipe_score)
    scored_pipes
    
    
    best_ =automl_churn.best_pipeline
    profit_ = best_.score( x_test, y_test,[cost_benefit_matrix])
    profit = profit_['Cost Benefit Matrix']*len(x)
    
    
    st.write('The total profit is:', profit)
    
    y_predict = best_.predict(x_test)
    
    plotly_plot_mat = graph_confusion_matrix(y_test,y_predict)

    #st.write()
    st.write('More information and cross validation data for all objectives scores')
    
    #pipe_desc = best_.graph()
    
    #metrics = pd.DataFrame(profit_)
    st.write(describe_pipe)
    
    #st.plotly_chart(pipe_desc, use_container_width = True)
        
#-----------------------------------------------------------------------------------------------------#    
    
with more_plots:
    
    y_pred_proba = best_.predict_proba(x_test)
    
    y_encoded = y_test.ww.map({'No': 0, 'Yes': 1})
    
    plotly_plot_f = best_.graph_feature_importance()
    
    plotly_plot_roc = graph_roc_curve(y_encoded, y_pred_proba)
    
    
    st.write('Plots for the best model: (Feature importance, ROC curve, Confusion Matrix)')
    
    st.plotly_chart(plotly_plot_f, use_container_width = True)
    
    st.plotly_chart(plotly_plot_roc, use_container_width = True)
    
    st.plotly_chart(plotly_plot_mat,use_container_width = True)