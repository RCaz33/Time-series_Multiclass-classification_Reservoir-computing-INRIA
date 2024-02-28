# import core librairies
import streamlit as st
import os
import random
import time
import subprocess



# Function to install dependencies if not already installed
def install_dependencies():
    with st.spinner('Installing dependencies...'):
        time.sleep(0.05)
        subprocess.check_call(["pip", "install", "-r", "requirements.txt"])

if not os.path.exists(".installed"):
    install_dependencies()
    open(".installed", "w").close()

# import librairies
import pandas as pd
import numpy as np
from utils import calculate_position_irregular, stepwise_predictions, linearize, make_binary, double_integrate
import pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import plotly.express as px
import seaborn as sns
import reservoirpy as rpy





def main():
    st.title('AI4Industry 2024: UseCase CATIE')
    st.header('6Tron sensor for number determination')
    row1_col1, row1_col2= st.columns(2)
    with st.container(border=True):
        with row1_col1:
            group = st.selectbox('select a dataset', ['Horizontal group1','Horizontal group2','Vertical group3'])
            st.text("After selecting the number to predict\nand the iteration, press button below")
        # get iterations
        if group == 'Horizontal group1':
            folder = '../data/group3/config_1'
            _ = [int(i.split('_')[1].split('.')[0]) for i in os.listdir(folder)]
            iter_min, iter_max = np.min(_), np.max(_)
        elif group == 'Horizontal group2':
            folder = '../data/h_config1-lcb'
            _ = [int(i.split('_')[1].split('.')[0]) for i in os.listdir(folder)]
            iter_min, iter_max = np.min(_), np.max(_)
        elif group == 'Vertical group3':
            folder = '../data/v_config1-lcb'
            _ = [int(i.split('_')[1].split('.')[0]) for i in os.listdir(folder)]
            iter_min, iter_max = np.min(_), np.max(_)
        
            
    with st.container(border=True):
        with row1_col2:
            number = st.number_input('Select a number',min_value=0,max_value=9,value=5)
            selected_number = str(number)
            iteration = st.number_input('Select iteration', min_value=iter_min+1,max_value=iter_max+1,value=10)
            

    # get filename
    if group == 'Horizontal group1':
        filename = folder + '/' + selected_number + '_' + str(iteration) + '.csv'
    elif group == 'Horizontal group2':
        filename = folder + '/h_' + selected_number + '_' + str(iteration) + '.csv'
    elif group == 'Vertical group3':
        filename = folder + '/v_config1_' + selected_number + '_' + str(iteration) + '.csv'

    button = st.button('Press to proceed')

    if button:
        row2_col1, row2_col2= st.columns(2)

        # load file
        data = pd.read_csv(filename)
        # calculate position 2D
        position_x = calculate_position_irregular(data.raw_acceleration_x, data.t)
        position_y = calculate_position_irregular(data.raw_acceleration_y, data.t)
        position_z = calculate_position_irregular(data.raw_acceleration_z, data.t)
        # calculate position for model ESN
        data['pos_x'] = double_integrate(data,'raw_acceleration_x')
        data['pos_y'] = double_integrate(data,'raw_acceleration_y')
        data['pos_z'] = double_integrate(data,'raw_acceleration_z')

        

        with st.spinner('Calculate step-by-step predictions...'):
                # calculate features
                num,_ = stepwise_predictions(data)
                # make predictions
                model = pickle.load(open('ressources/RandomForestClasssifer_2D.sav', 'rb'))
                scaler = pickle.load(open('ressources/scaler', 'rb'))
                preds = (model.predict(scaler.transform(num.fillna(0))))
                with open('ressources/esn_model3.pickle', 'rb') as f:
                    model_esn = pickle.load(f)
                print(data.shape)
                print(model_esn.nodes)
                print(data)
                preds_esn = [np.argmax(a) for a in model_esn.run(np.array(data))]
                
                # display predictions
                target_bin = np.tile(make_binary(number),(len(preds),1))
                preds_bin = [make_binary(b) for b in preds]
                preds_bin_esn = [make_binary(b) for b in preds_esn]


        with st.spinner('Create rendering...'):
            # make figure
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 6))  # Create a figure with two axes
            t = len(data)
            # initialize scatter plot
            ax1.scatter(position_x[0], position_y[0], c="b", s=5)
            
            ax1.set_title("Position")
            ax2.set_yticklabels("")
            ax2.set_title("Random Forest")
            ax3.set_title("Reservoir Computing")

            
            def update(i):
                ax1.scatter(position_x[:i], position_y[:i], c="b", s=5)
                ax1.set(xlim=[position_x.min(),position_x.max()],\
                    ylim=[position_y.min(),position_y.max()],\
                        xlabel='position X', ylabel='position Y')
                pred_bin = np.concatenate((preds_bin[:1+i], np.tile(np.zeros(10), (len(preds)-i+1, 1))), axis=0)
                pred_esn = np.concatenate((preds_bin_esn[:1+i], np.tile(np.zeros(10), (len(preds)-i+1, 1))), axis=0)
                sns.heatmap(pred_bin,alpha=0.5, cbar=False, yticklabels=False, ax=ax2)
                sns.heatmap(target_bin,alpha=0.2, cbar=False, yticklabels=False, ax=ax2)
                sns.heatmap(pred_esn,alpha=0.5, cbar=False, yticklabels=False, ax=ax3)
                sns.heatmap(target_bin,alpha=0.2, cbar=False, yticklabels=False, ax=ax3)

            ani = FuncAnimation(fig, update, frames=len(data), interval=200)
            ani.save('animation.gif', writer='pillow', fps=10)
                    
        st.title('Animated Plot')
        st.image('animation.gif')


if __name__ == '__main__':
    main()