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
    # Create a flag file to indicate that dependencies are installed
    open(".installed", "w").close()


### install dependencies



import pandas as pd
import numpy as np
from utils import calculate_position_irregular, stepwise_predictions, linearize, make_binary
import pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import plotly.express as px
import seaborn as sns






def main():
    st.title('6Tron sensor number determination')



    row1_col1, row1_col2= st.columns(2)
    with st.container(border=True):
        with row1_col1:
            group = st.selectbox('select a group', ['group1_h','group2_h','group3_v'])
        with row1_col2:
            number = st.selectbox('select a number', range(10))

        # get iterations
        if group == 'group1_h':
            folder = '../data/group3/config_1'
            iterations = np.max([int(i.split('_')[1].split('.')[0]) for i in os.listdir(folder)])
        elif group == 'group2_h':
            folder = '../data/h_config1-lcb'
            iterations = np.max([int(i.split('_')[2].split('.')[0]) for i in os.listdir(folder)])
        elif group == 'group3_v':
            folder = '../data/v_config1-lcb'
            iterations = np.max([int(i.split('_')[3].split('.')[0]) for i in os.listdir(folder)])

        with row1_col2:
            iteration = st.selectbox('select iteration', np.arange(iterations)+1)

    # get filename
    if folder == '../data/group3/config_1':
        filename = folder + '/' + str(number) + '_' + str(iteration) + '.csv'
    elif folder == '../data/h_config1-lcb':
        filename = folder + '/h_' + str(number) + '_' + str(iteration) + '.csv'
    elif folder == '../data/v_config1-lcb':
        filename = folder + '/v_config1_' + str(number) + '_' + str(iteration) + '.csv'


    row2_col1, row2_col2= st.columns(2)

    # load file
    data = pd.read_csv(filename)
    # calculate position
    position_x = calculate_position_irregular(data.raw_acceleration_x, data.t)
    position_y = calculate_position_irregular(data.raw_acceleration_y, data.t)
    position_z = calculate_position_irregular(data.raw_acceleration_z, data.t)



    with st.spinner('Calculate step-by-step predictions...'):
            # calculate features
            num,_ = stepwise_predictions(data)
            # make predictions
            model = pickle.load(open('ressources/RandomForestClasssifer_2D.sav', 'rb'))
            scaler = pickle.load(open('ressources/scaler', 'rb'))
            preds = (model.predict(scaler.transform(num.fillna(0))))
            # display predictions
            preds_bin = [make_binary(b) for b in preds]
            target_bin = np.tile(make_binary(number),(len(preds),1))


    but_ = st.button('press to start simulation')
    if but_:
        # make figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))  # Create a figure with two axes
        t = len(data)
        # initialize scatter plot
        ax1.scatter(position_x[0], position_y[0], c="b", s=5)
        ax1.set(xlim=[position_x.min(),position_x.max()],\
                ylim=[position_y.min(),position_y.max()],\
                    xlabel='position X', ylabel='position Y')
        the_plot = st.pyplot(fig)
        def update():
            x = position_x[:i]
            y = position_y[:i]
            #data_slice = np.stack([x, y]).T
            #scat.set_offsets(data_slice)
            ax1.scatter(x,y,c='b',s=5)

            pred_bin = (np.concatenate((preds_bin[:1+i] , np.tile(np.zeros(10),(len(preds)-i+1,1))), axis=0))
            sns.heatmap(pred_bin,alpha=0.5, cbar=False, yticklabels=False, ax=ax2)
            sns.heatmap(target_bin,alpha=0.2, cbar=False, yticklabels=False, ax=ax2)
            the_plot.pyplot(fig)
        

        for i in range(t): 
            update()


if __name__ == '__main__':
    main()