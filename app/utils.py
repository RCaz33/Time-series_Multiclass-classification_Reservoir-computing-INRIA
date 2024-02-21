import time
import pandas as pd
import numpy as np



def calculate_position_irregular(acceleration, timesteps):
    position = np.zeros_like(acceleration)
    for i in range(1, len(acceleration)):
        dt1 = timesteps[i - 1]
        dt2 = timesteps[i]
        average_acceleration = (acceleration[i - 1] + acceleration[i]) / 2
        position[i] = position[i - 1] + dt1 * average_acceleration + dt2 * average_acceleration
    return position


def double_integrate(df, col):
    df.sort_values(by='t', inplace=True)
    velocities = df[col].cumsum()
    positions = velocities.cumsum()
    return positions

def linearize(u):
    """
    Fonction pour mettre en 1 seule lignes le tableau de statistiques pd.describe()
    """
    all=[]
    for line in range(len(u)):
        all.append(u.iloc[line])
    return pd.concat(all, axis=0).T

def stepwise_predictions(data__):
    all_data=[]
    t=[]
    for i in range(len(data__)):
        data = data__.iloc[:i,:]   
        t0=time.time()
        # ajout des positions
        data['pos_x'] = double_integrate(data,'raw_acceleration_x')
        data['pos_y'] = double_integrate(data,'raw_acceleration_y')
        data['pos_z'] = double_integrate(data,'raw_acceleration_z')
        # feature selection / engineering 
        data.drop(columns='t', inplace=True)
        u = data.describe().T
        u.drop(columns='count', inplace=True)
        number = pd.DataFrame(linearize(u)).T
        # time segments
        n = len(data)//3   # nbe de points par quartiers
        for j in range(3):
            quart_ = data[j*n:(j+1)*n]#.drop(columns='t')
            int_ = quart_.describe().T
            int_.drop(columns='count', inplace=True)
            number = pd.concat([number,pd.DataFrame(linearize(int_)).T],axis=1)
        # changement de signes
        for feature in data.columns[1:-1]:  # on ne prend pas le temps ni le label
            col_name = 'sign_change_' + feature
            data[col_name] = data[feature].apply(lambda x: 1 if x >= 0 else -1)
            number[col_name] = (data[col_name] * data[col_name].shift(-1) < 0).sum()
        t.append(time.time()-t0)
        all_data.append(number)   
    return pd.concat(all_data), t


def make_binary(arr):
    mask = np.zeros(10)
    mask[arr] = 1
    return mask
