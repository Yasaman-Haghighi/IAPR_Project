import re
import os

import cv2 as cv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from model_2 import model_2
from card_segmentation import *
from onej import model_onej
from model_8j import model_8j
from suits_nn import model_suits
from PIL import Image

## Here we load saved weights from our own networks

model_2.load_weights('C:/Users/Neo/iapr/project/all_weights/rank/model_2_w8')
model_onej.load_weights('C:/Users/Neo/iapr/project/all_weights/one_j/one_j_w8')
model_suits.load_weights('C:/Users/Neo/iapr/project/all_weights/suit/w8_suits_v7')
model_8j.load_weights('C:/Users/Neo/iapr/project/all_weights/eight_j/8_j_w8')

## The following function loops through the input folder and all the folders inside. It checks for image extentions and
## calls the card detection function that ouputs ranks, suits, dealer, centers and errors. If errors != 0 it will 
## preprocess the image with a second method and call the card detection function again. Each suit and rank is ran 
## through the corresponding network. For ranks we double check with model_onej if our prediction is 1 or 10, and with
## model_8j if we predict an 8. Throughout the loops we calculate scores for both standard and advanced modes.
## Default argument is plot=False, if specified as True it will display all the detected masks ranks and suits.

def compute_all(plot=False):
    thr = np.array(100, dtype='uint8')
    thr_suits=np.array(100, dtype='uint8')
    upper = np.array(255, dtype='uint8')
    extensions = [".jpg", ".jpeg", ".png"]
    folder_path = 'C:/Users/Neo/iapr/project/train_games'
    columns=['','P1','P2','P3','P4','D']
    games_df = pd.DataFrame(columns=columns)
    games_pts_standard=[0,0,0,0]
    games_pts_advanced=[0,0,0,0]
    for fldr in sorted(os.listdir(folder_path)):
        sub_folder_path = os.path.join(folder_path, fldr)
        pts_standard=[0,0,0,0]
        pts_advanced=[0,0,0,0]
        for f,n in zip(sorted(os.listdir(sub_folder_path)),range(len(os.listdir(sub_folder_path)))):
            if os.path.splitext(f)[1] in extensions:
                file_path = os.path.join(sub_folder_path, f)
                file_path = file_path.replace('\\', '/')
                img = cv.imread(file_path)
                mask = preprocessing(img)
                ranks, suits, dealer, centers, errors = card_detection(img, mask)
                if errors:
                    mask=preprocessing_new(img)
                    ranks, suits, dealer, centers, errors = card_detection(img, mask)
                if plot:
                    fig, ax = plt.subplots(1, 2, figsize=(17, 9))
                    display_image(img, axes=ax[0])
                    ax[0].set_title('Input')
                    display_image(mask, axes=ax[1], cmap='gray')
                    ax[1].set_title('Detected Mask')
                    plt.show()
                single_round=[]
                single_round.append(int(re.findall(r'\d+', f)[0]))
                rank_values=[0,0,0,0]
                suit_values=[]
                for i in range(len(np.array(ranks))):
                    rank_th = cv.inRange(ranks[i], thr, upper)
                    p1=np.array(Image.fromarray(rank_th).resize(size=(28, 28)))
                    p1=np.expand_dims(p1, axis=0)
                    rank_pred=np.argmax(model_2.predict(p1))
                    p2 = cv.inRange(suits[i], thr_suits, upper)
                    p2=np.expand_dims(p2, axis=0)
                    suit_pred=np.argmax(model_suits.predict(p2))
                    if suit_pred==0:
                        suit_pred="S"
                        suit_values.append('S')
                    elif suit_pred==1:
                        suit_pred="C"
                        suit_values.append('C')
                    elif suit_pred==2:
                        suit_pred="D"
                        suit_values.append('D')
                    elif suit_pred==3:
                        suit_pred="H"
                        suit_values.append('H')
                    if rank_pred==1 or rank_pred==10:
                        rank_pred=np.argmax(model_onej.predict(p1))
                        if rank_pred==0:
                            rank_values[i]=10
                            rank_pred=10
                        elif rank_pred==1 :
                            rank_values[i]=1
                            rank_pred=1
                    if rank_pred==8:
                        rank_pred=np.argmax(model_8j.predict(p1))
                        if rank_pred==0:
                            rank_values[i]=10
                            rank_pred=10
                        elif rank_pred==1 :
                            rank_values[i]=8
                            rank_pred=8
                    if rank_pred == 10:
                        rank_pred="J"
                        rank_values[i]=10
                    elif rank_pred == 11:
                        rank_pred="Q"
                        rank_values[i]=11
                    elif rank_pred == 12:
                        rank_pred="K"
                        rank_values[i]=12
                    else:
                        rank_values[i]=rank_pred
                        rank_pred=str(rank_pred)
                    single_round.append(rank_pred+suit_pred)
                    if plot:
                        fig, ax = plt.subplots(1, 2, figsize=(10, 9))
                        display_image(ranks[i], axes=ax[0])
                        ax[0].set_title('Detected rank')
                        display_image(suits[i], axes=ax[1])
                        ax[1].set_title('Detected suit')
                    print("Predicted rank and suit : {}".format(single_round[i+1]))
                round_advanced=np.vstack([suit_values,rank_values])
                df_round_advanced=pd.DataFrame(round_advanced)
                this_round_suit=round_advanced[0,find_dealer(centers, dealer)]
                tmp_df=df_round_advanced.T[df_round_advanced.T[0].str.contains(this_round_suit)]
                adv_winner_index=tmp_df[1].astype({1: 'int32'}).idxmax(axis=1)
                pts_standard[np.argmax(rank_values)] +=1
                pts_advanced[adv_winner_index]+=1
                single_round.append(find_dealer(centers, dealer) + 1)
                df_length = len(games_df)
                games_df.loc[df_length] = single_round
        games_pts_standard=np.vstack([games_pts_standard,pts_standard])
        games_pts_advanced=np.vstack([games_pts_advanced,pts_advanced])
    games_pts_standard = np.delete(games_pts_standard, 0, 0)
    games_pts_advanced = np.delete(games_pts_advanced, 0, 0)
    return(games_df,games_pts_standard,games_pts_advanced)

## This is only relevant if we have the answers to our set, it will output 7 boolean matrices comparing every single
## element of our predicted dataframe to the provided one
def boolean_matrices(df):
    num_rounds=13
    num_games=7
    tmp=[]
    for k in range(num_games):
        tmp.append(k*num_rounds)
    extensions = [".csv"]
    folder_path = 'C:/Users/Neo/iapr/project/train_games'
    for fldr,i in zip(os.listdir(folder_path),tmp):
        sub_folder_path = os.path.join(folder_path, fldr)
        for f,n in zip(os.listdir(sub_folder_path),range(len(os.listdir(sub_folder_path)))):
            if os.path.splitext(f)[1] in extensions:
                file_path = os.path.join(sub_folder_path, f)
                file_path = file_path.replace('\\', '/')
                print(file_path)
                print(df[0+i:i+13].sort_values("").values==pd.read_csv(file_path).values)