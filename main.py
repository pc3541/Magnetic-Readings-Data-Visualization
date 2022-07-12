#Streamlit

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from numpy_ext import rolling_apply
import datetime
import time
from matplotlib.pyplot import figure
import voila
import io

st.sidebar.title("Magnetic Readings Data Visualization")
input_file1 = st.sidebar.file_uploader("Upload first .raw file (for gradient/deviation analysis):")
gradient_numerator = st.sidebar.number_input("nT: gradient threshold (_/min)", step=1)
gradient_denominator = st.sidebar.number_input("min: gradient threshold (nT/_)", step=1)
variation_from_chord = st.sidebar.number_input("nT: threshold for variation from 600s chord")
input_file2 = st.sidebar.file_uploader("Upload second .raw file:")
input_file2 = st.sidebar.file_uploader("Upload .xyz file (flight lines):")

time_list = list(range(0, 86401))

def calc_slope(x):
    slope = np.polyfit(range(len(x)), x, 1)[0]
    slope = slope/6
    return slope

def getSec(s):
    datee = datetime.datetime.strptime(s, "%H%M%S")
    return datee.hour * 3600 + datee.minute * 60 + datee.second

def run():
    st.progress(10)
    df = pd.read_csv(input_file1, delim_whitespace=True, header=None)
    df = df.drop(df.columns[[0, 4]], axis=1)
    df = df.dropna()
    dataframe = df.astype(str)
    dataframe_final = dataframe[~dataframe.iloc[:,2].str.contains("?", regex=False)]
    dataframe = dataframe.astype(float)
    dataframe_final = dataframe.astype(int)
    dataframe_final[dataframe_final.columns[3]] = dataframe_final[dataframe_final.columns.values[3]]/10
    dataframe_final.columns = ["Unit", "Date","Time", "Magnetic_Readings"]

    dataframe_final["Time"] = dataframe_final["Time"].astype(str)
    for i in range(len(dataframe_final)):
        dataframe_final["Time"].values[i] = getSec(dataframe_final["Time"].values[i])

    dataframe_final["Time"] = dataframe_final["Time"].astype(int)
    df_time = pd.DataFrame(time_list, columns=["Time"])
    df_merged = pd.merge(df_time, dataframe_final, on='Time')

    df2 = pd.read_csv(input_file2, delim_whitespace=True, header=None)
    df2 = df2.drop(df2.columns[[0, 4]], axis=1)
    df2 = df2.dropna()
    dataframe2 = df2.astype(str)
    dataframe_final2 = dataframe2[~dataframe2.iloc[:,2].str.contains("?", regex=False)]
    dataframe2 = dataframe2.astype(float)
    dataframe_final2 = dataframe2.astype(int)
    dataframe_final2[dataframe_final2.columns[3]] = dataframe_final2[dataframe_final2.columns.values[3]]/10
    dataframe_final2.columns = ["Unit","Date","Time", "Magnetic_Readings"]

    dataframe_final2["Time"] = dataframe_final2["Time"].astype(str)
    for i in range(len(dataframe_final2)):
        dataframe_final2["Time"].values[i] = getSec(dataframe_final2["Time"].values[i])

    dataframe_final2["Time"] = dataframe_final2["Time"].astype(int)
    df_merged2 = pd.merge(df_time, dataframe_final2, on='Time')

    dataframe_final["Gradients"] = abs((dataframe_final["Magnetic_Readings"].rolling(gradient_denominator.value*10).apply(calc_slope))*(gradient_denominator.value*60))
    df_merged_slopes = pd.merge(df_merged, dataframe_final, how='left')
    plt.figure(figsize=(20,4))
    plt.scatter(df_merged_slopes["Time"], df_merged_slopes["Gradients"], 0.25, "black")
    plt.xlabel("Time (sec)")
    plt.ylabel("Gradient (nT/" + str(gradient_denominator.value) + " min)")
    plt.axhline(y=gradient_numerator.value, color='r', linestyle='-', label=("Threshold: " + str(gradient_numerator.value) + " nt/" + str(gradient_denominator.value) + " min"))
    plt.legend(loc = 'upper left')
    plt.show()

    dataframe_final["600s Chord"] = abs(dataframe_final['Magnetic_Readings'].rolling(100, center=True).apply(lambda x: x.iloc[0]+x.iloc[-1]))/2
    df_merged_chord = pd.merge(df_merged, dataframe_final, how="left")
    df_merged_chord["Variation From 600s Chord"] = abs(df_merged_chord['Magnetic_Readings'] - df_merged_chord["600s Chord"])
    plt.figure(figsize=(20,4))
    plt.scatter(df_merged_chord["Time"], df_merged_chord["Variation From 600s Chord"], 0.25, "black")
    plt.xlabel("Time (sec)")
    plt.ylabel("Variation From 600s Chord (nT)")
    plt.axhline(y=variation_from_chord.value, color='r', linestyle='-', label=("Threshold: " + str(variation_from_chord.value) + " nt"))
    plt.legend(loc = 'upper left')
    plt.show()

    aberrant = pd.DataFrame(columns=df_merged_chord.columns)
    cond = df_merged_chord["Variation From 600s Chord"] > variation_from_chord.value
    rows = df_merged.loc[cond, :]
    aberrant = pd.concat([aberrant, rows], ignore_index=True)
    cond = df_merged_slopes["Gradients"] > gradient_numerator.value
    rows = df_merged.loc[cond, :]
    aberrant = pd.concat([aberrant, rows], ignore_index=True)

    molecule = pd.read_csv(input_file3, names=['Line', 'Aircraft', 'Flight', 'YYMMDD', 'Date', 'Time', 'DateU',
   'TimeU', 'Zn', 'Easting', 'Northing', 'Lat', 'Long', 'xTrack',
   'Knots2D', 'Knots3D', 'KnotsAir', 'ZFid_ms', 'KFid', 'AFid', 'MagTF1U',
   'Mag8D', 'VecX', 'VecY', 'VecZ', 'VecTF', 'MagRatio', 'GPSHt', 'Undul',
   'Sats', 'HDop', 'DGPS', 'RadAlt', 'BaroHPa', 'Temp', 'Humid', 'AN5_v',
   'Dn', 'Up', 'Samp', 'Live', 'RawTC', 'RawK', 'RawU', 'RawTh', 'RawUp_U',
   'Cosm', 'OrigCSum'],delim_whitespace=True, skiprows=[0,1,2], header=None)
    molecule = molecule[["Line","Date","Time"]]
    molecule = molecule.replace('*', np.NaN)
    molecule = molecule.dropna()
    molecule = molecule.groupby('Line').apply(lambda x: x.iloc[[-1, 0]]).reset_index(drop=True)
    molecule['Time2'] = molecule.groupby('Line')['Time'].shift()
    molecule = molecule.dropna()
    molecule = molecule.reset_index(drop=True)
    molecule = molecule.astype(str)
    molecule = molecule.astype({'Line': 'int32',"Time":"float64", "Time2":"float64"})

    if (df_merged_slopes["Gradients"] > gradient_numerator.value).any() == True or (df_merged_chord["Variation From 600s Chord"] > variation_from_chord.value).any() == True:
        plt.figure(figsize=(20,4))
        plt.scatter(df_merged["Time"], df_merged["Magnetic_Readings"], 0.25, "black", label="Unit " + str(df_merged["Unit"].iat[0]))
        plt.scatter(df_merged2["Time"], df_merged2["Magnetic_Readings"], 0.25, "grey", label="Unit " + str(df_merged2["Unit"].iat[0]))
        plt.scatter(aberrant["Time"], aberrant["Magnetic_Readings"], 0.25, "red")
        y_lower = plt.gca().get_ylim()[0]
        y_upper = plt.gca().get_ylim()[1]
        for index, row in molecule.iterrows():
            plt.vlines(x=[[row["Time"],row["Time2"]]], ymin=y_lower, ymax=y_upper, colors=row["Edges"], ls='--', lw=0.5, label=row["Line"])
        plt.xlabel("Time (sec)")
        plt.ylabel("Magnetic Readings (nT)")
        plt.legend(loc = 'upper left')
        plt.show()
    else:
        plt.figure(figsize=(20,4))
        plt.scatter(df_merged["Time"], df_merged["Magnetic_Readings"], 0.25, "black", label="Unit " + str(df_merged["Unit"].iat[0]))
        plt.scatter(df_merged2["Time"], df_merged2["Magnetic_Readings"], 0.25, "grey", label="Unit " + str(df_merged["Unit"].iat[0]))
        y_lower = plt.gca().get_ylim()[0]
        y_upper = plt.gca().get_ylim()[1]
        x_bounds = plt.gca().get_xlim()
        for index, row in molecule.iterrows():
            plt.vlines(x=[[row["Time"],row["Time2"]]], ymin=y_lower, ymax=y_upper, colors="black", ls='--', lw=0.5)
            plt.text(row["Time"],y_upper,row["Line"],rotation="vertical",fontsize=7.5)
        plt.xlabel("Time (sec)")
        plt.ylabel("Magnetic Readings (nT)")
        plt.legend(loc = 'upper left')
        plt.show()

if st.button("Run analysis"):
    run()
