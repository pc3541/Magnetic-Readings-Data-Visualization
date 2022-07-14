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
import io

st.sidebar.title("Magnetic Readings Data Visualization")
input_file1 = st.sidebar.file_uploader("Upload first .raw file (for gradient/deviation analysis):")
gradient_numerator = st.sidebar.number_input("nT: gradient threshold (_/min)", value=10, step=1)
gradient_denominator = st.sidebar.number_input("min: gradient threshold (nT/_)", value=10, step=1)
variation_from_chord = st.sidebar.number_input("nT: threshold for variation from chord", value=10, step=1)
variation_chord_duration = st.sidebar.number_input("min: chord duration", value=10, step=1)
input_file2 = st.sidebar.file_uploader("Upload second .raw file:")
input_file3 = st.sidebar.file_uploader("Upload .xyz file (flight lines):")
time_start = st.sidebar.number_input("Desired time segment start (HHMMSS)", value=0, step=1)
time_end = st.sidebar.number_input("Desired time segment end (HHMMSS)", value=240000, step=1)

time_list = list(range(0, 86401))

def calc_slope(x):
    slope = np.polyfit(range(len(x)), x, 1)[0]
    slope = slope/6
    return slope

def getSec(s):
    datee = datetime.datetime.strptime(s, "%H%M%S")
    return datee.hour * 3600 + datee.minute * 60 + datee.second

def run():
    time_start = getSec(str(time_start))
    time_end = getSec(str(time_end))
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

    dataframe_final["Gradients"] = abs((dataframe_final["Magnetic_Readings"].rolling(gradient_denominator*10).apply(calc_slope))*(gradient_denominator*60))
    df_merged_slopes = pd.merge(df_merged, dataframe_final, how='left')
    fig = plt.figure(figsize=(20,4))
    plt.scatter(df_merged_slopes["Time"], df_merged_slopes["Gradients"], 0.25, "black")
    plt.xlabel("Time (sec)")
    plt.ylabel("Gradient (nT/" + str(gradient_denominator) + " min)")
    plt.axhline(y=gradient_numerator, color='r', linestyle='-', label=("Threshold: " + str(gradient_numerator) + " nt/" + str(gradient_denominator) + " min"))
    plt.legend(loc = 'upper left')
    plt.xlim(time_start, time_end)
    st.pyplot(fig)

    dataframe_final["Chord"] = abs(dataframe_final['Magnetic_Readings'].rolling(variation_chord_duration*10, center=True).apply(lambda x: x.iloc[0]+x.iloc[-1]))/2
    df_merged_chord = pd.merge(df_merged, dataframe_final, how="left")
    df_merged_chord["Variation From Chord"] = abs(df_merged_chord['Magnetic_Readings'] - df_merged_chord["Chord"])
    fig = plt.figure(figsize=(20,4))
    plt.scatter(df_merged_chord["Time"], df_merged_chord["Variation From Chord"], 0.25, "black")
    plt.xlabel("Time (sec)")
    plt.ylabel("Variation From " + str(variation_chord_duration) + " min Chord (nT)")
    plt.axhline(y=variation_from_chord, color='r', linestyle='-', label=("Threshold: " + str(variation_from_chord) + " nt/" + str(variation_chord_duration) + " min"))
    plt.legend(loc = 'upper left')
    plt.xlim(time_start, time_end)
    st.pyplot(fig)

    aberrant = pd.DataFrame(columns=df_merged_chord.columns)
    cond = df_merged_chord["Variation From Chord"] > variation_from_chord
    rows = df_merged.loc[cond, :]
    aberrant = pd.concat([aberrant, rows], ignore_index=True)
    cond = df_merged_slopes["Gradients"] > gradient_numerator
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

    if (df_merged_slopes["Gradients"] > gradient_numerator).any() == True or (df_merged_chord["Variation From Chord"] > variation_from_chord).any() == True:
        fig = plt.figure(figsize=(20,4))
        plt.scatter(df_merged["Time"], df_merged["Magnetic_Readings"], 0.25, "black", label="Unit " + str(df_merged["Unit"].iat[0]))
        plt.scatter(df_merged2["Time"], df_merged2["Magnetic_Readings"], 0.25, "grey", label="Unit " + str(df_merged2["Unit"].iat[0]))
        plt.scatter(aberrant["Time"], aberrant["Magnetic_Readings"], 0.25, "red")
        y_lower = plt.gca().get_ylim()[0]
        y_upper = plt.gca().get_ylim()[1]
        for index, row in molecule.iterrows():
            plt.vlines(x=[[row["Time"],row["Time2"]]], ymin=y_lower, ymax=y_upper, colors="black", ls='--', lw=0.5)
            plt.text(row["Time"],y_upper,row["Line"],rotation="vertical",fontsize=7.5)
        plt.xlabel("Time (sec)")
        plt.ylabel("Magnetic Readings (nT)")
        plt.legend(loc = 'upper left')
        plt.xlim(time_start, time_end)
        st.pyplot(fig)
    else:
        fig = plt.figure(figsize=(20,4))
        plt.scatter(df_merged["Time"], df_merged["Magnetic_Readings"], 0.25, "black", label="Unit " + str(df_merged["Unit"].iat[0]))
        plt.scatter(df_merged2["Time"], df_merged2["Magnetic_Readings"], 0.25, "grey", label="Unit " + str(df_merged2["Unit"].iat[0]))
        y_lower = plt.gca().get_ylim()[0]
        y_upper = plt.gca().get_ylim()[1]
        for index, row in molecule.iterrows():
            plt.vlines(x=[[row["Time"],row["Time2"]]], ymin=y_lower, ymax=y_upper, colors="black", ls='--', lw=0.5)
            plt.text(row["Time"],y_upper,row["Line"],rotation="vertical",fontsize=7.5)
        plt.xlabel("Time (sec)")
        plt.ylabel("Magnetic Readings (nT)")
        plt.legend(loc = 'upper left')
        plt.xlim(time_start, time_end)
        st.pyplot(fig)

if st.sidebar.button("Run analysis"):
    run()
