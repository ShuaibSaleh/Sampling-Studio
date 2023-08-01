from cmath import phase
from math import ceil
from re import X
from tkinter import CENTER, font
from turtle import color
from turtle import title
from lib2to3.pgen2.token import EQEQUAL
from pyparsing import alphas
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import plotly
import csv
import os
import math
from traitlets import default
import scipy
from scipy.signal import find_peaks

# Page Layout

st.set_page_config(layout='wide')
with open('app.css') as fileStyle:
    st.markdown(f'<style>{fileStyle.read()}</style>', unsafe_allow_html=True)

# Functions
# Plotting


def plot(gen=False, signal='', time=[], value=[], sampleTime=[], sampleValue=[], value_rec=[], hide_original=False, sampling=False, interp=False, x_axis='Time (s)', y_axis='Amplitude'):
    fig = plt.figure(figsize=(1, 6))
    plt.xlabel(x_axis, fontsize=17)
    plt.ylabel(y_axis, fontsize=17)
    plt.title(signal, fontsize=25)
    if (gen):
        plt.xlim([0,2])

    if(not hide_original):
        plt.plot(time, value)  # Plot oriniganl function

    if (sampling):
        plt.plot(sampleTime, sampleValue, 'ro')  # Sampling points

    if (sampling and hide_original and not interp):
        plt.title("Sampling points", fontsize=25)

    if (interp and not hide_original):
        plt.plot(time, value_rec, '--')  # Sampling interpulation

    if (interp and hide_original):
        plt.title("Reconstructed Signal", fontsize=25)
        plt.plot(time, value_rec, 'orange')  # Sampling interpulation

    st.plotly_chart(fig, use_container_width=True)

# Summation of multiple sinewaves


def summation_sins(amplitude, frequency, time_axis):
    number_of_records = len(frequency)
    sinewave = np.zeros(len(time_axis))
    for i in range(number_of_records):
        sinewave += amplitude[i] * np.cos(2 * np.pi * frequency[i] * time_axis)
    return sinewave


# Noise generation

def Noise_using_snr(snr, signal_value):
    sigpower = sum([math.pow(abs(signal_value[i]), 2) for i in range(len(signal_value))])
    sigpower = sigpower/len(signal_value)
    noisepower = sigpower/(math.pow(10, snr/10))
    random_state = np.random.RandomState(0)
    noise = math.sqrt(noisepower)*(random_state.uniform(-1, 1, size=len(signal_value)))
    return noise

#  Generation Page

inputs_col, space, graph_col, space, sigProcessing_col = st.columns([1, 0.1, 3, 0.1, 1])

with inputs_col:

    # Upload Signal
    uploaded_file = st.file_uploader("Upload Signal as CSV", type={"csv"})

    # Summation of inputs
    frequency_input = st.number_input("Frequency (Hz)", value=1, step=1)
    amplitude_input = st.number_input("Amplitude", value=1, step=1)
    signal_name = st.text_input("Signal Name", value="Signal_name")

    if st.button("Add Signal"):

        if frequency_input > 0 and amplitude_input != 0 and signal_name != "":
            Data = [frequency_input, amplitude_input, signal_name]
            upload = False
            if os.path.exists("DataFile.csv"):
                with open('DataFile.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(Data)
            else:
                with open('DataFile.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(['frequency', "Amplitude", "Signal_name"])
                    writer.writerow(Data)

    # Deletion
    # remove_specific_row_from_csv(df, "id", id_signal)
    signal_names = pd.read_csv("DataFile.csv").iloc[:, 2]
    added_signal = st.selectbox(
        'select signal you want to delete', (signal_names))

    if st.button("Delete Signal"):
        df = pd.read_csv("DataFile.csv")
        df = df[df.Signal_name != added_signal]
        df.to_csv("DataFile.csv", index=False)

with sigProcessing_col:

    DataTowCo = pd.read_csv("DataFile.csv")
    frequency = DataTowCo.iloc[:, 0]
    Amplitude = DataTowCo.iloc[:, 1]

    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        df = input_df
        time_data = df[df.head(0).columns[0]]
        amplitude_data = df[df.head(0).columns[1]]
        try:
            max_freq = df.iloc[:, 2][0]
        except:
            max_freq=1


    if uploaded_file is None:
        max_freq = max(frequency, default=1)

    # Noise and Sampling Sliders
    snr = st.slider('SNR', min_value=1, max_value=100, step=1, value=100)
    s_Interpolation = st.checkbox('Show Interpolation', value=True)

    sampling = st.checkbox('Show Samples Points')

    sampling_options = st.selectbox('Sampling Frequency Options', ["Hz", f"Nequist Sampling: {max_freq} Hz"], key="Options")
    if sampling_options == "Hz":

        sample_freq = st.slider('Sampling Frequency (Hz)',min_value=1, max_value=100, step=1,value=1)

    else:

        sample_freq = max_freq*st.slider(
            label="", min_value=0.5, max_value=10.0, value=1.0, step=0.5)

    # Sampling CheckBox

    hide_original = st.checkbox('Hide Original Signal')


with graph_col:

    # Plotting Signals
    if uploaded_file is None:
        if os.path.exists("DataFile.csv"):
            DataTowCo = pd.read_csv("DataFile.csv")
            frequency = DataTowCo.iloc[:, 0]
            Amplitude = DataTowCo.iloc[:, 1]
            
            # Original Attributes
            time = np.arange(0, 3, 0.001)
            y_signal = summation_sins(Amplitude, frequency, time)

            # Noise Addittion
            noise = Noise_using_snr(snr, y_signal)
            y_signal = y_signal + noise

            # Sampling Attributes
            sample_rate = sample_freq  # Sampling Frequency
            sample_periodic_time = 1 / sample_rate  # How much time for a full cycle
            time_samples = np.arange(0, 3, sample_periodic_time)    # To spread the samples right on the graph
            
            interp = scipy.interpolate.interp1d(time, y_signal, kind="quadratic")
            y_samples = interp(time_samples) #Get value of sample points

            # Sinc Interpolation
            y_reconstruction = np.zeros(len(time))
            for i in range(0, len(time)):
                for n in range(0, len(time_samples)):
                    y_reconstruction[i] += y_samples[n] * np.sinc((time[i]-time_samples[n])/sample_periodic_time)

            # Plotting Original Signal, Samples and Interpolation
            plot(True,f"", time, y_signal, time_samples, y_samples, y_reconstruction,hide_original, sampling, s_Interpolation)     # Plotting Original Signal

    elif uploaded_file is not None:

        # Read data
        file_name = uploaded_file.name
        file_name = file_name[0:-4]
        x_axis = input_df.iloc[0:0, 0].name
        y_axis = input_df.iloc[0:0, 1].name

        time_maximum = time_data.max()
        time_minimum = time_data.min()
        numberOfRecords = len(time_data)

        # Sampling Attributes
        sample_rate = sample_freq  # sampling frequency
        sample_periodic_time = 1 / sample_rate

        # Add Noise
        noise = Noise_using_snr(snr, amplitude_data)
        amplitude_data = amplitude_data + noise

        numberOfSamples = (time_maximum - time_minimum)/sample_periodic_time
        steps = ceil(numberOfRecords / numberOfSamples)
        time_samples = time_data[0:numberOfRecords:steps]  # Spreading Samples
        y_samples = amplitude_data[0:numberOfRecords:steps]

        # Sinc Interpolation
        y_reconstruction = np.zeros(len(time_data))
        for i in range(0, len(time_data)):
            for x, y in zip(time_samples, y_samples):
                y_reconstruction[i] += y * \
                    np.sinc((time_data[i]-x)/sample_periodic_time)

        # Plotting Original Signal, Samples and Interpolation
        plot(False,file_name, time_data, amplitude_data, time_samples, y_samples, y_reconstruction,hide_original, sampling, s_Interpolation, x_axis, y_axis)  # Plotting Original Signal
