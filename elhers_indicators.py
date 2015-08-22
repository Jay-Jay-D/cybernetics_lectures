# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 10:40:56 2015

@author: jj
"""
import numpy as np
import pandas as pd
from numpy import pi


def fisher(df, norm_window=20):
    """
    Fisher transform, make the series distributions more Gaussian.
    --> Ref: Cybernetic, Figure 1.7
    :param df: Pandas DataFrame with the series to be transformed.
    :param norm_window: The look-back time used to estimate the normalization channel.
    :return: a DataFrame with the transformed series.
    """
    # Normalize the price series.
    df_norm = (df - pd.rolling_min(df, norm_window)) / (pd.rolling_max(df, norm_window) -
                                                        pd.rolling_min(df, norm_window))
    # Center the series on its midpoint and then doubled so that df_value
    # swings between −1 and +1.
    df_value = 2 * (df_norm - 0.5)
    # Smoothing df_value by applying EMA with alpha 0.5.
    df_value = 0.5 * df_value + 0.5 * df_value.shift()
    # Avoid division by zero and weird behavior.
    df_value[df_value < -0.999] = -0.999
    df_value[df_value > 0.999] = 0.999
    # Estimate the Fisher transform and smoothing again with an EMA with alpha 0.5.
    df_fisher = 0.5 * np.log((1 + df_value) / (1 - df_value))
    return 0.5 * df_fisher + 0.5 * df_fisher.shift()


def fisher_inverse(df, norm_window=20, normalization='statistical'):
    """
    Inverse Fisher transform.
    :param df:
    :param norm_window:
    :return:
    :normalization: ('chanelling' or 'statistical')
    """
    # Normalize the price serie.
    if normalization == 'channelling':
        df_norm = (df - pd.rolling_min(df, norm_window)) / \
                  (pd.rolling_max(df, norm_window) - pd.rolling_min(df, norm_window))
        # Center the serie on its midpoint and then doubled so that df_value
        # swings between −1 and +1.
        df_value = 2 * (df_norm - 0.5)
    elif normalization == 'statistical':
        df_value = 2 * (df - pd.rolling_mean(df, norm_window)) / \
                   pd.rolling_std(df, norm_window)
    # Avoid division by zero and weird behavior.
    df_value[df_value < -0.999] = -0.999
    df_value[df_value > 0.999] = 0.999
    df_inverse_fisher = (np.exp(2 * df_value) - 1) / (np.exp(2 * df_value) + 1)
    return df_inverse_fisher


def cyber_cycle(na_series, period):
    """
    Fast Trend.
    --> Ref: Cybernetics, Figure 4.2
    :param na_series: numpy array with the series to be filtered.
    :param period: the SMA equivalent period.
    :return: a numpy array with the series filtered.
    """
    alpha = 2.0 / (period + 1)       # 0.07
    a_ = (1 - alpha / 2.0) ** 2
    b_ = (1 - alpha)

    smooth = np.zeros(len(na_series))
    cycle = np.zeros(len(na_series))
    smooth[:3] = na_series[:3]

    for n in range(3, len(na_series)):
        smooth[n] = (na_series[n] + 2 * na_series[n-1] + 2 * na_series[n-2] + na_series[n-3]) / 6
        if n < 10:
            cycle[n] = (na_series[n] - 2 * na_series[n-1] + na_series[n-2]) / 4
        else:
            cycle[n] = a_ * (smooth[n] - 2 * smooth[n-1] + smooth[n-2]) + \
                       2 * b_ * cycle[n-1] - (b_ ** 2) * cycle[n-2]
    return cycle


def instantaneous_trend(na_series, period):
    """
    Fast Trend.
    --> Ref: Cybernetics, eq 2.9
    :param na_series: numpy array with the series to be filtered.
    :param period: the SMA equivalent period.
    :return: a numpy array with the series filtered.
    """
    alpha = 2.0 / (period + 1)       # 0.07
    a_ = (alpha / 2.0) ** 2
    b_ = (1 - alpha)

    it = np.zeros(len(na_series))
    it[:2] = na_series[:2]

    for n in range(2, len(na_series)):
        it[n] = (alpha - a_) * na_series[n] + (2 * a_) * na_series[n-1] - \
                (alpha - 3 * a_) * na_series[n-2] + \
                (2 * b_) * it[n-1] - (b_ ** 2) * it[n-2]
    return it


def super_smoother(na_series, period, initialize_filter=True):
    """
    Fast Trend.
    A low-pass filter with almost flat frequency response in the  passband. This filter has a
    similar SMA smoothing with less lag.
    --> Ref: Cycle Analytics, eq 3-3
    :param na_series: numpy array with the series to be filtered.
    :param period: (integer) The period of the cutoff frequency.
    :param initialize_filter: (boolean) Initialize the first filter values with the input series? 
                              Set True for faster trend convergence.
                              Set False for use with oscilators.
    :return: a numpy array with the series filtered.
    """
    a = np.exp(-1.414 * pi / period)
    b = 2 * a * np.cos(1.414 * pi / period)
    c2 = b
    c3 = -(a ** 2)
    c1 = 1 - c2 - c3

    ss = np.zeros(len(na_series))
    if initialize_filter:
        ss[:2] = na_series[:2]

    for n in range(2, len(na_series)):
        ss[n] = c1 / 2 * (na_series[n] + na_series[n - 1]) + c2 * ss[n - 1] + c3 * ss[n - 2]
    return ss


def decycle(na_series, period):
    """
    Fast trend.
    Low-pass filter with cutoff frequency equal to period.
    --> Ref: Cycle Analytics, Code Listing 4-1.
    :param na_series: numpy array with the series to be filtered.
    :param period: the period of the cutoff frequency.
    :return: a numpy array with the series filtered.
    """
    alpha = (np.cos(2 * pi / period) + np.sin(2 * pi / period) - 1) / np.cos(2 * pi / period)

    dcycle = np.zeros(len(na_series))
    dcycle[:1] = na_series[:1]

    for n in range(1, len(na_series)):
        dcycle[n] = (alpha / 2) * (na_series[n] + na_series[n - 1]) + (1 - alpha) * dcycle[n - 1]
    return dcycle


def high_pass_filter(na_series, period, initialize_filter=False):
    """
    High-pass filter with cutoff frequency equal to period.
    --> Ref: Cybernetics, eq. 2.6
    :param na_series: numpy array with the series to be filtered.
    :param period: the period of the cutoff frequency.
    :return: a numpy array with the series filtered.
    :param initialize_filter: (boolean) Initialize the first filter values with the input series? 
                              Set True for faster trend convergence.
                              Set False for use with oscilators.
    """
    alpha = (np.cos(2 * pi / period) + np.sin(2 * pi / period) - 1) / np.cos(2 * pi / period)
    a_ = (1 - alpha / 2) ** 2
    b_ = 1 - alpha

    hpf = np.zeros(len(na_series))
    if initialize_filter:
        hpf[:2] = na_series[:2]

    for n in range(2, len(na_series)):
        hpf[n] = a_ * (na_series[n] - 2 * na_series[n - 1] + na_series[n - 2]) + \
                 2 * b_ * hpf[n - 1] - (b_ ** 2) * hpf[n - 2]
    return hpf


def roofing_indicator(na_series, short_period, long_period):
    """
    A wide band-pass filter that allow pass frequencies with period between short_period and
    long-period.
    --> Ref: Cycle Analyitics, Code Listing 7-3.
    :param na_series: a numpy array with the series to be filtered.
    :param short_period: period of the higher cutoff frequency.
    :param long_period: period of the lower cutoff frequency.
    :return: a numpy array with the series filtered.
    """
    hpf = high_pass_filter(na_series, long_period, initialize_filter=False)
    ss = super_smoother(hpf, short_period, initialize_filter=False)
    return ss


def sinewave_indicator(na_series, period, smoothing=10):
    # Modified high pass filter
    alpha = (1 - np.sin(2 * pi / period)) / np.cos(2 * pi / period)
    a_ = 0.5 * (1 + alpha)
    hpf = np.zeros(len(na_series))
    for n in range(1, len(na_series)):
        hpf[n] = a_ * (na_series[n] - na_series[n - 1]) + alpha * hpf[n - 1]
    si = pd.rolling_sum(super_smoother(hpf, smoothing), 3)
    return si / np.abs(si)


def band_pass_filter(na_series, period):
    delta = 0.3                                          # Bandwidth
    gamma = np.cos(2 * pi * delta / period)              # gamma1^-1 (ok)
    sigma = 1.0 / gamma - (1.0 / gamma ** 2 - 1) ** 0.5  # alpha1    (ok)
    lambda_ = np.cos(2 * pi / period)                    # beta1     (ok)
    
    n1 = 0.5 * (1 - sigma)
    d1 = lambda_ * (1 + sigma)
    bpf = np.zeros(len(na_series))
    for n in range(2, len(na_series)):
        bpf[n] = n1 * (na_series[n] - na_series[n - 2]) - \
                 d1 * bpf[n - 1] + sigma * bpf[n - 2]
    return bpf
