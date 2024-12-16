import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Preprocessing import  cheby
from Windowing import blackman_window
from Features import *
from sklearn.model_selection import train_test_split
import pickle
from Boruta import select_features_boruta
from PSO import pso_feature_optimizer
from RNN import *

def main(acc, gyr, mag, mic, gps, act_classes, loc_classes, labels):
    filtered_acc = cheby(input_signal=acc, fs=0.001)
    filtered_gyr = cheby(input_signal=gyr, fs=0.001)
    filtered_mag = cheby(input_signal=mag, fs=0.001)
    filtered_mic = cheby(input_signal=mic, fs=0.001)
    filtered_gps = cheby(input_signal=gps, fs=0.001)

    windowed_stacked_acc = blackman_window(filtered_acc)
    windowed_stacked_gyr = blackman_window(filtered_gyr)
    windowed_stacked_mag = blackman_window(filtered_mag)
    windowed_stacked_mic = blackman_window(filtered_mic)
    windowed_stacked_gps = blackman_window(filtered_gps)

    skew_acc = calculate_skewness(windowed_stacked_acc)
    skew_gyr = calculate_skewness(windowed_stacked_gyr)
    skew_mag = calculate_skewness(windowed_stacked_mag)

    kurt_acc = calculate_kurtosis(windowed_stacked_acc)
    kurt_gyr = calculate_kurtosis(windowed_stacked_gyr)
    kurt_mag = calculate_kurtosis(windowed_stacked_mag)

    cm1_acc = comulant_1(windowed_stacked_acc)
    cm1_gyr = comulant_1(windowed_stacked_gyr)
    cm1_mag = comulant_1(windowed_stacked_mag)
    cm1_gps = comulant_1(windowed_stacked_gps)

    cm2_acc = comulant_2(windowed_stacked_acc)
    cm2_gyr = comulant_2(windowed_stacked_gyr)
    cm2_mag = comulant_2(windowed_stacked_mag)
    cm2_gps = comulant_2(windowed_stacked_gps)

    cm3_acc = comulant_3(windowed_stacked_acc)
    cm3_gyr = comulant_3(windowed_stacked_gyr)
    cm3_mag = comulant_3(windowed_stacked_mag)
    cm3_gps = comulant_3(windowed_stacked_gps)

    par_acc = parseval(windowed_stacked_acc)
    par_gyr = parseval(windowed_stacked_gyr)
    par_mag = parseval(windowed_stacked_mag)

    sf_acc = spec_flatness(windowed_stacked_acc)
    sf_gyr = spec_flatness(windowed_stacked_gyr)
    sf_mag = spec_flatness(windowed_stacked_mag)

    harm_acc = harmonics(windowed_stacked_acc)
    harm_gyr = harmonics(windowed_stacked_gyr)
    harm_mag = harmonics(windowed_stacked_mag)

    speed_acc = speed(windowed_stacked_acc)

    heading_dir = heading_direction(acc[0], acc[1], acc[2],
                                    mag[0], mag[1], mag[2],
                                    gyr[0], gyr[1], gyr[2]
                                    )

    stay_dur = stay_duration(speed_acc)

    pit_acc = pitch(windowed_stacked_mic)
    pit_gyr = pitch(windowed_stacked_mic)
    pit_mag = pitch(windowed_stacked_mic)

    harm_ratio_acc = harmonics_ratio(x=windowed_stacked_acc)
    harm_ratio_gyr = harmonics_ratio(x=windowed_stacked_gyr)
    harm_ratio_mag = harmonics_ratio(x=windowed_stacked_mag)

    spec_flux_acc = spectral_flux(windowed_stacked_acc)
    spec_flux_gyr = spectral_flux(windowed_stacked_gyr)
    spec_flux_mag = spectral_flux(windowed_stacked_mag)

    act_feats = np.concatenate((skew_acc,
                                skew_gyr,
                                skew_mag,
                                kurt_acc,
                                kurt_gyr,
                                kurt_mag,
                                cm1_acc,
                                cm1_gyr,
                                cm1_mag,
                                cm1_gps,
                                cm2_acc,
                                cm2_gyr,
                                cm2_mag,
                                cm2_gps,
                                cm3_acc,
                                cm3_gyr,
                                cm3_mag,
                                cm3_gps,
                                par_acc,
                                par_gyr,
                                par_mag,
                                sf_acc,
                                sf_gyr,
                                sf_mag,
                                harm_acc,
                                harm_gyr,
                                harm_mag
                                ))
    
    loc_feats = np.concatenate((speed_acc,
                                heading_dir,
                                stay_dur,
                                pit_acc,
                                pit_gyr,
                                pit_mag,
                                harm_ratio_acc,
                                harm_ratio_gyr,
                                harm_ratio_mag,
                                spec_flux_acc,
                                spec_flux_gyr,
                                spec_flux_mag
                                ))

    selected_features_act = select_features_boruta(X=act_feats, y=labels)
    selected_features_loc = select_features_boruta(X=loc_feats, y=labels)

    augmented_data_act = pso_feature_optimizer(selected_features_act)
    augmented_data_loc = pso_feature_optimizer(selected_features_loc)

    input_shape_act = augmented_data_act.shape[1:]
    input_shape_loc = augmented_data_loc.shape[1:]

    num_classes_act = input_shape_act
    num_classes_loc = input_shape_loc

    rnn_model_act = train_rnn_activity_classification(X=num_classes_act, y=labels)
    rnn_model_loc = train_rnn_location_classification(X=num_classes_loc, y=labels)

    pickle.dump(rnn_model_act, 'activity.pkl')
    pickle.dump(rnn_model_loc, 'location.pkl')




if __name__ == "__main__":
    # Read data files and run main
    main()
