# -*- coding: utf-8 -*-


import shap
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def _clean_dir(path):
    for f in glob.glob(os.path.join(path, '*')):
        os.remove(f)


def data_anomal_filter(data, shap_values, percentile):
    data = np.array(data)
    percent = np.percentile(data, percentile, axis=0)
    data_filtered_index = [index for index in range(data.shape[0]) if np.sum(data[index] > percent) == 0]
    data_filtered = data[data_filtered_index]
    shap_values_filtered = shap_values[data_filtered_index]
    return data_filtered, shap_values_filtered


def data_extremum(data, shap_values, feature_names):
    extremum_dict = {}
    for i in range(len(feature_names)):
        feature_max = max(data[:, i])
        feature_min = min(data[:, i])
        shap_max = max(shap_values[:, i])
        shap_min = min(shap_values[:, i])
        extremum_dict[feature_names[i]] = [feature_min-0.1, feature_max+0.1, shap_min-0.01, shap_max+0.01]
    return extremum_dict


def dependence_plot(data, shap_values, feature_inds, feature_names, extremum_dict, save_path):
    _clean_dir(save_path)
    feature_inds = list(feature_inds)
    feature_inds.reverse()
    plt.style.use('ggplot')
    plt.rcParams['figure.figsize'] = (7, 4)
    n_samples = data.shape[0]

    for ind in feature_inds:
        feature = feature_names[ind]
        x = data[:, ind]
        y = shap_values[:, ind]

        n_unique_x = len(np.unique(x))
        n_unique_y = len(np.unique(y))
        if n_unique_x <= 1 or n_unique_y <= 1:
            continue

        bins_x = n_unique_x if n_unique_x <= 30 else min(50, max(10, int(np.sqrt(n_samples))))
        bins_y = n_unique_y if n_unique_y <= 30 else min(50, max(10, int(np.sqrt(n_samples))))

        x_range = [extremum_dict[feature][0], extremum_dict[feature][1]]
        y_range = [extremum_dict[feature][2], extremum_dict[feature][3]]

        counts, _, _ = np.histogram2d(x, y, bins=[bins_x, bins_y], range=[x_range, y_range])
        vmax = max(counts.max(), 2)

        plt.hist2d(x, y, bins=[bins_x, bins_y], range=[x_range, y_range],
                   norm=LogNorm(vmin=1, vmax=vmax), cmap='hot_r')
        plt.colorbar()
        plt.title(feature)
        plt.xlabel('Feature value')
        plt.ylabel('SHAP value')

        plt.axis([x_range[0], x_range[1], y_range[0], y_range[1]])
        plt.grid()
        plt.savefig(save_path+'_'+feature+'.png', dpi=500)
        plt.close('all')


def force_plot(expected_value, shap_values, data, save_path, max_samples=20):
    _clean_dir(save_path)
    n = min(shap_values.shape[0], max_samples)
    feature_names = list(data.columns)
    for i in range(len(feature_names)):
        if isinstance(data[feature_names[i]].iloc[0], float):
            data[feature_names[i]] = data[feature_names[i]].round(decimals=2)
    for i in range(n):
        shap.force_plot(expected_value, shap_values[i, :], data.iloc[i, :], matplotlib=True, show=False)
        plt.savefig(save_path+str(i)+'.png', dpi=200, bbox_inches='tight')
        plt.close('all')


def group_plot(expected_value, shap_values, data, save_path):
    np.random.seed(1234)
    p = np.random.permutation(data.shape[0])[:2000]
    shap_values = shap_values[p]
    data = data.iloc[p]
    feature_names = list(data.columns)
    for i in range(len(feature_names)):
        if isinstance(data[feature_names[i]].iloc[0], float):
            data[feature_names[i]] = data[feature_names[i]].round(decimals=2)
    group = shap.force_plot(expected_value, shap_values, data)
    shap.save_html(save_path, group)



