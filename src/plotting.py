import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from synth import Synth


PLOT_DIR_PATH = "../plots"
# To up-scale plots so they are high quality when made larger
SCALER = 1.5
COLOURS = ["#e6194B","#3cb44b", "#ffe119", "#4363d8", "#f58231", "#42d4f4", "#f032e6", "#fabebe", "#469990", "#e6beff", "#9A6324", "#800000", "#aaffc3", "#000075", "#a9a9a9"]

def fitness_graph(path, mode='best', show=True, save=False, scaler=SCALER):
    if save:
        if not os.path.exists(PLOT_DIR_PATH):
            os.makedirs(PLOT_DIR_PATH)

    # Load data and variables for saving
    data = json.load(open(path))
    gene_representation = data["gene"]
    mut_prob = data["mutation-prob"]
    cross_prob = data["crossover-prob"]

    # Plot for all targets
    _set_font_size(10*scaler)
    plt.figure(figsize=(6*scaler,4*scaler))
    plt.grid(zorder=1)
    for i in range(5):
        target = data["targets"][i]

        # Accumulate best fitness values for every run per target
        run = target["runs"][0]
        fits = []
        for generation in run["gen_stats"]:
            fits.append(generation[mode])

        # Actually plot
        c = COLOURS[i-1]
        plt.plot(fits, label=f"Target {i+1}", zorder=2, color=c, linestyle="solid", linewidth=3*scaler)
        if run["early_stopping"]:
            plt.plot(len(fits)-1, 5, 'bo', color=c, markersize=5*scaler, zorder=3)
    plt.ylabel("Error (MSE)")
    #plt.ylim((0,500))
    plt.xlabel("Generations")
    #plt.title(f"Fitness development for target {t}\n({gene_representation}, cp={cross_prob}, mp={mut_prob})", fontsize=12*scaler)
    plt.legend(loc='upper right')
    if save:
        plt.savefig(os.path.join(PLOT_DIR_PATH, f"{os.path.basename(path)[:-5]}-{mode}.pdf"), bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

def grouped_metric_graph(log_dir_path, show=True, save=False, scaler=SCALER):
    if save:
        if not os.path.exists(PLOT_DIR_PATH):
            os.makedirs(PLOT_DIR_PATH)

    # Collect all metrics for all files
    gene_representation = ""
    metric_labels = ["mean_fitness", "proportion_of_early_stopping", "fitness_evaluations_per_run"]
    metric_titles = {
        "mean_fitness": "Mean error (MSE)",
        "proportion_of_early_stopping": "Proportion of early stopping",
        "fitness_evaluations_per_run": "Mean number of evaluations per run"
    }
    total_metrics = []
    for log in sorted(os.listdir(log_dir_path)):
        log_metrics = []
        if log == ".DS_Store":
            continue
        print(f"Opening {os.path.join(log_dir_path, log)}")
        data = json.load(open(os.path.join(log_dir_path, log)))
        gene_representation = data["gene"]
        log_metrics = [data["crossover-prob"], data["mutation-prob"]]

        # Get mean value per metric
        for metric in metric_labels:
            log_metrics.append(data["metrics"][metric])

        # Gather metrics per target
        metric_vals_per_target = { "mean_fitness": [], "proportion_of_early_stopping": [], "fitness_evaluations_per_run": [] }
        for target in data["targets"]:
            for metric in metric_labels:
                metric_vals_per_target[metric].append(target["metrics"][metric])
        # Compute std value pet metric
        for metric in metric_labels:
            log_metrics.append(np.std(metric_vals_per_target[metric]))
        total_metrics.append(log_metrics)

    # Add all metrics in a dataframe
    columns = ["cp", "mp"]
    columns.extend(metric_labels)
    columns.extend([m+"_std" for m in metric_labels])
    df = pd.DataFrame(total_metrics, columns=columns)
    # Plotting
    for i, metric in enumerate(metric_labels, start=1):
        plt.figure(figsize=(6*scaler,4*scaler))
        _set_font_size(10*scaler)

        plt.grid()
        idcs = np.arange(len(df["cp"].unique()))
        bar_width = 0.3

        # Plot bars for each mutation probability
        for i, mp in enumerate(df["mp"].unique()):
            mp_data = df[df["mp"] == mp]
            plt.bar(idcs + i*bar_width, mp_data[metric].values,
                    width=bar_width, label="mp="+str(mp), color=COLOURS[i],zorder=2)
            plt.errorbar(idcs + i*bar_width, mp_data[metric].values, mp_data[metric+"_std"].values, color="black", fmt="o",
                         capsize=4*scaler, markersize=5*scaler, linewidth=0.8*scaler, zorder=3)

        # Ticks for crossover probability
        plt.xticks(idcs+bar_width, labels=["cp="+str(cp) for cp in df["cp"].unique()])
        plt.legend(loc='upper right')
        plt.ylabel(metric_titles[metric])
        plt.ylim(ymin=0)
        plt.tight_layout()
        #plt.title(f"{metric_text} for {gene_representation} gene", fontsize=12*scaler)
        if save:
            plt.savefig(os.path.join(PLOT_DIR_PATH, f"{gene_representation}_{metric}_grouped.pdf"), bbox_inches="tight")
        if show:
            plt.show()
        plt.close()

def metric_graph(log_dir_path, show=True, save=False, scaler=SCALER):
    if save:
        if not os.path.exists(PLOT_DIR_PATH):
            os.makedirs(PLOT_DIR_PATH)

    gene_representation = ""
    xticklabels = []
    metrics_mean = { "mean_fitness": [], "proportion_of_early_stopping": [], "fitness_evaluations_per_run": [] }
    metrics_std  = { "mean_fitness": [], "proportion_of_early_stopping": [], "fitness_evaluations_per_run": [] }
    for log in sorted(os.listdir(log_dir_path)):
        if log == ".DS_Store":
            continue
        print(f"Opening {os.path.join(log_dir_path, log)}")
        data = json.load(open(os.path.join(log_dir_path, log)))
        gene_representation = data["gene"]
        xticklabels.append(f"cp={data['crossover-prob']}\nmp={data['mutation-prob']}")

        # Get mean value per metric
        for metric in data["metrics"]:
            metrics_mean[metric].append(data["metrics"][metric])

        # Gather metrics per target
        metric_vals_per_target = { "mean_fitness": [], "proportion_of_early_stopping": [], "fitness_evaluations_per_run": [] }
        for target in data["targets"]:
            for metric in target["metrics"]:
                metric_vals_per_target[metric].append(target["metrics"][metric])
        # Compute std value pet metric
        for metric in metric_vals_per_target.keys():
            metrics_std[metric].append(np.std(metric_vals_per_target[metric]))

    # Plotting
    for metric in metrics_mean.keys():
        _set_font_size(10*scaler)
        plt.figure(figsize=(10*scaler,5*scaler))
        plt.grid(zorder=1)
        idcs = np.arange(1, len(metrics_mean[metric])+1)
        plt.bar(idcs, metrics_mean[metric], color=COLOURS[:len(idcs)], zorder=2)
        plt.errorbar(idcs, metrics_mean[metric], metrics_std[metric], color="black", fmt="o",
                     capsize=4*scaler, markersize=5*scaler, linewidth=0.8*scaler, zorder=3)
        plt.xlabel("Probabilities")
        plt.xticks(ticks=idcs, labels=xticklabels)
        metric_text = metric.replace("_"," ").capitalize()
        plt.ylabel(metric_text)
        plt.title(f"{metric_text} for {gene_representation} gene", fontsize=12*scaler)
        if save:
            plt.savefig(os.path.join(PLOT_DIR_PATH, f"{gene_representation}_{metric}"), bbox_inches="tight")
        if show:
            plt.show()
        plt.close()

def final_metric_graph(log_dir_path, show=True, save=False, scaler=SCALER):
    if save:
        if not os.path.exists(PLOT_DIR_PATH):
            os.makedirs(PLOT_DIR_PATH)

    xticklabels = []
    metrics_mean = { "mean_fitness": [], "proportion_of_early_stopping": [], "fitness_evaluations_per_run": [] }
    metrics_std  = { "mean_fitness": [], "proportion_of_early_stopping": [], "fitness_evaluations_per_run": [] }
    metric_titles = {
        "mean_fitness": "Mean error (MSE)",
        "proportion_of_early_stopping": "Proportion of early stopping",
        "fitness_evaluations_per_run": "Mean number of evaluations per run"
    }
    for log in ['binary.json', 'categorical.json']:
        if log == ".DS_Store":
            continue
        print(f"Opening {os.path.join(log_dir_path, log)}")
        data = json.load(open(os.path.join(log_dir_path, log)))
        gene_representation = data["gene"]
        xticklabels.append(data["gene"].capitalize())

        # Get mean value per metric
        for metric in data["metrics"]:
            metrics_mean[metric].append(data["metrics"][metric])

        # Gather metrics per target
        metric_vals_per_target = { "mean_fitness": [], "proportion_of_early_stopping": [], "fitness_evaluations_per_run": [] }
        for target in data["targets"]:
            for metric in target["metrics"]:
                metric_vals_per_target[metric].append(target["metrics"][metric])
        # Compute std value pet metric
        for metric in metric_vals_per_target.keys():
            metrics_std[metric].append(np.std(metric_vals_per_target[metric]))

    # Plotting
    for metric in metrics_mean.keys():
        _set_font_size(10*scaler*1.3)
        plt.figure(figsize=(6*scaler,5*scaler))
        plt.grid(zorder=1)
        idcs = np.arange(1, len(metrics_mean[metric])+1)
        plt.bar(idcs, metrics_mean[metric], color=COLOURS[:len(idcs)], zorder=2, width=0.7)
        plt.errorbar(idcs, metrics_mean[metric], metrics_std[metric], color="black", fmt="o",
                     capsize=4*scaler, markersize=5*scaler, linewidth=0.8*scaler, zorder=3)
        plt.xlabel("Encoding")
        plt.xticks(ticks=idcs, labels=xticklabels)
        plt.ylim(ymin=0)
        plt.ylabel(metric_titles[metric])
        #plt.title(f"{metric_text} for each gene representation", fontsize=12*scaler)
        if save:
            plt.savefig(os.path.join(PLOT_DIR_PATH, f"final_{metric}.pdf"), bbox_inches="tight")
        if show:
            plt.show()
        plt.close()

def shape_plot(save=True, show=False):
    mys = Synth()
    shapes = ['Sine', 'Triangle', 'Square', 'SquareH', 'Sawtooth', 'Pulse', 'Semicircle', 'WhiteNoise']
    sa = []
    for s in shapes:
        mys.set_parameters(osc_2=s, amp_1=0, amp_2=1)
        sa.append(mys.get_sound_array())

    plt.figure(figsize=(5, 6))
    for i, s in enumerate(shapes):
        plt.subplot(4, 2, i+1)
        if s == 'WhiteNoise':
            plt.plot(sa[i], linewidth=2)
        else:
            plt.plot(sa[i][:250], linewidth=4)
        plt.yticks(ticks=[-1, 1], labels=['-1', '1'])
        plt.tick_params(axis='x',          # changes apply to the x-axis
                        which='both',      # both major and minor ticks are affected
                        bottom=False,      # ticks along the bottom edge are off
                        top=False,         # ticks along the top edge are off
                        labelbottom=False) # labels along the bottom edge are off
        plt.title(s, fontsize=15)

    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(PLOT_DIR_PATH, f"shapes.pdf"), bbox_inches="tight")
    if show:
        plt.show()

def _set_font_size(font_size):
    plt.rc('font', size=font_size)          # controls default text sizes
    plt.rc('axes', titlesize=font_size*1.3)     # fontsize of the axes title
    plt.rc('axes', labelsize=font_size*1.3)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=font_size)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=font_size)    # fontsize of the tick labels
    plt.rc('legend', fontsize=font_size)    # legend fontsize

if __name__ == "__main__":
    test_path = "../logs/hyperparameter-tuning/binary"
    #metric_graph(test_path, show=True, save=True, scaler=SCALER)
    grouped_metric_graph("../logs/hyperparameter-tuning/binary", show=False, save=True, scaler=SCALER)
    grouped_metric_graph("../logs/hyperparameter-tuning/categorical", show=False, save=True, scaler=SCALER)
    final_metric_graph("../logs/main-comparison", show=False, save=True, scaler=SCALER)

    # CODE BELOW GENERATES PLOTS FOR EVERY TARGET IN ONE LOG FILE (so e.g. 20 for hyperparameter-tuning)
    # file_ = os.listdir(test_path)[0]
    fitness_graph("../logs/hyperparameter-tuning/categorical/categorical-cp-0.5-mp-0.3.json", mode='best', save=True, show=False)
    fitness_graph("../logs/hyperparameter-tuning/categorical/categorical-cp-0.5-mp-0.3.json", mode='worst', save=True, show=False)

    # Create a plot of all synth shapes
    shape_plot(save=True, show=False)
