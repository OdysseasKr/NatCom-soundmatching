import json
import matplotlib.pyplot as plt
import numpy as np
import os

PLOT_DIR_PATH = "../plots"
# To up-scale plots so they are high quality when made larger
SCALER = 1.5 
COLOURS = ["#e6194B","#3cb44b", "#ffe119", "#4363d8", "#f58231", "#42d4f4", "#f032e6", "#fabebe", "#469990", "#e6beff", "#9A6324", "#800000", "#aaffc3", "#000075", "#a9a9a9"]

def best_worst_fitness_graph(path, show=True, save=False, scaler=SCALER):
    if save:
        if not os.path.exists(PLOT_DIR_PATH):
            os.makedirs(PLOT_DIR_PATH)

    # Load data and variables for saving
    data = json.load(open(path))
    gene_representation = data["gene"]
    mut_prob = data["crossover-prob"]
    cross_prob = data["mutation-prob"]

    # Plot for all targets
    for t,target in enumerate(data["targets"], start=1):
        _set_font_size(10*scaler)
        plt.figure(figsize=(7*scaler,5*scaler))
        plt.grid(zorder=1)

        # Accumulate best/worst fitness values for every run per target
        for i,run in enumerate(target["runs"], start=1):
            best_fits, worst_fits = [], []
            for generation in run["gen_stats"]:
                best_fits.append(generation["best"])
                worst_fits.append(generation["worst"])
            # Actually plot
            c = COLOURS[i-1]
            plt.plot(best_fits, label=f"Run {i}", zorder=2, color=c, linestyle="solid", linewidth=3*scaler)
            plt.plot(worst_fits, zorder=2, color=c, linestyle="dashed", linewidth=3*scaler)
            if run["early_stopping"]:
                plt.plot(len(best_fits)-1, 5, 'bo', color=c, markersize=10*scaler, zorder=3)
        plt.ylabel("Fitness (MSE)")
        plt.ylim((0,500))
        plt.xlabel("Generations")
        plt.title(f"Fitness development for target {t}\n({gene_representation}, cp={cross_prob}, mp={mut_prob})", fontsize=12*scaler)
        plt.legend()
        if save:
            plt.savefig(os.path.join(PLOT_DIR_PATH, f"{os.path.basename(path)[:-5]}-target-{t}.png"), bbox_inches="tight")
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

def _set_font_size(font_size):
    plt.rc('font', size=font_size)          # controls default text sizes
    plt.rc('axes', titlesize=font_size)     # fontsize of the axes title
    plt.rc('axes', labelsize=font_size)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=font_size)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=font_size)    # fontsize of the tick labels
    plt.rc('legend', fontsize=font_size)    # legend fontsize

if __name__ == "__main__":
    test_path = "../logs/hyperparameter-tuning/binary"
    metric_graph(test_path, show=True, save=True, scaler=SCALER)

    # CODE BELOW GENERATES PLOTS FOR EVERY TARGET IN ONE LOG FILE (so e.g. 20 for hyperparameter-tuning)
    # file_ = os.listdir(test_path)[0]
    best_worst_fitness_graph("/Users/fbergh/Documents/Radboud/master/1/NatCom/project/NatCom-soundmatching/logs/hyperparameter-tuning/categorical/categorical-mp-0.3-cp-0.7.json", True, False)