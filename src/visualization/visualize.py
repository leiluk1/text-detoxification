import matplotlib.pyplot as plt
import seaborn as sns


def plot_losses(model='t5'):
    """
    Plot the training and validation losses for a given model.
    
    Args:
        model: Model name (default t5).
    """

    if model == 't5':
        train_losses = [3.935, 3.012, 2.740, 2.578, 2.457, 2.358, 2.274, 2.197, 2.126, 2.061]
        val_losses = [3.144, 2.821, 2.661, 2.581, 2.535, 2.494, 2.472, 2.472, 2.452, 2.451]
        epochs = [i for i in range(1, 11)]
        plt.plot(epochs, train_losses, label="Training loss", color='red')
        plt.plot(epochs, val_losses, label="Validation loss", color='darkblue')
        grid = plt.grid(True)
        plt.title('T5 model loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        sns.set(style="darkgrid")
        plt.show()
        
    if model == 'transformer':
        train_losses = [1.9855, 1.7895, 1.7645, 1.7104, 1.6939, 1.661, 1.65, 1.6383, 1.6352, 1.5962, 1.5898, 1.5895, 1.5794, 1.588,
                        1.5519, 1.5485, 1.5499, 1.5506, 1.5456, 1.5146, 1.5129, 1.5221, 1.5282, 1.5043, 1.4932, 1.497, 1.4931, 1.5016,
                        1.4845, 1.477, 1.48, 1.4841, 1.4692, 1.4665, 1.468, 1.4705, 1.4586, 1.4625, 1.4556, 1.4508, 1.461, 1.4607, 1.4433, 
                        1.4605, 1.4532, 1.444, 1.4493]
        train_epochs = [0.21, 0.42, 0.63, 0.84, 1.05, 1.26, 1.47, 1.68, 1.89, 2.11, 2.32, 2.53, 2.74, 2.95, 3.16, 3.37, 3.58, 3.79, 4. , 4.21,
                        4.42, 4.63, 4.84, 5.05, 5.26, 5.47, 5.68, 5.89, 6.11, 6.32, 6.53, 6.74, 6.95, 7.16, 7.37, 7.58, 7.79, 8., 8.21, 8.42, 
                        8.63, 8.84, 9.05, 9.26, 9.47, 9.68, 9.89]
        eval_losses = [1.58316969871521, 1.536309003829956, 1.5098899602890015, 1.495600938796997, 1.483681559562683, 1.4756686687469482, 
                       1.469192385673523, 1.4670323133468628, 1.4648141860961914, 1.4639936685562134]
        eval_epochs = [i for i in range(1, 11)]
        plt.plot(train_epochs, train_losses, label="Training loss", color='red')
        plt.plot(eval_epochs, eval_losses, label="Validation loss", color='darkblue')
        grid = plt.grid(True)
        plt.title('Transformer model loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        sns.set(style="darkgrid")
        plt.show()


def plot_eval_metrics():
    """
    Plots the evaluation metrics for the T5 model on the validation set.
    """
    eval_bleu = [26.8431, 27.6637, 28.0732, 28.2589, 28.3299, 28.5929, 28.6465, 28.6889, 28.7409, 28.7357]
    eval_rouge1 = [0.576, 0.5841, 0.5864, 0.5893, 0.5901, 0.592, 0.5921, 0.5929, 0.5926, 0.5931]
    eval_rouge2 = [0.3601, 0.3712, 0.3746, 0.3781, 0.3794, 0.3794, 0.3823, 0.3833, 0.3831, 0.3833]
    epochs = [i for i in range(1, 11)]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3))
    metrics = [eval_bleu, eval_rouge1, eval_rouge2]

    # Loop through the subplots and plot the data
    for i, ax in enumerate(axs):
        if i == 0:
            ax.plot(epochs, metrics[i], color='red')
            ax.set_title('BLEU score on validation set', fontsize=8)
            ax.set_xlabel('Epochs')
            ax.grid(True)
            
        elif i == 1:
            ax.plot(epochs, metrics[i], color='darkblue')
            ax.set_title('ROUGE-1 F1 score on validation set', fontsize=8)
            ax.set_xlabel('Epochs')
            ax.grid(True)
        else:
            ax.plot(epochs, metrics[i], color='green')
            ax.set_title('ROUGE-2 F1 score on validation set', fontsize=8)
            ax.set_xlabel('Epochs')
            ax.grid(True)

    plt.tight_layout()
    sns.set(style="darkgrid")
    plt.show()
    
    
def plot_metrics_hist(all_metrics, toxicity_scores, approach_type):
    """
    Plots the histograms for metric values obtained on test set.
    
     Args:
        all_metrics: List of dictionaries containig metrics for each sample.
        toxicity_scores: List of toxicity scores.
        approach_type: The approach type (baseline/transformer/t5).
    """
    
    # Plot the BLEU-4 and ROUGE-1, ROUGE-2 score distributions
    metrics = ['BLEU-4', 'rouge1_fmeasure', 'rouge2_fmeasure']
    for metric in metrics:
        metric_values = []
        for metric_dict in all_metrics:
            metric_values.append(metric_dict[metric])
        sns.set(style="darkgrid")
        plt.hist(metric_values, bins=20, color='skyblue')
        plt.title("The " + approach_type + ": " + metric + " score distribution")
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.show()
    
    # Plot the toxicity score distribution
    sns.set(style="darkgrid")
    plt.hist(toxicity_scores, bins=20, color='skyblue')
    plt.title("The " + approach_type + ": toxicity score distribution")
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.show()
