import os
import torch
import argparse
import numpy as np
from tqdm import tqdm


from adapt import get_method
from histopathc.datasets import prepare_histopathc_data
from utils.misc import load_clip_model, set_global_seeds, save_configuration


def argparser():
    parser = argparse.ArgumentParser("Histopath-C Benchmark")

    # CLIP library settings
    parser.add_argument('--clip_type', type=str, default='open_clip', choices=("open_clip", "clip", "conch"), 
                        help='Type of CLIP model to use (open_clip is quilt)')

    # Directories
    parser.add_argument('--save_dir', type=str, default='save/', 
                        help='Path for saving base training weights and results')

    # General settings
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')

    # Model
    parser.add_argument('--backbone', type=str, default='hf-hub:wisdomik/QuiltNet-B-32', 
                        help='Model backbone to use')

    # Dataset settings
    parser.add_argument('--dataset', type=str, default='nct', choices=('nct', 'skin', 'renal', 'lc25_all', 'lc25_colon', 'lc25_lung','mhist'), 
                        help='Dataset to use')
    parser.add_argument('--data_dir', type=str, default='/path/to/data/', 
                        help='Root directory for datasets')
    parser.add_argument('--workers', type=int, default=0, 
                        help='Number of workers for data loading')
    parser.add_argument('--temp_dir', type=str, default=None, 
                        help='Directory for loading templates')
    # Corruptions
    parser.add_argument('--corruptions_list', nargs='+', default=None, type=str, 
                        help='List of corruptions to apply to the dataset (Cifar datasets)')

    # Method name
    parser.add_argument('--method', type=str, default='tent', choices=('tent', 'tpt', 'lame', 'clipartt', 'latte'),
                                                                        help='Method to use for adaptation')

    # Adaptation settings
    parser.add_argument('--adapt', action='store_true', 
                        help='Enable adaptation')
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.0001, 
                        help='Learning rate')
    parser.add_argument('--steps', default=10, type=int, 
                        help='Number of iterations for adaptation')
    parser.add_argument('--trials', default=3, type=int, 
                        help='Number of trials to repeat the experiments')


    return parser


def add_method_specific_args(parser, method):
    '''
    Add method-specific arguments to the parser
    '''
    if method == 'latte':
        parser.add_argument('--use_laplacian', action='store_true', 
                            help='Enable laplacian optimization')
        parser.add_argument('--w_reg', type=float, default=1.0, 
                            help='Laplacian regularizer weight')
        parser.add_argument('--avg_type', type=str, default='text_avg', choices=('text_avg', 'loss_avg', 'loss_avg_reference', 'text_avg_reference'), 
                            help='Type of averaging')
        parser.add_argument('--lora', action='store_true', 
                    help='To use LoRA')
        parser.add_argument('--lora_ln', action='store_true', 
                            help='To use LoRA + LN params')
        parser.add_argument('--lora_params', nargs='+', default=['q', 'k', 'v'], type=str, 
                            help='List of LoRA parameters to use')
        parser.add_argument('--lora_rank', default=2, type=int, 
                            help='rank in LoRA')
        parser.add_argument('--lora_alpha', default=1, type=int, 
                            help='alpha for LoRA')
        parser.add_argument('--lora_dropout', type=float, default=0.25, 
                            help='Droput rate for LoRA')
        parser.add_argument('--lora_layers', nargs='+', default=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), type=int, 
                            help='The indices of the vision layers to use.')


    elif method == 'tent':
        parser.add_argument('--use_laplacian', action='store_true', 
                            help='Enable laplacian optimization')
        parser.add_argument('--w_reg', type=float, default=1.0, 
                            help='Laplacian regularizer weight')
        parser.add_argument('--avg_type', type=str, default='text_avg', choices=('text_avg', 'loss_avg', 'loss_avg_reference', 'text_avg_reference'), 
                            help='Type of averaging')
    
    elif method == 'lame':
        parser.add_argument('--affinity', type=str, default='linear', choices=('linear', 'knn', 'rbf'),
                             help='Affinity matrix for Laplacian')
    
    elif method == 'clipartt':
        parser.add_argument('--K', default=3, type=int, 
                            help='Number of classes taken to build the pseudo label')
    
    elif method == 'tpt':
        parser.add_argument('--n_prompts', default=1, type=int, 
                            help='Number of prompts to tune')
        parser.add_argument('--prompt_length', default=4, type=int, 
                            help='Prompt length')
    
    elif method == 'watt':
        parser.add_argument('--type', type=str, default='sequential', choices=('parallel', 'sequential'), 
                            help='Type of WATT adaptation (parallel or sequential)')
        parser.add_argument('--l', default=2, type=int, 
                            help='Number of adaptation iterations for each text embedding before weight averaging')
        parser.add_argument('--m', default=5, type=int, 
                            help='Number of repetitions of the adaptation and weight averaging process')
        
    # Add other methods here
    else:    
        raise ValueError(f"Unknown method: {method}")
    
    return parser


def main():
    # Initial argument parsing to get the method
    initial_parser = argparser()
    initial_args, _ = initial_parser.parse_known_args()

    # Create a new parser with method-specific arguments
    parser = argparser()
    parser = add_method_specific_args(parser, initial_args.method)
    args = parser.parse_args()

    # Set the global random seed for reproducibility
    set_global_seeds(args.seed)

    # Save the configuration settings
    save_configuration(args)

    # Set the device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prepare the results file
    results_path = os.path.join(args.save_dir, "results.txt")

    # Iterate over the corruptions
    for corruption in args.corruptions_list:
        # Prepare the model, tokenizer, and transforms
        model, tokenizer, original_transforms = load_clip_model(args.clip_type, args.backbone, device)

        # Prepare the data loader
        data_loader, classes = prepare_histopathc_data(args.dataset, args.data_dir, corruption=corruption, 
                                                     original_transforms=original_transforms, batch_size=args.batch_size, 
                                                     num_workers=args.workers)

        # Setting up the method
        args.classes = classes
        adapt_method = get_method(model, tokenizer, args)

        acc = []
        loss_seed_report = []
        for t in range(args.trials):
            correct = 0
            loss_batch_report = []
            for batch_idx, (inputs, labels) in tqdm(enumerate(data_loader), total=len(data_loader)):
                # Move data to the device
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                # reset the model before adapting to a new batch
                adapt_method.reset()
                
                # perform adaptation
                if args.adapt:
                    if args.method != 'lame':
                        loss_iter_report = adapt_method.adapt(inputs)
                        loss_batch_report.append(loss_iter_report)
                    else:
                        pred = adapt_method.adapt(inputs)
                        loss_iter_report = 0.0
                        loss_batch_report.append(loss_iter_report)

                # perform evaluation
                if args.method != 'lame':
                    pred = adapt_method.evaluate(inputs)

                # Calculate the number of correct predictions
                correctness = pred.eq(labels.view(1, -1).expand_as(pred))
                correct += correctness.sum().item()
                # print(correct)

            # Convert the batch report to a numpy array for easier averaging
            loss_batch_report = np.array(loss_batch_report)

            # Average loss over batches for each iteration
            avg_loss_per_iter = np.mean(loss_batch_report, axis=0)  # Shape: [10] (for 10 iterations)
            loss_seed_report.append(avg_loss_per_iter)

            acc.append(correct / len(data_loader.dataset))
            print(correct / len(data_loader.dataset))

        # Convert the seed report to a numpy array and average over trials (seeds)
        loss_seed_report = np.array(loss_seed_report)
        avg_loss_over_seeds = np.mean(loss_seed_report, axis=0)  


        print(f"{corruption}: " + str(round(np.array(acc).mean()*100, 2)) + ' +/- ' + str(round(np.array(acc).std()*100, 2)) + '\n')
        with open(results_path, 'a+') as fichier:
            fichier.write(f"{corruption}," + str(round(np.array(acc).mean()*100, 2)) + ' +/- ' + str(round(np.array(acc).std()*100, 2)) + '\n')

if __name__ == "__main__":
    main()
