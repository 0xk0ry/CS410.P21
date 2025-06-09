import os
import sys
import argparse
from utils import plot_metrics

def find_csv_files(root_dir):
    """Find all CSV metric files in the given directory and subdirectories."""
    csv_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('_metrics.csv') or file.endswith('_metrics_zero.csv') or file.endswith('_metrics_random.csv'):
                csv_path = os.path.join(root, file)
                csv_files.append(csv_path)
    return csv_files

def get_model_name(csv_path):
    """Extract model name from CSV path."""
    # Get filename without extension
    filename = os.path.basename(csv_path)
    parts = filename.split('_metrics')
    model_name = parts[0].replace('train_', '')
    
    # Add additional info from filename if available
    if '_zero' in filename:
        model_name += '_zero'
    elif '_random' in filename:
        model_name += '_random'
    
    # Get dataset info from path
    if 'MNIST' in csv_path:
        model_name = 'mnist_' + model_name
    elif 'CIFAR10' in csv_path:
        model_name = 'cifar10_' + model_name
    
    return model_name

def main():
    parser = argparse.ArgumentParser(description='Generate plots from training metric CSV files.')
    parser.add_argument('--datasets', nargs='+', default=['MNIST', 'CIFAR10'], 
                        help='Datasets to process (MNIST, CIFAR10)')
    parser.add_argument('--models', nargs='+', default=['fgsm', 'fgsm_rs', 'pgd', 'free'],
                        help='Models to process (fgsm, fgsm_rs, pgd, free)')
    parser.add_argument('--exp_dir', type=str, default='../exp',
                        help='Root directory containing experiment results')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save plots (default: same as CSV file location)')
    
    args = parser.parse_args()
    
    # Convert relative path to absolute path if necessary
    if not os.path.isabs(args.exp_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.exp_dir = os.path.join(script_dir, args.exp_dir)
    
    # Process each dataset
    for dataset in args.datasets:
        dataset_dir = os.path.join(args.exp_dir, dataset)
        if not os.path.exists(dataset_dir):
            print(f"Warning: Dataset directory not found: {dataset_dir}")
            continue
        
        # Process each model type
        for model_type in args.models:
            model_dir = os.path.join(dataset_dir, model_type)
            if not os.path.exists(model_dir):
                print(f"Warning: Model directory not found: {model_dir}")
                continue
            
            # Find CSV files
            csv_files = find_csv_files(model_dir)
            if not csv_files:
                print(f"No CSV metric files found in {model_dir}")
                continue
            
            print(f"Processing {len(csv_files)} CSV files in {model_dir}...")
            
            # Generate plots for each CSV file
            for csv_path in csv_files:
                model_name = get_model_name(csv_path)
                output_dir = args.output_dir if args.output_dir else os.path.dirname(csv_path)
                
                print(f"Generating plots for {model_name} from {csv_path}")
                try:
                    plot_metrics(csv_path, model_name, output_dir)
                    print(f"Plots saved to {output_dir}")
                except Exception as e:
                    print(f"Error generating plots for {model_name}: {str(e)}")

    print("All plots generated successfully!")

if __name__ == "__main__":
    main()