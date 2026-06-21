#!/usr/bin/env python
import argparse
from FLAlgorithms.servers.serveravg import FedAvg
from FLAlgorithms.servers.serverFedProx import FedProx
from FLAlgorithms.servers.serverFedDistill import FedDistill
from FLAlgorithms.servers.serverpFedGen import FedGen
from FLAlgorithms.servers.serverpFedEnsemble import FedEnsemble
from utils.model_utils import create_model
from utils.plot_utils import *
import torch
from multiprocessing import Pool

def create_server_n_user(args, i):
    model = create_model(args.model, args.dataset, args.algorithm)
    if ('FedAvg' in args.algorithm):
        server=FedAvg(args, model, i)
    elif 'FedGen' in args.algorithm:
        server=FedGen(args, model, i)
    elif ('FedProx' in args.algorithm):
        server = FedProx(args, model, i)
    elif ('FedDistill' in args.algorithm):
        server = FedDistill(args, model, i)
    elif ('FedEnsemble' in args.algorithm):
        server = FedEnsemble(args, model, i)
    else:
        print("Algorithm {} has not been implemented.".format(args.algorithm))
        exit()
    return server


def run_job(args, i):
    torch.manual_seed(i)
    print("\n\n         [ Start training iteration {} ]           \n\n".format(i))
    # Generate model
    server = create_server_n_user(args, i)
    if args.train:
        server.train(args)
        server.test()

def main(args):
    for i in range(args.seed_start, args.seed_start + args.times):
        run_job(args, i)
    print("Finished training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Mnist-alpha0.1-ratio0.5")
    parser.add_argument("--model", type=str, default="cnn")
    parser.add_argument("--train", type=int, default=1, choices=[0,1])
    parser.add_argument("--algorithm", type=str, default="pFedMe")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gen_batch_size", type=int, default=32, help='number of samples from generator')
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Local learning rate")
    parser.add_argument("--personal_learning_rate", type=float, default=0.01, help="Personalized learning rate to caculate theta aproximately using K steps")
    parser.add_argument("--ensemble_lr", type=float, default=1e-4, help="Ensemble learning rate.")
    parser.add_argument("--beta", type=float, default=1.0, help="Average moving parameter for pFedMe, or Second learning rate of Per-FedAvg")
    parser.add_argument("--lamda", type=int, default=1, help="Regularization term")
    parser.add_argument("--mix_lambda", type=float, default=0.1, help="Mix lambda for FedMXI baseline")
    parser.add_argument("--embedding", type=int, default=0, help="Use embedding layer in generator network")
    parser.add_argument("--num_glob_iters", type=int, default=200)
    parser.add_argument("--local_epochs", type=int, default=20)
    parser.add_argument("--num_users", type=int, default=20, help="Number of Users per round")
    parser.add_argument("--K", type=int, default=1, help="Computation steps")
    parser.add_argument("--times", type=int, default=3, help="running time")
    parser.add_argument("--seed_start", type=int, default=0,
                        help="Index of the first seed to run (default 0). "
                             "Combined with --times: runs seeds "
                             "[seed_start, seed_start + times). "
                             "Use this to drive individual seeds in "
                             "resumable sweeps (e.g. seed_start=2 times=1 "
                             "runs only seed 2 without redoing seeds 0/1).")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu","cuda"], help="run device (cpu | cuda)")
    parser.add_argument("--result_path", type=str, default="results/models", help="directory path to save results")
    parser.add_argument("--missing_rate", type=float, default=0.1, help="Missing data ratio for AMDAE imputation (0.0-1.0)")
    parser.add_argument("--missing_pattern", type=str, default="random",
                        choices=["random", "mcar", "mar", "mnar", "fixed_intervals", "continuous_periods"],
                        help="Missing-data mechanism used by MissingDataSimulator: "
                             "random/mcar (default, paper Eq.1), mar (label-conditional), "
                             "mnar (magnitude-conditional), fixed_intervals, continuous_periods.")
    parser.add_argument("--force_imputer", type=str, default=None,
                        choices=["amdae", "mean", "median", "zero", "none", None],
                        help="Bypass composite-score selection and force a specific imputer. "
                             "'amdae' guarantees AM-DAE imputed data (use for the headline runs); "
                             "'mean'/'median'/'zero' force the corresponding baseline imputer; "
                             "'none' simulates missingness but skips imputation entirely "
                             "(missing positions stay zero; for the no-imputation ablation row). "
                             "Default None lets the patched composite pick the winner "
                             "(typically AM-DAE under RELIABLE_METRICS).")

    args = parser.parse_args()
    """
    # check if cuda is available
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = 'cpu'
        print("CUDA is not available. Using CPU.")
    """
    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm: {}".format(args.algorithm))
    print("Batch size: {}".format(args.batch_size))
    print("Learing rate       : {}".format(args.learning_rate))
    print("Ensemble learing rate       : {}".format(args.ensemble_lr))
    print("Average Moving       : {}".format(args.beta))
    print("Subset of users      : {}".format(args.num_users))
    print("Number of global rounds       : {}".format(args.num_glob_iters))
    print("Number of local rounds       : {}".format(args.local_epochs))
    print("Dataset       : {}".format(args.dataset))
    print("Local Model       : {}".format(args.model))
    print("Device            : {}".format(args.device))
    print("Missing rate      : {}".format(args.missing_rate))
    print("Missing pattern   : {}".format(args.missing_pattern))
    print("Force imputer     : {}".format(args.force_imputer))
    print("=" * 80)
    main(args)
