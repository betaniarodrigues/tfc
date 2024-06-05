import argparse

import os
import torch

dir_path = os.path.dirname(os.path.realpath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Parameters for Contrastive Predictive Coding for Human '
                    'Activity Recognition')

    # Data loading parameters
    parser.add_argument('--window', type=int, default=50, help='Window size')
    parser.add_argument('--overlap', type=int, default=25,
                        help='Overlap between consecutive windows')

    # Training settings
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--num_epochs', type=int, default=150)
    parser.add_argument('--gpu_id', type=str, default='0')

    # Dataset to train on
    parser.add_argument('--dataset', type=str, default='mobiact',
                        help='Choosing the dataset to perform the training on')

    # Conv encoder
    parser.add_argument('--kernel_size', type=int, default=3,
                        help='Size of the conv filters in the encoder')

    # Future prediction horizon
    parser.add_argument('--num_steps_prediction', type=int, default=28,
                        help='Number of steps in the future to predict')

    # ------------------------------------------------------------
    # Classification parameters
    parser.add_argument('--classifier_lr', type=float, default=5e-4,)
    parser.add_argument('--classifier_batch_size', type=int, default=256)
    parser.add_argument('--saved_model', type=str, default=None,
                        help='Full path of the learned CPC model')
    parser.add_argument('--learning_schedule', type=str, default='last_layer',
                        choices=['last_layer', 'all_layers'],
                        help='last layer freezes the encoder weights but '
                             'all_layers does not.')
    # ------------------------------------------------------------

    # Random seed for reproducibility
    parser.add_argument('--random_seed', type=int, default=42)

    parser.add_argument('--data_percentage', type=int, default=100,
                        help='Percentage of data to use for training (default: 100)')

    args = parser.parse_args()

    # Setting parameters by the dataset
    
    args.root_dir = '/workspaces/betania.silva/view_concatenated'
    args.input_size = 6

    if args.dataset == 'UCI_raw_12':
        args.data_file = 'UCI_raw_12'
        args.num_classes = 13
    elif args.dataset == 'UCI_raw':
        args.data_file = 'UCI_raw'
        args.num_classes = 7
    elif args.dataset == 'MotionSense_raw':
        args.data_file = 'MotionSense_raw'
        args.num_classes = 6
    elif args.dataset == 'KuHar_raw':
        args.data_file = 'KuHar_raw'
        args.num_classes = 18
    elif args.dataset == 'RealWorld_raw':
        args.data_file = 'RealWorld_raw'
        args.num_classes = 9
    elif args.dataset == 'KuHar_MotionSense':
        args.data_file = 'KuHar_MotionSense'
        args.num_classes = 18
    elif args.dataset == 'UCI_KuHar':
        args.data_file = 'UCI_KuHar'
        args.num_classes = 18
    elif args.dataset == 'UCI_raw_g':
        args.data_file = 'UCI_raw_g'
        args.num_classes = 7
    elif args.dataset == 'UCI_MotionSense_KuHar_RealWorld':
        args.data_file = 'UCI_MotionSense_KuHar_RealWorld'
        args.num_classes = 18
    elif args.dataset == 'KuHar_RealWorld':
        args.data_file = 'KuHar_RealWorld'
        args.num_classes = 18
    elif args.dataset == 'UCI_MotionSense':
        args.data_file = 'UCI_MotionSense'
        args.num_classes = 13
    elif args.dataset == 'UCI_RealWorld':
        args.data_file = 'UCI_RealWorld'
        args.num_classes = 9
    elif args.dataset == 'MotionSense_RealWorld':
        args.data_file = 'MotionSense_RealWorld'
        args.num_classes = 9
    elif args.dataset == 'KuHar_UCI_RealWorld':
        args.data_file = 'KuHar_UCI_RealWorld'
        args.num_classes = 18
    elif args.dataset == 'KuHar_MotionSense_RealWorld':
        args.data_file = 'KuHar_MotionSense_RealWorld'
        args.num_classes = 18
    elif args.dataset == 'UCI_MotionSense_RealWorld':
        args.data_file = 'UCI_MotionSense_RealWorld'
        args.num_classes = 13

    # if args.dataset == 'uci':
    #     args.input_size = 6
    #     args.num_classes = 6
    #     args.root_dir = '/workspaces/betania.silva/view_concatenated'
    #     args.data_file = 'UCI'

    args.device = torch.device(
        "cuda:" + str(args.gpu_id) if torch.cuda.is_available() else "cpu")
    
    # Conv padding size
    args.padding = int(args.kernel_size // 2)

    return args

 