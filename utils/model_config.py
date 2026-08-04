CONFIGS_ = {
    # input_channel, n_class, hidden_dim, latent_dim
    'cifar': ([16, 'M', 32, 'M', 'F'], 3, 10, 2048, 64),
    'cifar100-c25': ([32, 'M', 64, 'M', 128, 'F'], 3, 25, 128, 128),
    'cifar100-c30': ([32, 'M', 64, 'M', 128, 'F'], 3, 30, 2048, 128),
    'cifar100-c50': ([32, 'M', 64, 'M', 128, 'F'], 3, 50, 2048, 128),

    'emnist': ([6, 16, 'F'], 1, 26, 784, 32),
    'mnist': ([6, 16, 'F'], 1, 10, 784, 32),
    'mnist_cnn1': ([6, 'M', 16, 'M', 'F'], 1, 10, 64, 32),
    'mnist_cnn2': ([16, 'M', 32, 'M', 'F'], 1, 10, 128, 32),
    'celeb': ([16, 'M', 32, 'M', 64, 'M', 'F'], 3, 2, 64, 32),
    'ucihar': ([6, 16, 'F'], 1, 6, 576, 32),  # 1-ch 24x24, Conv->12x12->6x6, flat=576
    'pamap2': ([6, 16, 'F'], 1, 12, 64, 32),  # 1-ch 8x8 (51 IMU + 13 zero pad), Conv->4x4->2x2, flat=16*2*2=64
    'wisdm':  ([6, 16, 'F'], 1, 6, 576, 32),  # 1-ch 24x24 (192 samples @ 20Hz x 3 axes = 576), same head as UCI HAR
    # Run 3 medical image slots: 3-ch 32x32 RGB, Conv(s=2)->16x16->8x8, flat=16*8*8=1024
    'fedisic':  ([6, 16, 'F'], 3, 8, 1024, 32),   # ISIC 2019, 8 lesion classes
    'ham10000': ([6, 16, 'F'], 3, 7, 1024, 32),   # ISIC 2018 Task 3, 7 lesion classes
    # CIFAR-10: 3-ch 32x32 RGB, Conv(s=2)->16x16->8x8, flat=16*8*8=1024, 10 classes
    # (kept separate from the legacy 'cifar' key whose arithmetic was inconsistent)
    'cifar10':  ([6, 16, 'F'], 3, 10, 1024, 32),
}

# temporary roundabout to evaluate sensitivity of the generator
GENERATORCONFIGS = {
    # hidden_dimension, latent_dimension, input_channel, n_class, noise_dim
    'cifar': (512, 32, 3, 10, 64),
    'celeb': (128, 32, 3, 2, 32),
    'mnist': (256, 32, 1, 10, 32),
    'mnist-cnn0': (256, 32, 1, 10, 64),
    'mnist-cnn1': (128, 32, 1, 10, 32),
    'mnist-cnn2': (64, 32, 1, 10, 32),
    'mnist-cnn3': (64, 32, 1, 10, 16),
    'emnist': (256, 32, 1, 26, 32),
    'emnist-cnn0': (256, 32, 1, 26, 64),
    'emnist-cnn1': (128, 32, 1, 26, 32),
    'emnist-cnn2': (128, 32, 1, 26, 16),
    'emnist-cnn3': (64, 32, 1, 26, 32),
    'ucihar': (256, 32, 1, 6, 32),
    'pamap2': (256, 32, 1, 12, 32),
    'wisdm':  (256, 32, 1, 6, 32),
    'fedisic':  (512, 32, 3, 8, 64),   # RGB generator, 8 classes
    'ham10000': (512, 32, 3, 7, 64),   # RGB generator, 7 classes
    'cifar10':  (512, 32, 3, 10, 64),  # RGB generator, 10 CIFAR-10 classes
}



RUNCONFIGS = {
    'emnist':
        {
            'ensemble_lr': 1e-4,
            'ensemble_batch_size': 128,
            'ensemble_epochs': 50,
            'num_pretrain_iters': 20,
            'ensemble_alpha': 1,  # teacher loss (server side)
            'ensemble_beta': 0, # adversarial student loss
            'unique_labels': 26,
            'generative_alpha':10,
            'generative_beta': 1,
            'weight_decay': 1e-2
        },

    'mnist':
        {
            'ensemble_lr': 3e-4,
            'ensemble_batch_size': 128,
            'ensemble_epochs': 50,
            'num_pretrain_iters': 20,
            'ensemble_alpha': 1,    # teacher loss (server side)
            'ensemble_beta': 0,     # adversarial student loss
            'ensemble_eta': 1,      # diversity loss
            'unique_labels': 10,    # available labels
            'generative_alpha': 10, # used to regulate user training
            'generative_beta': 10, # used to regulate user training
            'weight_decay': 1e-2
        },

    'celeb':
        {
            'ensemble_lr': 3e-4,
            'ensemble_batch_size': 128,
            'ensemble_epochs': 50,
            'num_pretrain_iters': 20,
            'ensemble_alpha': 1,  # teacher loss (server side)
            'ensemble_beta': 0,  # adversarial student loss
            'unique_labels': 2,
            'generative_alpha': 10,
            'generative_beta': 10, 
            'weight_decay': 1e-2
        },

    'ucihar':
        {
            'ensemble_lr': 1e-4,
            'ensemble_batch_size': 128,
            'ensemble_epochs': 50,
            'num_pretrain_iters': 20,
            'ensemble_alpha': 1,    # teacher loss (server side)
            'ensemble_beta': 0,     # adversarial student loss
            'unique_labels': 6,     # 6 activity classes
            'generative_alpha': 10,
            'generative_beta': 1,
            'weight_decay': 1e-2
        },

    'pamap2':
        {
            'ensemble_lr': 1e-4,
            'ensemble_batch_size': 128,
            'ensemble_epochs': 50,
            'num_pretrain_iters': 20,
            'ensemble_alpha': 1,    # teacher loss (server side)
            'ensemble_beta': 0,     # adversarial student loss
            'unique_labels': 12,    # 12 PAMAP2 activity classes
            'generative_alpha': 10,
            'generative_beta': 1,
            'weight_decay': 1e-2
        },

    'wisdm':
        {
            'ensemble_lr': 1e-4,
            'ensemble_batch_size': 128,
            'ensemble_epochs': 50,
            'num_pretrain_iters': 20,
            'ensemble_alpha': 1,    # teacher loss (server side)
            'ensemble_beta': 0,     # adversarial student loss
            'unique_labels': 6,     # 6 WISDM activity classes
            'generative_alpha': 10,
            'generative_beta': 1,
            'weight_decay': 1e-2
        },

    'fedisic':
        {
            'ensemble_lr': 1e-4,
            'ensemble_batch_size': 128,
            'ensemble_epochs': 50,
            'num_pretrain_iters': 20,
            'ensemble_alpha': 1,    # teacher loss (server side)
            'ensemble_beta': 0,     # adversarial student loss
            'unique_labels': 8,     # 8 ISIC 2019 lesion classes
            'generative_alpha': 10,
            'generative_beta': 10,  # medical images benefit from stronger
                                    # generator regularisation (per FedGen
                                    # paper CIFAR settings)
            'weight_decay': 1e-2
        },

    'ham10000':
        {
            'ensemble_lr': 1e-4,
            'ensemble_batch_size': 128,
            'ensemble_epochs': 50,
            'num_pretrain_iters': 20,
            'ensemble_alpha': 1,    # teacher loss (server side)
            'ensemble_beta': 0,     # adversarial student loss
            'unique_labels': 7,     # 7 HAM10000 (ISIC 2018 Task 3) classes
            'generative_alpha': 10,
            'generative_beta': 10,
            'weight_decay': 1e-2
        },

    'cifar10':
        {
            'ensemble_lr': 3e-4,
            'ensemble_batch_size': 128,
            'ensemble_epochs': 50,
            'num_pretrain_iters': 20,
            'ensemble_alpha': 1,    # teacher loss (server side)
            'ensemble_beta': 0,     # adversarial student loss
            'ensemble_eta': 1,      # diversity loss
            'unique_labels': 10,    # 10 CIFAR-10 classes
            'generative_alpha': 10, # user regularisation
            'generative_beta': 10,  # RGB benefits from stronger generator
                                    # regularisation (per FedGen CIFAR settings)
            'weight_decay': 1e-2
        },

}

