import torch
import speechbrain as sb


class MyModel(torch.nn.Module):
    """
    """

    def __init__(
        self,
        input_shape=None, # (1, T, C, 1)
        conv1_kernels=16,
        conv2_kernels=32,
        conv3_kernels=64,
        conv1_kernelsize=(3, 1),
        conv2_kernelsize=(5, 1),
        conv3_kernelsize=(7, 1),
        conv_pool=(4, 1),
        cnn_pool_type="max",
        activation_type="leaky_relu",
        dropout=0.2,
    ):
        super().__init__()
        if input_shape is None:
            raise ValueError("Must specify input_shape")
        if activation_type == "gelu":
            activation = torch.nn.GELU()
        elif activation_type == "elu":
            activation = torch.nn.ELU()
        elif activation_type == "relu":
            activation = torch.nn.ReLU()
        elif activation_type == "leaky_relu":
            activation = torch.nn.LeakyReLU()
        elif activation_type == "prelu":
            activation = torch.nn.PReLU()
        else:
            raise ValueError("Wrong hidden activation function")
        self.default_sf = 128  # sampling rate of the original publication (Hz)
        T = input_shape[1]
        C = input_shape[2]

        # CONVOLUTIONAL MODULE
        self.conv_module = torch.nn.Sequential()

        # 1st layer
        self.conv_module.add_module(
            "conv_1",
            sb.nnet.CNN.Conv2d(
                in_channels=1,
                out_channels=conv1_kernels,
                kernel_size=conv1_kernelsize,
                padding="same",
                padding_mode="constant",
                bias=False,
                swap=True,
            ),
        )
        self.conv_module.add_module(
            "bnorm_1",
            sb.nnet.normalization.BatchNorm2d(
                input_size=conv1_kernels
            ),
        )
        self.conv_module.add_module(
            "activation_1",
            activation
        )
        self.conv_module.add_module(
            "pool_1",
            sb.nnet.pooling.Pooling2d(
                pool_type=cnn_pool_type,
                kernel_size=conv_pool,
                stride=conv_pool,
            ),
        )

       # 2nd layer
        self.conv_module.add_module(
            "conv_2",
            sb.nnet.CNN.Conv2d(
                in_channels=conv1_kernels,
                out_channels=conv2_kernels,
                kernel_size=conv2_kernelsize,
                padding="same",
                padding_mode="constant",
                bias=False,
                swap=True,
            ),
        )
        self.conv_module.add_module(
            "bnorm_2",
            sb.nnet.normalization.BatchNorm2d(
                input_size=conv2_kernels
            ),
        )
        self.conv_module.add_module(
            "activation_2",
            activation
        )
        self.conv_module.add_module(
            "pool_2",
            sb.nnet.pooling.Pooling2d(
                pool_type=cnn_pool_type,
                kernel_size=conv_pool,
                stride=conv_pool,
            ),
        )

       # 3rd layer
        self.conv_module.add_module(
            "conv_3",
            sb.nnet.CNN.Conv2d(
                in_channels=conv2_kernels,
                out_channels=conv3_kernels,
                kernel_size=conv3_kernelsize,
                padding="same",
                padding_mode="constant",
                bias=False,
                swap=True,
            ),
        )
        self.conv_module.add_module(
            "bnorm_3",
            sb.nnet.normalization.BatchNorm2d(
                input_size=conv3_kernels
            ),
        )
        self.conv_module.add_module(
            "activation_3",
            activation
        )
        self.conv_module.add_module(
            "pool_3",
            sb.nnet.pooling.Pooling2d(
                pool_type=cnn_pool_type,
                kernel_size=conv_pool,
                stride=conv_pool,
            ),
        )

        # fully connected layer
        self.conv_module.add_module(
            "flatten",
            torch.nn.Flatten(),
        )
        self.conv_module.add_module(
            "linear",
            torch.nn.Linear(
                in_features=9856,
                out_features=4,
            ),
        )
        self.conv_module.add_module(
            "dropout",
            torch.nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        """
        x = self.conv_module(x)
        return x