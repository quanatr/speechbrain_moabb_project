import torch
import speechbrain as sb

class DeepConvNet(torch.nn.Module):
    def __init__(
        self,
        input_shape=None, # (batch_size, time_points, channels(electrodes), num_feature_maps)
        cnn_first_kernels=25,
        cnn_second_kernels=25,
        cnn_third_kernels=50,
        cnn_fourth_kernels=100,
        cnn_fifth_kernels=200,
        cnn_bnorm_momentum=0.1,
        cnn_bnorm_eps=1e-5,
        cnn_max_norm=2,
        cnn_kernelsize=(10, 1),
        cnn_pool_type="max",
        cnn_pool_size=(3, 1),
        dense_n_neurons=4,
        dense_max_norm=0.5,
        dropout=0.4,
        activation_type='elu',
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

        #####################################################
        # CONVOLUTIONAL MODULE
        #####################################################
        self.conv_module = torch.nn.Sequential()

        #####################################################
        # 1st conv layer
        self.conv_module.add_module(
            "conv_0",
            sb.nnet.CNN.Conv2d(
                in_channels=1,
                out_channels=cnn_first_kernels,
                kernel_size=cnn_kernelsize,
                padding="valid",
                bias=False,
                max_norm=cnn_max_norm,
                swap=True,
            ),
        )

        #####################################################
        # 2nd conv layer
        self.conv_module.add_module(
            "conv_1",
            sb.nnet.CNN.Conv2d(
                in_channels=cnn_first_kernels,
                out_channels=cnn_second_kernels,
                kernel_size=(1, C),
                padding="valid",
                bias=False,
                max_norm=cnn_max_norm,
                swap=True,
            ),
        )

        # 2nd layer batchnorm
        self.conv_module.add_module(
            "bnorm_1",
            sb.nnet.normalization.BatchNorm2d(
                input_size=cnn_second_kernels,
                momentum=cnn_bnorm_momentum,
                eps=cnn_bnorm_eps,
                affine=True,
            ),
        )

        # 2nd layer activation
        self.conv_module.add_module(
            "act_1",
            activation
        )

        # 2nd layer pooling
        self.conv_module.add_module(
            "pool_1",
            sb.nnet.pooling.Pooling2d(
                pool_type=cnn_pool_type,
                kernel_size=cnn_pool_size,
                stride=cnn_pool_size,
            ),
        )

        # 2nd layer dropout
        self.conv_module.add_module(
            "dropout_1",
            torch.nn.Dropout(p=dropout),
        )

        #####################################################
        # 3rd conv layer
        self.conv_module.add_module(
            "conv_2",
            sb.nnet.CNN.Conv2d(
                in_channels=cnn_second_kernels,
                out_channels=cnn_third_kernels,
                kernel_size=cnn_kernelsize,
                padding="valid",
                bias=False,
                max_norm=cnn_max_norm,
                swap=True,
            ),
        )

        # 3rd layer batchnorm
        self.conv_module.add_module(
            "bnorm_2",
            sb.nnet.normalization.BatchNorm2d(
                input_size=cnn_third_kernels,
                momentum=cnn_bnorm_momentum,
                eps=cnn_bnorm_eps,
                affine=True,
            ),
        )

        # 3rd layer activation
        self.conv_module.add_module(
            "act_2",
            activation
        )

        # 3rd layer pooling
        self.conv_module.add_module(
            "pool_2",
            sb.nnet.pooling.Pooling2d(
                pool_type=cnn_pool_type,
                kernel_size=cnn_pool_size,
                stride=cnn_pool_size,
            ),
        )

        # 3rd layer dropout
        self.conv_module.add_module(
            "dropout_2",
            torch.nn.Dropout(p=dropout),
        )

        #####################################################
        # 4th conv layer
        self.conv_module.add_module(
            "conv_3",
            sb.nnet.CNN.Conv2d(
                in_channels=cnn_third_kernels,
                out_channels=cnn_fourth_kernels,
                kernel_size=cnn_kernelsize,
                padding="valid",
                bias=False,
                max_norm=cnn_max_norm,
                swap=True,
            ),
        )

        # 4th layer batchnorm
        self.conv_module.add_module(
            "bnorm_3",
            sb.nnet.normalization.BatchNorm2d(
                input_size=cnn_fourth_kernels,
                momentum=cnn_bnorm_momentum,
                eps=cnn_bnorm_eps,
                affine=True,
            ),
        )

        # 4th layer activation
        self.conv_module.add_module(
            "act_3",
            activation
        )

        # 4th layer pooling
        self.conv_module.add_module(
            "pool_3",
            sb.nnet.pooling.Pooling2d(
                pool_type=cnn_pool_type,
                kernel_size=cnn_pool_size,
                stride=cnn_pool_size,
            ),
        )

        # 4th layer dropout
        self.conv_module.add_module(
            "dropout_3",
            torch.nn.Dropout(p=dropout),
        )

        #####################################################
        # 5th conv layer
        self.conv_module.add_module(
            "conv_4",
            sb.nnet.CNN.Conv2d(
                in_channels=cnn_fourth_kernels,
                out_channels=cnn_fifth_kernels,
                kernel_size=cnn_kernelsize,
                padding="valid",
                bias=False,
                max_norm=cnn_max_norm,
                swap=True,
            ),
        )

        # 5th layer batchnorm
        self.conv_module.add_module(
            "bnorm_4",
            sb.nnet.normalization.BatchNorm2d(
                input_size=cnn_fifth_kernels,
                momentum=cnn_bnorm_momentum,
                eps=cnn_bnorm_eps,
                affine=True,
            ),
        )

        # 5th layer activation
        self.conv_module.add_module(
            "act_4",
            activation
        )

        # 5th layer pooling
        self.conv_module.add_module(
            "pool_4",
            sb.nnet.pooling.Pooling2d(
                pool_type=cnn_pool_type,
                kernel_size=cnn_pool_size,
                stride=cnn_pool_size,
            ),
        )

        # 5th layer dropout
        self.conv_module.add_module(
            "dropout_4",
            torch.nn.Dropout(p=dropout),
        )

        #####################################################
        # DENSE MODULE
        #####################################################
        # Shape of intermediate feature map
        current_out = self.conv_module(
            torch.ones((1,) + tuple(input_shape[1:-1]) + (1,))
        )
        dense_input_size = self._num_flat_features(current_out)

        self.dense_module = torch.nn.Sequential()

        # flatten
        self.dense_module.add_module(
            "flatten", torch.nn.Flatten(),
        )

        # linear
        self.dense_module.add_module(
            "fc_out",
            sb.nnet.linear.Linear(
                input_size=dense_input_size,
                n_neurons=dense_n_neurons,
                max_norm=dense_max_norm,
            ),
        )

        # final activation
        self.dense_module.add_module("act_out", torch.nn.LogSoftmax(dim=1))

    def _num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        x = self.conv_module(x)
        x = self.dense_module(x)
        return x