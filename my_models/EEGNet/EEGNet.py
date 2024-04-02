import torch
import speechbrain as sb

class EEGNet(torch.nn.Module):
    def __init__(
        self,
        input_shape=None, # (batch_size, time_points, channels(electrodes), num_feature_maps)
        cnn_temporal_kernels=8,
        cnn_temporal_kernelsize=(64, 1),
        cnn_depth_multiplier=2,
        cnn_depth_max_norm=1,
        cnn_depth_pool_size=(4, 1),
        cnn_sep_temporal_multiplier=2,
        cnn_sep_temporal_kernelsize=(16, 1),
        cnn_sep_temporal_pool_size=(8, 1),
        cnn_pool_type="avg",
        dense_n_neurons=4,
        dense_max_norm=0.25,
        dropout=0.25,
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
        # temporal layer
        self.conv_module.add_module(
            "temp_conv",
            sb.nnet.CNN.Conv2d(
                in_channels=1,
                out_channels=cnn_temporal_kernels,
                kernel_size=cnn_temporal_kernelsize,
                padding="same",
                padding_mode="constant",
                bias=False,
                swap=True,
            ),
        )

        # temporal batchnorm
        self.conv_module.add_module(
            "temp_bnorm",
            sb.nnet.normalization.BatchNorm2d(
                input_size=cnn_temporal_kernels,
                affine=True,
            ),
        )

        #####################################################
        # num of spatial depthwise filters
        cnn_spatial_kernels = (
            cnn_temporal_kernels * cnn_depth_multiplier
            )

        # depthwise layer
        self.conv_module.add_module(
            "depth_conv_0",
            sb.nnet.CNN.Conv2d(
                in_channels=cnn_temporal_kernels,
                out_channels=cnn_spatial_kernels,
                kernel_size=(1, C),
                padding="valid",
                bias=False,
                max_norm=cnn_depth_max_norm,
                swap=True,
            ),
        )

        # depthwise batchnorm
        self.conv_module.add_module(
            "depth_bnorm",
            sb.nnet.normalization.BatchNorm2d(
                input_size=cnn_spatial_kernels,
                affine=True,
            ),
        )

        # depthwise activation
        self.conv_module.add_module(
            "depth_act",
            activation
        )

        # depthwise pooling
        self.conv_module.add_module(
            "depth_pool",
            sb.nnet.pooling.Pooling2d(
                pool_type=cnn_pool_type,
                kernel_size=cnn_depth_pool_size,
                stride=cnn_depth_pool_size,
            ),
        )

        # depthwise dropout
        self.conv_module.add_module(
            "depth_dropout",
            torch.nn.Dropout(p=dropout),
        )

        #####################################################
        # num of separable temporal filters
        cnn_sep_temporal_kernels = (
            cnn_temporal_kernels * cnn_sep_temporal_multiplier
            )

        # depthwise layer
        self.conv_module.add_module(
            "depth_conv_1",
            sb.nnet.CNN.Conv2d(
                in_channels=cnn_spatial_kernels,
                out_channels=cnn_sep_temporal_kernels,
                kernel_size=cnn_sep_temporal_kernelsize,
                groups=cnn_spatial_kernels,
                padding="same",
                padding_mode="constant",
                bias=False,
                swap=True,
            ),
        )

        # pointwise layer
        self.conv_module.add_module(
            "point_conv",
            sb.nnet.CNN.Conv2d(
                in_channels=cnn_sep_temporal_kernels,
                out_channels=cnn_sep_temporal_kernels,
                kernel_size=(1, 1),
                padding="valid",
                bias=False,
                swap=True,
            ),
        )

        # separable temporal activation
        self.conv_module.add_module(
            "sep_temp_act",
            activation
        )

        # seperable temporal pooling
        self.conv_module.add_module(
            "sep_temp_pool",
            sb.nnet.pooling.Pooling2d(
                pool_type=cnn_pool_type,
                kernel_size=cnn_sep_temporal_pool_size,
                stride=cnn_sep_temporal_pool_size,
            ),
        )

        # separable temporal dropout
        self.conv_module.add_module(
            "sep_temp_dropout",
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
                max_norm=dense_max_norm)
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