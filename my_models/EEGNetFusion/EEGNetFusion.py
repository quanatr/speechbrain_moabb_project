import torch
import speechbrain as sb
import numpy as np

class EEGNetFusion(torch.nn.Module):
    def __init__(
        self,
        input_shape=None, # (batch_size, time_points, channels(electrodes), num_feature_maps)

        # branch 1
        cnn_temporal_kernels_b1=4,
        cnn_temporal_kernelsize_b1=(64, 1),
        cnn_sep_temporal_kernelsize_b1=(8, 1),

        # branch 2
        cnn_temporal_kernels_b2=8,
        cnn_temporal_kernelsize_b2=(128, 1),
        cnn_sep_temporal_kernelsize_b2=(16, 1),

        # branch 3
        cnn_temporal_kernels_b3=16,
        cnn_temporal_kernelsize_b3=(256, 1),
        cnn_sep_temporal_kernelsize_b3=(32, 1),

        # common parameters
        cnn_depth_multiplier=2,
        cnn_depth_max_norm=1,
        cnn_depth_pool_size=(4, 1),
        cnn_sep_temporal_multiplier=1,
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

        self.cnn_temporal_kernel_counts = {
            "b1": cnn_temporal_kernels_b1,
            "b2": cnn_temporal_kernels_b2,
            "b3": cnn_temporal_kernels_b3,
        }

        self.cnn_temporal_kernelsizes = {
            "b1": cnn_temporal_kernelsize_b1,
            "b2": cnn_temporal_kernelsize_b2,
            "b3": cnn_temporal_kernelsize_b3,
        }

        self.cnn_sep_temporal_kernelsizes = {
            "b1": cnn_sep_temporal_kernelsize_b1,
            "b2": cnn_sep_temporal_kernelsize_b2,
            "b3": cnn_sep_temporal_kernelsize_b3,
        }

        #####################################################
        # BRANCH MODULE 1
        #####################################################
        self.branch_module_1 = torch.nn.Sequential()

        # temporal
        self.branch_module_1.add_module(
            f"temp_conv_b1",
            sb.nnet.CNN.Conv2d(
                in_channels=1,
                out_channels=self.cnn_temporal_kernel_counts["b1"],
                kernel_size=self.cnn_temporal_kernelsizes["b1"],
                padding="same",
                padding_mode="constant",
                bias=False,
                swap=True,
            ),
        )

        # temporal batchnorm
        self.branch_module_1.add_module(
            f"temp_bnorm_b1",
            sb.nnet.normalization.BatchNorm2d(
                input_size=self.cnn_temporal_kernel_counts["b1"],
                affine=True,
            ),
        )
        
        # num of spatial depthwise filters
        cnn_depth_kernels = (
            self.cnn_temporal_kernel_counts["b1"] * cnn_depth_multiplier
        )

        # depthwise
        self.branch_module_1.add_module(
            f"depth_conv_b1",
            sb.nnet.CNN.Conv2d(
                in_channels=self.cnn_temporal_kernel_counts["b1"],
                out_channels=cnn_depth_kernels,
                kernel_size=(1, C),
                padding="valid",
                bias=False,
                max_norm=cnn_depth_max_norm,
                swap=True,
            ),
        )

        # depthwise batchnorm
        self.branch_module_1.add_module(
            f"depth_bnorm_b1",
            sb.nnet.normalization.BatchNorm2d(
                input_size=cnn_depth_kernels,
                affine=True,
            ),
        )

        # depthwise activation
        self.branch_module_1.add_module(
            f"depth_act_b1",
            activation
        )

        # depthwise pooling
        self.branch_module_1.add_module(
            f"depth_pool_b1",
            sb.nnet.pooling.Pooling2d(
                pool_type=cnn_pool_type,
                kernel_size=cnn_depth_pool_size,
                stride=cnn_depth_pool_size,
                pool_axis=[1, 2],
            ),
        )

        # depthwise dropout
        self.branch_module_1.add_module(
            f"depth_dropout_b1",
            torch.nn.Dropout(p=dropout),
        )

        # num of separable temporal filters
        cnn_sep_temporal_kernels = (
            cnn_depth_kernels * cnn_sep_temporal_multiplier
        )

        # separable temporal (depthwise_component)
        self.branch_module_1.add_module(
            f"sep_temp_depth_conv_b1",
            sb.nnet.CNN.Conv2d(
                in_channels=cnn_depth_kernels,
                out_channels=cnn_sep_temporal_kernels,
                kernel_size=self.cnn_sep_temporal_kernelsizes["b1"],
                groups=cnn_depth_kernels,
                padding="same",
                padding_mode="constant",
                bias=False,
                swap=True,
            ),
        )

        # separable temporal (pointwise_component)
        self.branch_module_1.add_module(
            f"sep_temp_point_conv_b1",
            sb.nnet.CNN.Conv2d(
                in_channels=cnn_sep_temporal_kernels,
                out_channels=cnn_sep_temporal_kernels,
                kernel_size=(1, 1),
                padding="valid",
                bias=False,
                swap=True,
            ),
        )

        # seprarable temporal batchnorm
        self.branch_module_1.add_module(
            f"sep_temp_bnorm_b1",
            sb.nnet.normalization.BatchNorm2d(
                input_size=cnn_sep_temporal_kernels,
                affine=True,
            ),
        )

        # separable temporal activation
        self.branch_module_1.add_module(
            f"sep_temp_act_b1",
            activation
        )

        # seperable temporal pooling
        self.branch_module_1.add_module(
            f"sep_temp_pool_b1",
            sb.nnet.pooling.Pooling2d(
                pool_type=cnn_pool_type,
                kernel_size=cnn_sep_temporal_pool_size,
                stride=cnn_sep_temporal_pool_size,
                pool_axis=[1, 2],
            ),
        )

        # separable temporal dropout
        self.branch_module_1.add_module(
            f"sep_temp_dropout_b1",
            torch.nn.Dropout(p=dropout),
        )

        # flatten
        self.branch_module_1.add_module(
            f"flatten_b1",
            torch.nn.Flatten(),
        )

        #####################################################
        # BRANCH MODULE 2
        #####################################################
        self.branch_module_2 = torch.nn.Sequential()

        # temporal
        self.branch_module_2.add_module(
            f"temp_conv_b2",
            sb.nnet.CNN.Conv2d(
                in_channels=1,
                out_channels=self.cnn_temporal_kernel_counts["b2"],
                kernel_size=self.cnn_temporal_kernelsizes["b2"],
                padding="same",
                padding_mode="constant",
                bias=False,
                swap=True,
            ),
        )

        # temporal batchnorm
        self.branch_module_2.add_module(
            f"temp_bnorm_b2",
            sb.nnet.normalization.BatchNorm2d(
                input_size=self.cnn_temporal_kernel_counts["b2"],
                affine=True,
            ),
        )

        # num of spatial depthwise filters
        cnn_depth_kernels = (
            self.cnn_temporal_kernel_counts["b2"] * cnn_depth_multiplier
        )

        # depthwise
        self.branch_module_2.add_module(
            f"depth_conv_b2",
            sb.nnet.CNN.Conv2d(
                in_channels=self.cnn_temporal_kernel_counts["b2"],
                out_channels=cnn_depth_kernels,
                kernel_size=(1, C),
                padding="valid",
                bias=False,
                max_norm=cnn_depth_max_norm,
                swap=True,
            ),
        )

        # depthwise batchnorm
        self.branch_module_2.add_module(
            f"depth_bnorm_b2",
            sb.nnet.normalization.BatchNorm2d(
                input_size=cnn_depth_kernels,
                affine=True,
            ),
        )

        # depthwise activation
        self.branch_module_2.add_module(
            f"depth_act_b2",
            activation
        )

        # depthwise pooling
        self.branch_module_2.add_module(
            f"depth_pool_b2",
            sb.nnet.pooling.Pooling2d(
                pool_type=cnn_pool_type,
                kernel_size=cnn_depth_pool_size,
                stride=cnn_depth_pool_size,
                pool_axis=[1, 2],
            ),
        )

        # depthwise dropout
        self.branch_module_2.add_module(
            f"depth_dropout_b2",
            torch.nn.Dropout(p=dropout),
        )

        # num of separable temporal filters
        cnn_sep_temporal_kernels = (
            cnn_depth_kernels * cnn_sep_temporal_multiplier
        )

        # separable temporal (depthwise_component)
        self.branch_module_2.add_module(
            f"sep_temp_depth_conv_b2",
            sb.nnet.CNN.Conv2d(
                in_channels=cnn_depth_kernels,
                out_channels=cnn_sep_temporal_kernels,
                kernel_size=self.cnn_sep_temporal_kernelsizes["b2"],
                groups=cnn_depth_kernels,
                padding="same",
                padding_mode="constant",
                bias=False,
                swap=True,
            ),
        )

        # separable temporal (pointwise_component)
        self.branch_module_2.add_module(
            f"sep_temp_point_conv_b2",
            sb.nnet.CNN.Conv2d(
                in_channels=cnn_sep_temporal_kernels,
                out_channels=cnn_sep_temporal_kernels,
                kernel_size=(1, 1),
                padding="valid",
                bias=False,
                swap=True,
            ),
        )

        # seprarable temporal batchnorm
        self.branch_module_2.add_module(
            f"sep_temp_bnorm_b2",
            sb.nnet.normalization.BatchNorm2d(
                input_size=cnn_sep_temporal_kernels,
                affine=True,
            ),
        )

        # separable temporal activation
        self.branch_module_2.add_module(
            f"sep_temp_act_b2",
            activation
        )

        # seperable temporal pooling
        self.branch_module_2.add_module(
            f"sep_temp_pool_b2",
            sb.nnet.pooling.Pooling2d(
                pool_type=cnn_pool_type,
                kernel_size=cnn_sep_temporal_pool_size,
                stride=cnn_sep_temporal_pool_size,
                pool_axis=[1, 2],
            ),
        )

        # separable temporal dropout
        self.branch_module_2.add_module(
            f"sep_temp_dropout_b2",
            torch.nn.Dropout(p=dropout),
        )

        # flatten
        self.branch_module_2.add_module(
            f"flatten_b2",
            torch.nn.Flatten(),
        )

        #####################################################
        # BRANCH MODULE 3
        #####################################################
        self.branch_module_3 = torch.nn.Sequential()

        # temporal
        self.branch_module_3.add_module(
            f"temp_conv_b3",
            sb.nnet.CNN.Conv2d(
                in_channels=1,
                out_channels=self.cnn_temporal_kernel_counts["b3"],
                kernel_size=self.cnn_temporal_kernelsizes["b3"],
                padding="same",
                padding_mode="constant",
                bias=False,
                swap=True,
            ),
        )

        # temporal batchnorm
        self.branch_module_3.add_module(
            f"temp_bnorm_b3",
            sb.nnet.normalization.BatchNorm2d(
                input_size=self.cnn_temporal_kernel_counts["b3"],
                affine=True,
            ),
        )

        # num of spatial depthwise filters
        cnn_depth_kernels = (
            self.cnn_temporal_kernel_counts["b3"] * cnn_depth_multiplier
        )

        # depthwise
        self.branch_module_3.add_module(
            f"depth_conv_b3",
            sb.nnet.CNN.Conv2d(
                in_channels=self.cnn_temporal_kernel_counts["b3"],
                out_channels=cnn_depth_kernels,
                kernel_size=(1, C),
                padding="valid",
                bias=False,
                max_norm=cnn_depth_max_norm,
                swap=True,
            ),
        )

        # depthwise batchnorm
        self.branch_module_3.add_module(
            f"depth_bnorm_b3",
            sb.nnet.normalization.BatchNorm2d(
                input_size=cnn_depth_kernels,
                affine=True,
            ),
        )

        # depthwise activation
        self.branch_module_3.add_module(
            f"depth_act_b3",
            activation
        )

        # depthwise pooling
        self.branch_module_3.add_module(
            f"depth_pool_b3",
            sb.nnet.pooling.Pooling2d(
                pool_type=cnn_pool_type,
                kernel_size=cnn_depth_pool_size,
                stride=cnn_depth_pool_size,
                pool_axis=[1, 2],
            ),
        )

        # depthwise dropout
        self.branch_module_3.add_module(
            f"depth_dropout_b3",
            torch.nn.Dropout(p=dropout),
        )

        # num of separable temporal filters
        cnn_sep_temporal_kernels = (
            cnn_depth_kernels * cnn_sep_temporal_multiplier
        )

        # separable temporal (depthwise_component)
        self.branch_module_3.add_module(
            f"sep_temp_depth_conv_b3",
            sb.nnet.CNN.Conv2d(
                in_channels=cnn_depth_kernels,
                out_channels=cnn_sep_temporal_kernels,
                kernel_size=self.cnn_sep_temporal_kernelsizes["b3"],
                groups=cnn_depth_kernels,
                padding="same",
                padding_mode="constant",
                bias=False,
                swap=True,
            ),
        )

        # separable temporal (pointwise_component)
        self.branch_module_3.add_module(
            f"sep_temp_point_conv_b3",
            sb.nnet.CNN.Conv2d(
                in_channels=cnn_sep_temporal_kernels,
                out_channels=cnn_sep_temporal_kernels,
                kernel_size=(1, 1),
                padding="valid",
                bias=False,
                swap=True,
            ),
        )

        # seprarable temporal batchnorm
        self.branch_module_3.add_module(
            f"sep_temp_bnorm_b3",
            sb.nnet.normalization.BatchNorm2d(
                input_size=cnn_sep_temporal_kernels,
                affine=True,
            ),
        )

        # separable temporal activation
        self.branch_module_3.add_module(
            f"sep_temp_act_b3",
            activation
        )

        # seperable temporal pooling
        self.branch_module_3.add_module(
            f"sep_temp_pool_b3",
            sb.nnet.pooling.Pooling2d(
                pool_type=cnn_pool_type,
                kernel_size=cnn_sep_temporal_pool_size,
                stride=cnn_sep_temporal_pool_size,
                pool_axis=[1, 2],
            ),
        )

        # separable temporal dropout
        self.branch_module_3.add_module(
            f"sep_temp_dropout_b3",
            torch.nn.Dropout(p=dropout),
        )

        # flatten
        self.branch_module_3.add_module(
            f"flatten_b3",
            torch.nn.Flatten(),
        )

        #####################################################
        # DENSE MODULE
        #####################################################
        sample_input = (
            torch.ones((1,) + tuple(input_shape[1:-1]) + (1,))
        )

        # fusion
        branch_outputs = []
        output_b1 = self.branch_module_1(sample_input)
        branch_outputs.append(output_b1)
        output_b2 = self.branch_module_2(sample_input)
        branch_outputs.append(output_b2)
        output_b3 = self.branch_module_3(sample_input)
        branch_outputs.append(output_b3)
        merge = torch.cat(branch_outputs, dim=1)

        # Shape of intermediate feature map
        dense_input_size = self._num_flat_features(merge)

        self.dense_module = torch.nn.Sequential()

        # flatten
        self.dense_module.add_module(
            "flatten",
            torch.nn.Flatten(),
        )

        # linear
        self.dense_module.add_module(
            "fc_out",
            sb.nnet.linear.Linear(
                input_size=dense_input_size,
                n_neurons=dense_n_neurons,
                max_norm=dense_max_norm,)
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
        # group all branch outputs
        branch_outputs = []
        x_1 = self.branch_module_1(x)
        branch_outputs.append(x_1)
        x_2 = self.branch_module_2(x)
        branch_outputs.append(x_2)
        x_3 = self.branch_module_3(x)
        branch_outputs.append(x_3)

        # fusion
        merge = torch.cat(branch_outputs, dim=1)

        # dense module
        x = self.dense_module(merge)
        return x