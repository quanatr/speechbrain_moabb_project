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
        # CONVOLUTIONAL MODULE
        #####################################################
        self.branches = {
            "b1": torch.nn.Sequential(),
            # "b2": torch.nn.Sequential(),
            # "b3": torch.nn.Sequential(),
        }

        # loop through all branches
        for branch in self.branches.keys():
            self.branches[branch] 

            # temporal
            self.branches[branch].add_module(
                f"temp_conv_{branch}",
                sb.nnet.CNN.Conv2d(
                    in_channels=1,
                    out_channels=self.cnn_temporal_kernel_counts[branch],
                    kernel_size=self.cnn_temporal_kernelsizes[branch],
                    padding="same",
                    padding_mode="constant",
                    bias=False,
                    swap=True,
                ),
            )

            # temporal batchnorm
            self.branches[branch].add_module(
                f"temp_bnorm_{branch}",
                sb.nnet.normalization.BatchNorm2d(
                    input_size=self.cnn_temporal_kernel_counts[branch],
                    affine=True,
                ),
            )

            # num of spatial depthwise filters
            cnn_depth_kernels = (
                self.cnn_temporal_kernel_counts[branch] * cnn_depth_multiplier
            )

            # depthwise
            self.branches[branch].add_module(
                f"depth_conv_{branch}",
                sb.nnet.CNN.Conv2d(
                    in_channels=self.cnn_temporal_kernel_counts[branch],
                    out_channels=cnn_depth_kernels,
                    kernel_size=(1, C),
                    padding="valid",
                    bias=False,
                    max_norm=cnn_depth_max_norm,
                    swap=True,
                ),
            )

            # depthwise batchnorm
            self.branches[branch].add_module(
                f"depth_bnorm_{branch}",
                sb.nnet.normalization.BatchNorm2d(
                    input_size=cnn_depth_kernels,
                    affine=True,
                ),
            )

            # depthwise activation
            self.branches[branch].add_module(
                f"depth_act_{branch}",
                activation
            )

            # depthwise pooling
            self.branches[branch].add_module(
                f"depth_pool_{branch}",
                sb.nnet.pooling.Pooling2d(
                    pool_type=cnn_pool_type,
                    kernel_size=cnn_depth_pool_size,
                    stride=cnn_depth_pool_size,
                    pool_axis=[1, 2],
                ),
            )

            # depthwise dropout
            self.branches[branch].add_module(
                f"depth_dropout_{branch}",
                torch.nn.Dropout(p=dropout),
            )

            # num of separable temporal filters
            cnn_sep_temporal_kernels = (
                cnn_depth_kernels * cnn_sep_temporal_multiplier
            )

            # separable temporal (depthwise_component)
            self.branches[branch].add_module(
                f"sep_temp_depth_conv_{branch}",
                sb.nnet.CNN.Conv2d(
                    in_channels=cnn_depth_kernels,
                    out_channels=cnn_sep_temporal_kernels,
                    kernel_size=self.cnn_sep_temporal_kernelsizes[branch],
                    groups=cnn_depth_kernels,
                    padding="same",
                    padding_mode="constant",
                    bias=False,
                    swap=True,
                ),
            )

            # separable temporal (pointwise_component)
            self.branches[branch].add_module(
                f"sep_temp_point_conv_{branch}",
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
            self.branches[branch].add_module(
                f"sep_temp_act_{branch}",
                activation
            )

            # seperable temporal pooling
            self.branches[branch].add_module(
                f"sep_temp_pool_{branch}",
                sb.nnet.pooling.Pooling2d(
                    pool_type=cnn_pool_type,
                    kernel_size=cnn_sep_temporal_pool_size,
                    stride=cnn_sep_temporal_pool_size,
                    pool_axis=[1, 2],
                ),
            )

            # separable temporal dropout
            self.branches[branch].add_module(
                f"sep_temp_dropout_{branch}",
                torch.nn.Dropout(p=dropout),
            )

            # flatten
            self.branches[branch].add_module(
                f"flatten_{branch}",
                torch.nn.Flatten(),
            )

        #####################################################
        # DENSE MODULE
        #####################################################
        input_size = (
            torch.ones((1,) + tuple(input_shape[1:-1]) + (1,))
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        branch_outputs = []
        for branch in self.branches.values():
            branch = branch.to(device)
            input_size = input_size.to(device)
            output = branch(input_size)
            branch_outputs.append(output)

        merge = torch.cat(branch_outputs, dim=1)

        self.linear = sb.nnet.linear.Linear(
            input_size=self._num_flat_features(merge),
            n_neurons=dense_n_neurons,
            max_norm=dense_max_norm,
        )

        self.softmax = torch.nn.LogSoftmax(dim=1)

    def _num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        # group all branch outputs
        branch_outputs = []
        for branch in self.branches.values():
            branch_outputs.append(branch(x))

        # fusion
        merge = torch.cat(branch_outputs, dim=1)

        # linear layer
        x = self.linear(merge)

        # softmax layer
        x = self.softmax(x)

        return x