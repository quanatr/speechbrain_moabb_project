import torch
import speechbrain as sb

class Square(torch.nn.Module):
    def forward(self, x):
        return torch.square(x)

class Log(torch.nn.Module):
    def forward(self, x):
        return torch.log(torch.clamp(x, min=1e-6))

class MBShallowConvNet(torch.nn.Module):
    def __init__(
        self,
        input_shape=None,

        # branch 1
        cnn_temporal_kernels_b1=4,
        cnn_temporal_kernelsize_b1=(16, 1),
        dropout_b1=0,

        # branch 2
        cnn_temporal_kernels_b2=8,
        cnn_temporal_kernelsize_b2=(32, 1),
        dropout_b2=0.1,

        # branch 3
        cnn_temporal_kernels_b3=16,
        cnn_temporal_kernelsize_b3=(64, 1),
        dropout_b3=0.2,

        # common params
        cnn_bnorm_momentum=0.1,
        cnn_bnorm_eps=1e-5,
        cnn_max_norm=2,
        cnn_pool_type="avg",
        cnn_pool_size=(75, 1),
        cnn_pool_stride=(15, 1),
        dense_n_neurons=4,
        dense_max_norm=0.5,
        activation_type="elu",
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
        # BRANCH MODULE 1
        #####################################################
        self.branch_module_1 = torch.nn.Sequential()

        # temporal
        self.branch_module_1.add_module(
            "temp_conv_b1",
            sb.nnet.CNN.Conv2d(
                in_channels=1,
                out_channels=cnn_temporal_kernels_b1,
                kernel_size=cnn_temporal_kernelsize_b1,
                padding="same",
                padding_mode="constant",
                bias=False,
                max_norm=cnn_max_norm,
                swap=True,
            ),
        )

        # spatial
        cnn_spatial_kernels_b1 = cnn_temporal_kernels_b1

        self.branch_module_1.add_module(
            "spatial_conv_b1",
            sb.nnet.CNN.Conv2d(
                in_channels=cnn_temporal_kernels_b1,
                out_channels=cnn_spatial_kernels_b1,
                kernel_size=(1, C),
                padding="valid",
                bias=False,
                max_norm=cnn_max_norm,
                swap=True,
            ),
        )

        # spatial batchnorm
        self.branch_module_1.add_module(
            "spatial_bnorm_b1",
            sb.nnet.normalization.BatchNorm2d(
                input_size=cnn_spatial_kernels_b1,
                momentum=cnn_bnorm_momentum,
                eps=cnn_bnorm_eps,
                affine=True,
            ),
        )

        # square activation
        self.branch_module_1.add_module(
            "square_b1",
            Square(),
        )

        # pooling
        self.branch_module_1.add_module(
            "pool_b1",
            sb.nnet.pooling.Pooling2d(
                pool_type=cnn_pool_type,
                kernel_size=cnn_pool_size,
                stride=cnn_pool_stride,
            ),
        )

        # log activation
        self.branch_module_1.add_module(
            "log_b1",
            Log(),
        )

        # flatten
        self.branch_module_1.add_module(
            "flatten_b1",
            torch.nn.Flatten(),
        )

        # dropout
        self.branch_module_1.add_module(
            "dropout_b1",
            torch.nn.Dropout(p=dropout_b1),
        )

        #####################################################
        # BRANCH MODULE 2
        #####################################################
        self.branch_module_2 = torch.nn.Sequential()

        # temporal
        self.branch_module_2.add_module(
            "temp_conv_b2",
            sb.nnet.CNN.Conv2d(
                in_channels=1,
                out_channels=cnn_temporal_kernels_b2,
                kernel_size=cnn_temporal_kernelsize_b2,
                padding="same",
                padding_mode="constant",
                bias=False,
                max_norm=cnn_max_norm,
                swap=True,
            ),
        )

        # spatial
        cnn_spatial_kernels_b2 = cnn_temporal_kernels_b2

        self.branch_module_2.add_module(
            "spatial_conv_b2",
            sb.nnet.CNN.Conv2d(
                in_channels=cnn_temporal_kernels_b2,
                out_channels=cnn_spatial_kernels_b2,
                kernel_size=(1, C),
                padding="valid",
                bias=False,
                max_norm=cnn_max_norm,
                swap=True,
            ),
        )

        # spatial batchnorm
        self.branch_module_2.add_module(
            "spatial_bnorm_b2",
            sb.nnet.normalization.BatchNorm2d(
                input_size=cnn_spatial_kernels_b2,
                momentum=cnn_bnorm_momentum,
                eps=cnn_bnorm_eps,
                affine=True,
            ),
        )

        # square activation
        self.branch_module_2.add_module(
            "square_b2",
            Square(),
        )

        # pooling
        self.branch_module_2.add_module(
            "pool_b2",
            sb.nnet.pooling.Pooling2d(
                pool_type=cnn_pool_type,
                kernel_size=cnn_pool_size,
                stride=cnn_pool_stride,
            ),
        )

        # log activation
        self.branch_module_2.add_module(
            "log_b2",
            Log(),
        )

        # flatten
        self.branch_module_2.add_module(
            "flatten_b2",
            torch.nn.Flatten(),
        )

        # dropout
        self.branch_module_2.add_module(
            "dropout_b2",
            torch.nn.Dropout(p=dropout_b2),
        )

        #####################################################
        # BRANCH MODULE 3
        #####################################################
        self.branch_module_3 = torch.nn.Sequential()

        # temporal
        self.branch_module_3.add_module(
            "temp_conv_b3",
            sb.nnet.CNN.Conv2d(
                in_channels=1,
                out_channels=cnn_temporal_kernels_b3,
                kernel_size=cnn_temporal_kernelsize_b3,
                padding="same",
                padding_mode="constant",
                bias=False,
                max_norm=cnn_max_norm,
                swap=True,
            ),
        )

        # spatial
        cnn_spatial_kernels_b3 = cnn_temporal_kernels_b3

        self.branch_module_3.add_module(
            "spatial_conv_b3",
            sb.nnet.CNN.Conv2d(
                in_channels=cnn_temporal_kernels_b3,
                out_channels=cnn_spatial_kernels_b3,
                kernel_size=(1, C),
                padding="valid",
                bias=False,
                max_norm=cnn_max_norm,
                swap=True,
            ),
        )

        # spatial batchnorm
        self.branch_module_3.add_module(
            "spatial_bnorm_b3",
            sb.nnet.normalization.BatchNorm2d(
                input_size=cnn_spatial_kernels_b3,
                momentum=cnn_bnorm_momentum,
                eps=cnn_bnorm_eps,
                affine=True,
            ),
        )

        # square activation
        self.branch_module_3.add_module(
            "square_b3",
            Square(),
        )

        # pooling
        self.branch_module_3.add_module(
            "pool_b3",
            sb.nnet.pooling.Pooling2d(
                pool_type=cnn_pool_type,
                kernel_size=cnn_pool_size,
                stride=cnn_pool_stride,
            ),
        )

        # log activation
        self.branch_module_3.add_module(
            "log_b3",
            Log(),
        )

        # flatten
        self.branch_module_3.add_module(
            "flatten_b3",
            torch.nn.Flatten(),
        )

        # dropout
        self.branch_module_3.add_module(
            "dropout_b3",
            torch.nn.Dropout(p=dropout_b3),
        )

        #####################################################
        # DENSE MODULE
        #####################################################
        # create a dummy tensor to infer the number of features
        sample_input = (
            torch.ones((1,) + tuple(input_shape[1:-1]) + (1,))
        )

        # fusion layer
        branch_outputs = []
        output_b1 = self.branch_module_1(sample_input)
        branch_outputs.append(output_b1)
        output_b2 = self.branch_module_2(sample_input)
        branch_outputs.append(output_b2)
        output_b3 = self.branch_module_3(sample_input)
        branch_outputs.append(output_b3)
        merge = torch.cat(branch_outputs, dim=1)

        # get the number of features after the fusion
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
        self.dense_module.add_module(
            "act_out", 
            torch.nn.LogSoftmax(dim=1))

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