import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.optim import lr_scheduler


###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type="instance"):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == "batch":
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == "syncbatch":
        norm_layer = functools.partial(nn.SyncBatchNorm, affine=True, track_running_stats=True)
    elif norm_type == "instance":
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == "none":

        def norm_layer(x):
            return Identity()

    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == "linear":

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == "step":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError("learning rate policy [%s] is not implemented", opt.lr_policy)
    return scheduler


def init_weights(net, init_type="normal", init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError("initialization method [%s] is not implemented" % init_type)
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print("initialize network with %s" % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type="normal", init_gain=0.02):
    """Initialize a network: 1. register CPU/GPU device; 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.

    Return an initialized network.
    """
    import os

    if torch.cuda.is_available():
        if "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ["LOCAL_RANK"])
            net.to(local_rank)
            print(f"Initialized with device cuda:{local_rank}")
        else:
            net.to(0)
            print("Initialized with device cuda:0")
    init_weights(net, init_type, init_gain=init_gain)
    return net


def _parse_resunet_spec(netG, default_num_downs=8, default_bottleneck_blocks=4):
    """Parse ResUnet configuration from a netG string.

    Supported forms:
        ResUnet
        ResUnet_256
        ResUnet_128
        ResUnet_8
        ResUnet_8_b4
        ResUnet_b6

    The first numeric token is interpreted as either a canonical image size
    or the explicit number of downsampling stages. A token such as 'b4'
    overrides the number of bottleneck residual blocks.
    """
    spec = str(netG).lower()
    if not spec.startswith("resunet"):
        return None

    num_downs = default_num_downs
    bottleneck_blocks = default_bottleneck_blocks
    suffix = spec[len("resunet"):].strip("_")
    if suffix:
        for token in suffix.split("_"):
            if not token:
                continue
            if token.startswith("b") and token[1:].isdigit():
                bottleneck_blocks = int(token[1:])
                continue
            if token.isdigit():
                value = int(token)
                if value == 128:
                    num_downs = 7
                elif value == 256:
                    num_downs = 8
                elif value == 512:
                    num_downs = 9
                elif value >= 5:
                    num_downs = value
                else:
                    bottleneck_blocks = value

    return num_downs, bottleneck_blocks


def _parse_unetpp_spec(netG, default_num_stages=4):
    """Parse Unet++ configuration from a netG string.

    Supported forms:
        UnetPP
        UnetPP_256
        UnetPP_d4
        UnetPP_256_d4

    For canonical image-size tokens (128/256/512), this implementation uses
    the default 4-stage UNet++ topology. Explicit stage depth can be provided
    with tokens like d4/d5/d6.
    """
    spec = str(netG).lower().replace("unet++", "unetpp").replace("unet_plus_plus", "unetpp")
    if not spec.startswith("unetpp"):
        return None

    num_stages = default_num_stages
    suffix = spec[len("unetpp"):].strip("_")
    if suffix:
        for token in suffix.split("_"):
            if not token:
                continue
            if token.startswith("d") and token[1:].isdigit():
                num_stages = int(token[1:])
                continue
            if token.isdigit():
                value = int(token)
                if value in (128, 256, 512):
                    num_stages = default_num_stages
                elif 2 <= value <= 7:
                    num_stages = value

    return max(2, num_stages)


def _parse_mobilenet_spec(netG, default_width_mult=0.5, default_bottleneck_blocks=2, default_num_downs=4, default_expand_ratio=2.0):
    """Parse MobileNet configuration from a netG string.

    Supported forms:
        MobileNet
        MobileNet_w0.5
        MobileNet_b2
        MobileNet_w0.5_b2

    The default configuration is intentionally lightweight so it can be used
    as a fast test-time generator.
    """
    spec = str(netG).lower().replace("mobilenetv2", "mobilenet").replace("mobile_net", "mobilenet")
    if not spec.startswith("mobilenet"):
        return None

    width_mult = default_width_mult
    bottleneck_blocks = default_bottleneck_blocks
    num_downs = default_num_downs
    expand_ratio = default_expand_ratio

    suffix = spec[len("mobilenet"):].strip("_")
    if suffix:
        for token in suffix.split("_"):
            if not token:
                continue
            if token.startswith("b") and token[1:].isdigit():
                bottleneck_blocks = int(token[1:])
                continue
            if token.startswith("d") and token[1:].isdigit():
                num_downs = max(2, int(token[1:]))
                continue
            if token.startswith("e") and token[1:]:
                try:
                    expand_ratio = float(token[1:])
                except ValueError:
                    pass
                continue
            if token.startswith("w") and token[1:]:
                try:
                    width_mult = float(token[1:])
                except ValueError:
                    pass
                continue
            if token.isdigit():
                value = int(token)
                if value in (128, 256, 512):
                    num_downs = default_num_downs

    return width_mult, bottleneck_blocks, num_downs, expand_ratio


def define_G(input_nc, output_nc, ngf, netG, norm="batch", use_dropout=False, init_type="normal", init_gain=0.02):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_128 | unet_256 | MobileNet | ResUnet | UnetPP
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.

    Returns a generator
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == "resnet_9blocks":
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == "resnet_6blocks":
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == "unet_128":
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == "unet_256":
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        mobilenet_spec = _parse_mobilenet_spec(netG)
        if mobilenet_spec is not None:
            width_mult, bottleneck_blocks, num_downs, expand_ratio = mobilenet_spec
            net = MobileNetGenerator(
                input_nc,
                output_nc,
                ngf=ngf,
                width_mult=width_mult,
                bottleneck_blocks=bottleneck_blocks,
                num_downs=num_downs,
                expand_ratio=expand_ratio,
                norm_layer=norm_layer,
                use_dropout=use_dropout,
            )
        else:
            unetpp_spec = _parse_unetpp_spec(netG)
            if unetpp_spec is not None:
                net = UnetPlusPlusGenerator(
                    input_nc,
                    output_nc,
                    num_stages=unetpp_spec,
                    ngf=ngf,
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                )
            else:
                resunet_spec = _parse_resunet_spec(netG)
                if resunet_spec is not None:
                    num_downs, bottleneck_blocks = resunet_spec
                    net = ResUnetGenerator(
                        input_nc,
                        output_nc,
                        num_downs,
                        ngf,
                        norm_layer=norm_layer,
                        use_dropout=use_dropout,
                        bottleneck_blocks=bottleneck_blocks,
                    )
                else:
                    raise NotImplementedError("Generator model name [%s] is not recognized" % netG)
    return net


def define_D(input_nc, ndf, netD, n_layers_D=3, norm="batch", init_type="normal", init_gain=0.02):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == "basic":  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == "n_layers":  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == "pixel":  # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError("Discriminator model name [%s] is not recognized" % netD)
    return net


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ["wgangp"]:
            self.loss = None
        else:
            raise NotImplementedError("gan mode %s not implemented" % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ["lsgan", "vanilla"]:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == "wgangp":
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type="mixed", constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == "real":  # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == "fake":
            interpolatesv = fake_data
        elif type == "mixed":
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError(f"{type} not implemented")
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv, grad_outputs=torch.ones(disc_interpolates.size()).to(device), create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp  # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type="reflect"):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert n_blocks >= 0
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias), norm_layer(ngf), nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias), norm_layer(ngf * mult * 2), nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias), norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class _UnetPPConvBlock(nn.Module):
    """Two-conv block used by UNet++."""

    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(_UnetPPConvBlock, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        layers = [
            nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(output_nc),
            nn.ReLU(True),
            nn.Conv2d(output_nc, output_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(output_nc),
            nn.ReLU(True),
        ]
        if use_dropout:
            layers.append(nn.Dropout2d(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UnetPlusPlusGenerator(nn.Module):
    """Create a UNet++ (nested U-Net) generator."""

    def __init__(self, input_nc, output_nc, num_stages=5, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetPlusPlusGenerator, self).__init__()
        if num_stages < 2:
            raise ValueError("num_stages for UnetPlusPlusGenerator must be >= 2")

        self.num_stages = num_stages
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.channels = [min(ngf * (2 ** i), ngf * 8) for i in range(num_stages)]

        self.nodes = nn.ModuleDict()
        for j in range(num_stages):
            for i in range(num_stages - j):
                if j == 0:
                    in_ch = input_nc if i == 0 else self.channels[i - 1]
                else:
                    in_ch = self.channels[i + 1] + j * self.channels[i]
                out_ch = self.channels[i]
                self.nodes[f"x_{i}_{j}"] = _UnetPPConvBlock(in_ch, out_ch, norm_layer=norm_layer, use_dropout=use_dropout)

        self.head = nn.Sequential(
            nn.Conv2d(self.channels[0], output_nc, kernel_size=1, stride=1, padding=0),
            nn.Tanh(),
        )

    def forward(self, input):
        nodes = {}

        for i in range(self.num_stages):
            if i == 0:
                current = input
            else:
                current = self.pool(nodes[f"x_{i - 1}_0"])
            nodes[f"x_{i}_0"] = self.nodes[f"x_{i}_0"](current)

        for j in range(1, self.num_stages):
            for i in range(self.num_stages - j):
                up = F.interpolate(nodes[f"x_{i + 1}_{j - 1}"], size=nodes[f"x_{i}_0"].shape[2:], mode="bilinear", align_corners=False)
                dense = [nodes[f"x_{i}_{k}"] for k in range(j)]
                merged = torch.cat(dense + [up], dim=1)
                nodes[f"x_{i}_{j}"] = self.nodes[f"x_{i}_{j}"](merged)

        return self.head(nodes[f"x_0_{self.num_stages - 1}"])


class ResUnetGenerator(nn.Module):
    """Create a residual U-Net generator.

    This generator keeps the U-Net encoder-decoder topology, but each stage
    uses residual blocks to improve feature reuse and gradient flow.
    """

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, bottleneck_blocks=4):
        """Construct a ResUnet generator.

        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsampling stages
            ngf (int)       -- the base number of filters
            norm_layer      -- normalization layer
            use_dropout (bool) -- if use dropout layers in bottleneck blocks
            bottleneck_blocks (int) -- number of residual blocks at the bottleneck
        """
        super(ResUnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.stem = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
        )

        self.down_blocks = nn.ModuleList()
        self.skip_channels = []
        in_channels = ngf
        for i in range(num_downs):
            out_channels = min(ngf * (2 ** (i + 1)), ngf * 8)
            self.down_blocks.append(
                _ResUnetDownBlock(
                    in_channels,
                    out_channels,
                    norm_layer=norm_layer,
                    use_bias=use_bias,
                )
            )
            self.skip_channels.append(in_channels)
            in_channels = out_channels

        bottleneck = []
        for _ in range(bottleneck_blocks):
            bottleneck.append(
                ResnetBlock(
                    in_channels,
                    padding_type="zero",
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                    use_bias=use_bias,
                )
            )
        self.bottleneck = nn.Sequential(*bottleneck)

        self.up_blocks = nn.ModuleList()
        for skip_channels in reversed(self.skip_channels):
            self.up_blocks.append(
                _ResUnetUpBlock(
                    in_channels,
                    skip_channels,
                    skip_channels,
                    norm_layer=norm_layer,
                    use_bias=use_bias,
                    use_dropout=use_dropout,
                )
            )
            in_channels = skip_channels

        self.head = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh(),
        )

    def forward(self, input):
        """Standard forward."""
        x = self.stem(input)
        skips = []
        for down_block in self.down_blocks:
            skip, x = down_block(x)
            skips.append(skip)

        x = self.bottleneck(x)

        for up_block, skip in zip(self.up_blocks, reversed(skips)):
            x = up_block(x, skip)

        return self.head(x)


class _MobileNetConvBNReLU(nn.Module):
    """Conv-BN-ReLU block used by the MobileNet-style generator."""

    def __init__(self, input_nc, output_nc, kernel_size=3, stride=1, groups=1, norm_layer=nn.BatchNorm2d, activation=True):
        super(_MobileNetConvBNReLU, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        padding = kernel_size // 2
        layers = [
            nn.Conv2d(input_nc, output_nc, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=use_bias),
            norm_layer(output_nc),
        ]
        if activation:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class _MobileNetInvertedResidual(nn.Module):
    """Lightweight inverted residual block."""

    def __init__(self, input_nc, output_nc, stride=1, expand_ratio=2.0, norm_layer=nn.BatchNorm2d):
        super(_MobileNetInvertedResidual, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        hidden_nc = max(1, int(round(input_nc * expand_ratio)))
        self.use_residual = stride == 1 and input_nc == output_nc
        layers = []
        if hidden_nc != input_nc:
            layers += [
                nn.Conv2d(input_nc, hidden_nc, kernel_size=1, stride=1, padding=0, bias=use_bias),
                norm_layer(hidden_nc),
                nn.ReLU(inplace=True),
            ]
        layers += [
            nn.Conv2d(hidden_nc, hidden_nc, kernel_size=3, stride=stride, padding=1, groups=hidden_nc, bias=use_bias),
            norm_layer(hidden_nc),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_nc, output_nc, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(output_nc),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return out


class _MobileNetDownBlock(nn.Module):
    """MobileNet-style encoder block with strided downsampling."""

    def __init__(self, input_nc, output_nc, expand_ratio=2.0, norm_layer=nn.BatchNorm2d):
        super(_MobileNetDownBlock, self).__init__()
        self.block = nn.Sequential(
            _MobileNetInvertedResidual(input_nc, output_nc, stride=2, expand_ratio=expand_ratio, norm_layer=norm_layer),
            _MobileNetInvertedResidual(output_nc, output_nc, stride=1, expand_ratio=expand_ratio, norm_layer=norm_layer),
        )

    def forward(self, x):
        return self.block(x)


class _MobileNetUpBlock(nn.Module):
    """MobileNet-style decoder block with skip fusion."""

    def __init__(self, input_nc, skip_nc, output_nc, expand_ratio=2.0, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(_MobileNetUpBlock, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        fused_nc = input_nc + skip_nc
        hidden_nc = max(output_nc, int(round(output_nc * expand_ratio)))
        layers = [
            nn.Conv2d(fused_nc, output_nc, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(output_nc),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_nc, output_nc, kernel_size=3, stride=1, padding=1, groups=output_nc, bias=use_bias),
            norm_layer(output_nc),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_nc, hidden_nc, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(hidden_nc),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_nc, output_nc, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(output_nc),
            nn.ReLU(inplace=True),
        ]
        if use_dropout:
            layers.append(nn.Dropout2d(0.2))
        self.block = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class MobileNetGenerator(nn.Module):
    """A lightweight MobileNet-style encoder-decoder generator."""

    def __init__(self, input_nc, output_nc, ngf=64, width_mult=0.5, bottleneck_blocks=2, num_downs=4, expand_ratio=2.0, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(MobileNetGenerator, self).__init__()
        if num_downs < 2:
            raise ValueError("num_downs for MobileNetGenerator must be >= 2")

        def scale(channels):
            return max(8, int(round(channels * width_mult)))

        stem_nc = scale(max(8, ngf // 2))
        self.stem = _MobileNetConvBNReLU(input_nc, stem_nc, kernel_size=3, stride=1, norm_layer=norm_layer)

        encoder_base = [scale(v) for v in [ngf // 2, ngf, ngf + ngf // 2, ngf * 2]]
        while len(encoder_base) < num_downs:
            encoder_base.append(scale(encoder_base[-1] + ngf // 2))
        self.encoder_channels = encoder_base[:num_downs]

        self.down_blocks = nn.ModuleList()
        in_channels = stem_nc
        for out_channels in self.encoder_channels:
            self.down_blocks.append(_MobileNetDownBlock(in_channels, out_channels, expand_ratio=expand_ratio, norm_layer=norm_layer))
            in_channels = out_channels

        bottleneck = []
        for _ in range(bottleneck_blocks):
            bottleneck.append(_MobileNetInvertedResidual(in_channels, in_channels, stride=1, expand_ratio=expand_ratio, norm_layer=norm_layer))
        self.bottleneck = nn.Sequential(*bottleneck)

        self.up_blocks = nn.ModuleList()
        for skip_channels in reversed([stem_nc] + self.encoder_channels[:-1]):
            self.up_blocks.append(_MobileNetUpBlock(in_channels, skip_channels, skip_channels, expand_ratio=expand_ratio, norm_layer=norm_layer, use_dropout=use_dropout))
            in_channels = skip_channels

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, output_nc, kernel_size=1, stride=1, padding=0),
            nn.Tanh(),
        )

    def forward(self, input):
        x = self.stem(input)
        skips = [x]
        for down_block in self.down_blocks:
            x = down_block(x)
            skips.append(x)

        x = self.bottleneck(x)

        for up_block, skip in zip(self.up_blocks, reversed(skips[:-1])):
            x = up_block(x, skip)

        return self.head(x)


class _ResUnetDownBlock(nn.Module):
    """Residual encoder block with strided downsampling."""

    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, use_bias=False):
        super(_ResUnetDownBlock, self).__init__()
        self.res_block = ResnetBlock(
            input_nc,
            padding_type="reflect",
            norm_layer=norm_layer,
            use_dropout=False,
            use_bias=use_bias,
        )
        self.down = nn.Sequential(
            nn.Conv2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(output_nc),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, x):
        skip = self.res_block(x)
        down = self.down(skip)
        return skip, down


class _ResUnetUpBlock(nn.Module):
    """Residual decoder block with skip fusion."""

    def __init__(self, input_nc, skip_nc, output_nc, norm_layer=nn.BatchNorm2d, use_bias=False, use_dropout=False):
        super(_ResUnetUpBlock, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(output_nc),
            nn.ReLU(True),
        )
        fused_nc = output_nc + skip_nc
        self.fuse = nn.Sequential(
            nn.Conv2d(fused_nc, output_nc, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(output_nc),
            nn.ReLU(True),
            ResnetBlock(
                output_nc,
                padding_type="reflect",
                norm_layer=norm_layer,
                use_dropout=use_dropout,
                use_bias=use_bias,
            ),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            min_h = min(x.shape[-2], skip.shape[-2])
            min_w = min(x.shape[-1], skip.shape[-1])
            x = x[:, :, :min_h, :min_w]
            skip = skip[:, :, :min_h, :min_w]
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
    X -------------------identity----------------------
    |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:  # add skip connections
            return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias),
        ]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)
