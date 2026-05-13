import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
from . import networks


class OptionalLosses(nn.Module):
    def __init__(self, opt, netD, device):
        super(OptionalLosses, self).__init__()
        self.opt = opt
        self.netD = netD
        self.device = device
        self.use_perception_loss = getattr(opt, "use_perception_loss", False)
        self.use_feature_matching_loss = getattr(opt, "use_feature_matching_loss", False)
        self.use_gradient_loss = getattr(opt, "use_gradient_loss", False)
        self.vgg_features = None

        if self.use_perception_loss:
            self.vgg_features = self._build_vgg_features()
            self.vgg_features.eval()
            for parameter in self.vgg_features.parameters():
                parameter.requires_grad = False

    def _build_vgg_features(self):
        try:
            from torchvision import models as torchvision_models
            from torchvision.models import VGG16_Weights

            vgg = torchvision_models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            features = nn.Sequential(*list(vgg.features.children())[:16])
        except Exception:
            features = nn.Sequential()
        return features.to(self.device)

    def _normalize_for_vgg(self, image):
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        image = (image + 1.0) * 0.5
        mean = torch.tensor([0.485, 0.456, 0.406], device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
        return (image - mean) / std

    def _get_discriminator_layers(self):
        if hasattr(self.netD, "model") and isinstance(self.netD.model, nn.Sequential):
            return list(self.netD.model.children())
        if hasattr(self.netD, "net") and isinstance(self.netD.net, nn.Sequential):
            return list(self.netD.net.children())
        if isinstance(self.netD, nn.Sequential):
            return list(self.netD.children())
        return None

    def _discriminator_feature_maps(self, input_tensor):
        layers = self._get_discriminator_layers()
        if layers is None or len(layers) < 2:
            return []

        current = input_tensor
        for layer in layers[:-1]:
            current = layer(current)
        return [current]

    def perception_loss(self, fake_image, real_image):
        if not self.use_perception_loss or self.vgg_features is None or len(self.vgg_features) == 0:
            return fake_image.new_tensor(0.0)

        fake_input = self._normalize_for_vgg(fake_image)
        real_input = self._normalize_for_vgg(real_image)
        fake_features = self.vgg_features(fake_input)
        with torch.no_grad():
            real_features = self.vgg_features(real_input)
        return F.l1_loss(fake_features, real_features)

    def feature_matching_loss(self, fake_ab, real_ab):
        if not self.use_feature_matching_loss:
            return fake_ab.new_tensor(0.0)

        fake_features = self._discriminator_feature_maps(fake_ab)
        if not fake_features:
            return fake_ab.new_tensor(0.0)

        with torch.no_grad():
            real_features = self._discriminator_feature_maps(real_ab)

        if len(real_features) == 0:
            return fake_ab.new_tensor(0.0)

        return F.l1_loss(fake_features[0], real_features[0])

    def gradient_loss(self, fake_image, real_image):
        if not self.use_gradient_loss:
            return fake_image.new_tensor(0.0)

        fake_dx = fake_image[:, :, :, 1:] - fake_image[:, :, :, :-1]
        fake_dy = fake_image[:, :, 1:, :] - fake_image[:, :, :-1, :]
        real_dx = real_image[:, :, :, 1:] - real_image[:, :, :, :-1]
        real_dy = real_image[:, :, 1:, :] - real_image[:, :, :-1, :]

        loss_x = F.l1_loss(fake_dx, real_dx)
        loss_y = F.l1_loss(fake_dy, real_dy)
        return 0.5 * (loss_x + loss_y)


class Pix2PixModel(BaseModel):
    """This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm="batch", netG="unet_256", dataset_mode="aligned")
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode="vanilla")
            parser.add_argument("--lambda_L1", type=float, default=100.0, help="weight for L1 loss")
            parser.add_argument("--lambda_Perception", type=float, default=0.0, help="weight for perception loss")
            parser.add_argument("--lambda_FM", type=float, default=0.0, help="weight for feature matching loss")
            parser.add_argument("--lambda_Gradient", type=float, default=0.0, help="weight for gradient loss")
            parser.add_argument('--pre_train', action='store_true', help='pre-train the generator with only L1 loss while the discriminator is frozen')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ["G_GAN", "G_L1", "D_real", "D_fake"]
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ["real_A", "fake_B", "real_B"]
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ["G", "D"]
        else:  # during test time, only load G
            self.model_names = ["G"]
        self.device = opt.device
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # move to the device for custom loss
            self.criterionL1 = torch.nn.L1Loss()
            # 在 Pix2PixModel.__init__ 中，实例化 optional_losses 之前添加：
            if opt.lambda_Perception > 0:
                opt.use_perception_loss = True
            if opt.lambda_FM > 0:
                opt.use_feature_matching_loss = True
            if opt.lambda_Gradient > 0:
                opt.use_gradient_loss = True
            self.optional_losses = OptionalLosses(opt, self.netD, self.device)
            if opt.lambda_Perception > 0.0:
                self.loss_names.append("G_Perception")
            if opt.lambda_FM > 0.0:
                self.loss_names.append("G_FM")
            if opt.lambda_Gradient > 0.0:
                self.loss_names.append("G_Gradient")
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr_G, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr_D, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == "AtoB"
        self.real_A = input["A" if AtoB else "B"].to(self.device)
        self.real_B = input["B" if AtoB else "A"].to(self.device)
        self.image_paths = input["A_paths" if AtoB else "B_paths"]

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # Optional auxiliary losses
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        if self.opt.lambda_Perception > 0.0:
            self.loss_G_Perception = self.optional_losses.perception_loss(self.fake_B, self.real_B) * self.opt.lambda_Perception
            self.loss_G = self.loss_G + self.loss_G_Perception
        if self.opt.lambda_FM > 0.0:
            self.loss_G_FM = self.optional_losses.feature_matching_loss(fake_AB, torch.cat((self.real_A, self.real_B), 1)) * self.opt.lambda_FM
            self.loss_G = self.loss_G + self.loss_G_FM
        if self.opt.lambda_Gradient > 0.0:
            self.loss_G_Gradient = self.optional_losses.gradient_loss(self.fake_B, self.real_B) * self.opt.lambda_Gradient
            self.loss_G = self.loss_G + self.loss_G_Gradient
        # combine loss and calculate gradients
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                     # 计算 fake_B = G(A)

        # ============================================================
        # 预训练阶段：只更新 G，使用纯 L1（及可能的辅助损失），D 冻结
        # ============================================================
        if self.opt.pre_train:
            # 冻结判别器
            self.set_requires_grad(self.netD, False)
            # 判别器相关损失置零（用于日志打印）
            self.loss_D_real = torch.tensor(0.0, device=self.device)
            self.loss_D_fake = torch.tensor(0.0, device=self.device)
            self.loss_D = torch.tensor(0.0, device=self.device)
            self.loss_G_GAN = torch.tensor(0.0, device=self.device)

            for _ in range(self.opt.G_rounds):
                self.optimizer_G.zero_grad()

                # 若 G_rounds > 1，重新生成 fake_B
                if self.opt.G_rounds > 1:
                    self.fake_B = self.netG(self.real_A)

                # L1 损失
                self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
                self.loss_G = self.loss_G_L1

                # 可选的辅助损失（感知、特征匹配、梯度）
                if self.opt.lambda_Perception > 0.0:
                    self.loss_G_Perception = self.optional_losses.perception_loss(
                        self.fake_B, self.real_B) * self.opt.lambda_Perception
                    self.loss_G += self.loss_G_Perception
                if self.opt.lambda_FM > 0.0:
                    fake_ab = torch.cat((self.real_A, self.fake_B), 1)
                    real_ab = torch.cat((self.real_A, self.real_B), 1)
                    self.loss_G_FM = self.optional_losses.feature_matching_loss(fake_ab, real_ab) * self.opt.lambda_FM
                    self.loss_G += self.loss_G_FM
                if self.opt.lambda_Gradient > 0.0:
                    self.loss_G_Gradient = self.optional_losses.gradient_loss(
                        self.fake_B, self.real_B) * self.opt.lambda_Gradient
                    self.loss_G += self.loss_G_Gradient

                self.loss_G.backward()
                self.optimizer_G.step()

            return  # 预训练阶段到此结束

        # ============================================================
        # 正常对抗训练（原逻辑）
        # ============================================================
        # 更新 D
        for _ in range(self.opt.D_rounds):
            self.set_requires_grad(self.netD, True)
            self.netD.train()
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()

        # 更新 G
        self.set_requires_grad(self.netD, False)
        for _ in range(self.opt.G_rounds):
            self.optimizer_G.zero_grad()
            if self.opt.G_rounds > 1:
                self.fake_B = self.netG(self.real_A)
            self.backward_G()
            self.optimizer_G.step()