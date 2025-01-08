import torch
import torch.nn.functional as F
from main import TrainingStage
from utils.data import get_lossmask


class HighFrequencyLoss(torch.nn.Module):
    def __init__(self, weight=1.):
        super().__init__()
        self._weight = weight

    def compute_gradients(self, image):
        """
        Compute image gradients using Sobel filters.
        :param image: Input image tensor of shape (B, C, H, W).
        :return: Gradients in x and y directions (B, C, H, W).
        """
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32, device=image.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]], dtype=torch.float32, device=image.device).view(1, 1, 3, 3)

        # Apply Sobel filters to compute gradients
        grad_x = F.conv2d(image, sobel_x, padding=1, groups=image.shape[1])
        grad_y = F.conv2d(image, sobel_y, padding=1, groups=image.shape[1])

        return grad_x, grad_y

    def forward(self, data_dict, training_stage):
        if training_stage not in [TrainingStage.OPTIMIZE_OPACITY,
                                  TrainingStage.OPTIMIZE_GAUSSIANS,
                                  TrainingStage.FINETUNE_TEXTURE,
                                  TrainingStage.INIT_TEXTURE,
                                  TrainingStage.FINETUNE_POSE]:
            return 0
        if training_stage in [TrainingStage.FINETUNE_TEXTURE,
                              TrainingStage.INIT_TEXTURE]:
            use_mask = True
        else:
            use_mask = False
        gt_image = data_dict["rgb_image"]
        render_image = data_dict["rasterization"]

        if use_mask:
            loss_mask = get_lossmask(data_dict["merged_mask"][:, 0:1, ...], data_dict["mask_image"])
            gt_image = gt_image * loss_mask
            render_image = render_image * loss_mask

        # Compute gradients for ground truth and rendered images
        gt_grad_x, gt_grad_y = self.compute_gradients(gt_image)
        render_grad_x, render_grad_y = self.compute_gradients(render_image)

        # Compute L2 loss for gradient differences
        grad_loss_x = F.mse_loss(gt_grad_x, render_grad_x)
        grad_loss_y = F.mse_loss(gt_grad_y, render_grad_y)

        # Combine losses for x and y directions
        loss = (grad_loss_x + grad_loss_y) * self._weight
        return loss
