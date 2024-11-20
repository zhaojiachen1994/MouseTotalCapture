import numpy as np
import torch
import torch.nn as nn
from icecream import ic
from mmpose.models.builder import build_loss
from mmpose.models import HEADS
import time

def _make_radial_window(width, height, cx, cy, fn, window_width=10.0):
    """
    Returns a grid, where grid[i,j] = fn((i**2 + j**2)**0.5)

    :param width: Width of the grid to return
    :param height: Height of the grid to return
    :param cx: x center
    :param cy: y center
    :param fn: The function to apply
    :return:
    """
    # The length of cx and cy is the number of channels we need
    dev = cx.device
    channels = cx.size(0)

    # Explicitly tile cx and cy, ready for computing the distance matrix below, because pytorch doesn't broadcast very well
    # Make the shape [channels, height, width]
    cx = cx.repeat(height, width, 1).permute(2, 0, 1)
    cy = cy.repeat(height, width, 1).permute(2, 0, 1)

    # Compute a grid where dist[i,j] = (i-cx)**2 + (j-cy)**2, need to view and repeat to tile and make shape [channels, height, width]
    xs = torch.arange(width).view((1, width)).repeat(channels, height, 1).float().to(dev)
    ys = torch.arange(height).view((height, 1)).repeat(channels, 1, width).float().to(dev)
    delta_xs = xs - cx
    delta_ys = ys - cy
    dists = torch.sqrt((delta_ys ** 2) + (delta_xs ** 2))

    # apply the function to the grid and return it
    return fn(dists, window_width)


def _parzen_scalar(delta, width):
    """For reference"""
    del_ovr_wid = math.abs(delta) / width
    if delta <= width / 2.0:
        return 1 - 6 * (del_ovr_wid ** 2) * (1 - del_ovr_wid)
    elif delta <= width:
        return 2 * (1 - del_ovr_wid) ** 3


def _parzen_torch(dists, width):
    """
    A PyTorch version of the parzen window that works a grid of distances about some center point.
    See _parzen_scalar to see the

    :param dists: The grid of distances
    :param window: The width of the parzen window
    :return: A 2d grid, who's values are a (radial) parzen window
    """
    hwidth = width / 2.0
    del_ovr_width = dists / hwidth

    near_mode = (dists <= hwidth / 2.0).float()
    in_tail = ((dists > hwidth / 2.0) * (dists <= hwidth)).float()

    return near_mode * (1 - 6 * (del_ovr_width ** 2) * (1 - del_ovr_width)) \
        + in_tail * (2 * ((1 - del_ovr_width) ** 3))


def _uniform_window(dists, width):
    """
    A (radial) uniform window function
    :param dists: A grid of distances
    :param width: A width for the window
    :return: A 2d grid, who's values are 0 or 1 depending on if it's in the window or not
    """
    hwidth = width / 2.0
    return (dists <= hwidth).float()


def _identity_window(dists, width):
    """
    An "identity window". (I.e. a "window" which when multiplied by, will not change the input).
    """
    return torch.ones(dists.size())


class SoftArgmax2D(torch.nn.Module):
    """
    adafuse/lib/models/soft_argmax
    Implementation of a 1d soft arg-max function as an nn.Module, so that we can differentiate through arg-max operations.
    """

    def __init__(self, base_index=0, step_size=1, window_fn=None, window_width=10, softmax_temp=1.0):
        """
        The "arguments" are base_index, base_index+step_size, base_index+2*step_size, ... and so on for
        arguments at indices 0, 1, 2, ....

        Assumes that the input to this layer will be a batch of 3D tensors (so a 4D tensor).
        For input shape (B, C, W, H), we apply softmax across the W and H dimensions.
        We use a softmax, over dim 2, expecting a 3D input, which is created by reshaping the input to (B, C, W*H)
        (This is necessary because true 2D softmax doesn't natively exist in PyTorch...

        :param base_index: Remember a base index for 'indices' for the input
        :param step_size: Step size for 'indices' from the input
        :param window_function: Specify window function, that given some center point produces a window 'landscape'. If
            a window function is specified then before applying "soft argmax" we multiply the input by a window centered
            at the true argmax, to enforce the input to soft argmax to be unimodal. Window function should be specified
            as one of the following options: None, "Parzen", "Uniform"
        :param window_width: How wide do we want the window to be? (If some point is more than width/2 distance from the
            argmax then it will be zeroed out for the soft argmax calculation, unless, window_fn == None)
        """
        super(SoftArgmax2D, self).__init__()
        self.base_index = base_index
        self.step_size = step_size
        self.softmax = torch.nn.Softmax(dim=2)
        self.softmax_temp = softmax_temp
        self.window_type = window_fn
        self.window_width = window_width
        self.window_fn = _identity_window
        if window_fn == "Parzen":
            self.window_fn = _parzen_torch
        elif window_fn == "Uniform":
            self.window_fn = _uniform_window

    def _softmax_2d(self, x, temp):
        """
        For the lack of a true 2D softmax in pytorch, we reshape each image from (C, W, H) to (C, W*H) and then
        apply softmax, and then restore the original shape.

        :param x: A 4D tensor of shape (B, C, W, H) to apply softmax across the W and H dimensions
        :param temp: A scalar temperature to apply as part of the softmax function
        :return: Softmax(x, dims=(2,3))
        """
        B, C, W, H = x.size()
        x_flat = x.view((B, C, W * H)) / temp
        x_softmax = self.softmax(x_flat)
        return x_softmax.view((B, C, W, H))

    def forward(self, x, out_smax=False, hardmax=False):
        """
        Compute the forward pass of the 1D soft arg-max function as defined below:

        SoftArgMax2d(x) = (\sum_i \sum_j (i * softmax2d(x)_ij), \sum_i \sum_j (j * softmax2d(x)_ij))

        :param x: The input to the soft arg-max layer
        :return: [batch_size, 3, channels]
        """
        # Compute windowed softmax
        # Compute windows using a batch_size of "batch_size * channels"
        dev = x.device
        batch_size, channels, height, width = x.size()
        maxv, argmax = torch.max(x.view(batch_size * channels, -1), dim=1)
        argmax_x, argmax_y = torch.remainder(argmax, width).float(), torch.floor(
            torch.div(argmax.float(), float(width)))
        windows = _make_radial_window(width, height, argmax_x, argmax_y, self.window_fn, self.window_width)
        windows = windows.view(batch_size, channels, height, width).to(dev)
        smax = self._softmax_2d(x, self.softmax_temp) * windows
        smax = smax / torch.sum(smax.view(batch_size, channels, -1), dim=2).view(batch_size, channels, 1, 1)

        x_max = argmax_x.view(batch_size, 1, channels)
        y_max = argmax_y.view(batch_size, 1, channels)
        ones = torch.ones_like(x_max)
        xys_max = torch.cat([x_max, y_max, ones], dim=1)
        xys_max = xys_max.view(batch_size, 3, channels)
        # compute x index (sum over y axis, produce with indices and then sum over x axis for the expectation)
        x_end_index = self.base_index + width * self.step_size
        x_indices = torch.arange(start=self.base_index, end=x_end_index, step=self.step_size).type_as(smax)
        x_coords = torch.sum(torch.sum(smax, 2) * x_indices, 2)

        # compute y index (sum over x axis, produce with indices and then sum over y axis for the expectation)
        y_end_index = self.base_index + height * self.step_size
        y_indices = torch.arange(start=self.base_index, end=y_end_index, step=self.step_size).type_as(smax)
        y_coords = torch.sum(torch.sum(smax, 3) * y_indices, 2)

        # For debugging (testing if it's actually like the argmax?)
        # argmax_x = argmax_x.view(batch_size, channels)
        # argmax_y = argmax_y.view(batch_size, channels)
        # print("X err in soft argmax: {err}".format(err=torch.mean(torch.abs(argmax_x - x_coords))))
        # print("Y err in soft argmax: {err}".format(err=torch.mean(torch.abs(argmax_y - y_coords))))

        # Put the x coords and y coords along with 1s of (shape (B,C)) into an output with shape (B,C,3)
        xs = torch.unsqueeze(x_coords, 2)
        ys = torch.unsqueeze(y_coords, 2)
        ones = torch.ones_like(xs)
        xys = torch.cat([xs, ys, ones], dim=2)
        xys = xys.view(batch_size, channels, 3)
        xys = xys.permute(0, 2, 1).contiguous()  # (batch, 3, njoint)

        maxv = maxv.view(batch_size, 1, channels)
        zero_xys = torch.zeros_like(xys)
        zero_xys[:, 2, :] = 1
        xys = torch.where(maxv > 0.01, xys, zero_xys)

        if hardmax:
            if out_smax:
                return xys_max, maxv.view(batch_size, channels), smax
            else:
                return xys_max, maxv.view(batch_size, channels)

        if out_smax:
            return xys, maxv.view(batch_size, channels), smax
        else:
            return xys, maxv.view(batch_size, channels)


@HEADS.register_module()
class TriangulateHead(nn.Module):
    def __init__(self, img_shape=[256, 256], heatmap_shape=[64, 64],
                 softmax_heatmap=True,
                 loss_3d_sup=None, tr_loss=None,
                 det_conf_thr=None, train_cfg=None, test_cfg=None):
        super().__init__()
        [self.h_img, self.w_img] = img_shape
        [self.h_map, self.w_map] = heatmap_shape
        self.det_conf_thr = det_conf_thr  # weather use the 2d detect confidence to mask the fail detection points
        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg
        if loss_3d_sup is not None:
            self.sup_loss = build_loss(loss_3d_sup)
        if tr_loss is not None:
            self.tr_loss = build_loss(tr_loss)

        self.smax = SoftArgmax2D(window_fn='Uniform', window_width=5 * 3, softmax_temp=0.05)

    def forward(self, heatmap, proj_mats, vis_prob=None, reproject=True, on='cpu'):
        """
        :param heatmap: [num_cams, num_joints, h_heatmap, w_heatmap]
        :param proj_mats: [num_cams, 3, 4]
        :param vis_prob: [num_cams, n_joints]
        :param reproject:
        :return:
            kpt_3d: triangulation results, keypoint 3d coordinates, [n_joints, 3]
            res_triang: triangulation residual, [n_joints]
        """
        device= heatmap.device
        n_cams = proj_mats.shape[0]
        n_joints = heatmap.shape[1]
        kp_2d_hm, maxv = self.smax(heatmap)  # kp_2d_hm: [bs*num_cams, 3, num_joints], maxv: [bs*num_cams, num_joints]
        kp_2d_hm[:, -1, :] = maxv
        kp_2d_hm = kp_2d_hm.permute(0, 2, 1)
        kp_2d_croped = torch.zeros_like(kp_2d_hm, dtype=float)
        kp_2d_croped[:, :, 0] = kp_2d_hm[:, :, 0] * self.h_img / self.h_map
        kp_2d_croped[:, :, 1] = kp_2d_hm[:, :, 1] * self.w_img / self.w_map
        kp_2d_croped[:, :, 2] = kp_2d_hm[:, :, 2]
        points = kp_2d_croped[:, :, :2]
        det_conf = kp_2d_croped[:, :, 2].detach()    # [n_cams, n_joints]
        det_conf[det_conf < self.det_conf_thr] = 0.0
        if vis_prob is None:
            vis_prob = torch.ones([n_cams, n_joints], dtype=torch.float32, device=heatmap.device)
        confidences = det_conf * vis_prob
        # confidences = torch.ones([n_cams, n_joints], dtype=torch.float32, device=heatmap.device)
        points = points.permute(1, 0, 2).unsqueeze(-1)  # [n_cams, n_joints, 2]->[n_joints, n_cams,  2, 1]
        confidences = confidences.permute(1, 0).unsqueeze(-1).unsqueeze(-1)  # [n_cams, n_joints]->[n_joints, n_cams, 1, 1]
        proj_mats = proj_mats.unsqueeze(0)  # [n_cams, 3, 4] -> [1, n_cams, 3, 4]

        A = proj_mats[:, :, 2:3] * points
        A -= proj_mats[:, :, :2]
        A *= confidences
        A = A.reshape(n_joints, -1, 4)

        try:
            if on == 'cpu':
                A = A.to('cpu')
                u, s, vh = torch.svd(A)
                u = u.to(device)
                s = s.to(device)
                vh = vh.to(device)
            else:
                u, s, vh = torch.svd(A)

            point_3d_homo = -vh[:, :, 3]
            point_3d = point_3d_homo[:, :3] / (point_3d_homo[:, 3].unsqueeze(1))
            res_triang = s[:, -1:]
            point_3d = torch.cat([point_3d, res_triang], dim=1)
        except:
            print(f"skipping batch due to ill-condition")
            point_3d = torch.tensor([92, 4])
        return point_3d, points  # [n_joints, 4] [x, y, z, residual]

    def get_sup_loss(self, output, target, target_visible):
        losses = dict()
        losses['sup_3d_loss'] = self.sup_loss(output, target, target_visible)
        return losses

    def get_unSup_loss(self, res_triang, target_visible):
        losses = dict()
        losses['unSup_3d_loss'] = self.tr_loss(res_triang, target_visible)
        return losses
