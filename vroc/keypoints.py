import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csgraph, csr_matrix


def filter1D(img, weight, dim, padding_mode="replicate"):
    B, C, D, H, W = img.shape
    N = weight.shape[0]

    padding = torch.zeros(
        6,
    )
    padding[[4 - 2 * dim, 5 - 2 * dim]] = N // 2
    padding = padding.long().tolist()

    view = torch.ones(
        5,
    )
    view[dim + 2] = -1
    view = view.long().tolist()

    return F.conv3d(
        F.pad(img.view(B * C, 1, D, H, W), padding, mode=padding_mode),
        weight.view(view),
    ).view(B, C, D, H, W)


def smooth(img, sigma):
    device = img.device

    sigma = torch.tensor([sigma], device=device)
    N = torch.ceil(sigma * 3.0 / 2.0).long().item() * 2 + 1

    weight = torch.exp(
        -torch.pow(torch.linspace(-(N // 2), N // 2, N, device=device), 2)
        / (2 * torch.pow(sigma, 2))
    )
    weight /= weight.sum()

    img = filter1D(img, weight, 0)
    img = filter1D(img, weight, 1)
    img = filter1D(img, weight, 2)

    return img


def structure_tensor(img, sigma):
    B, C, D, H, W = img.shape

    struct = []
    for i in range(C):
        for j in range(i, C):
            struct.append(smooth((img[:, i, ...] * img[:, j, ...]).unsqueeze(1), sigma))

    return torch.cat(struct, dim=1)


def invert_structure_tensor(struct):
    a = struct[:, 0, ...]
    b = struct[:, 1, ...]
    c = struct[:, 2, ...]
    e = struct[:, 3, ...]
    f = struct[:, 4, ...]
    i = struct[:, 5, ...]

    A = e * i - f * f
    B = -b * i + c * f
    C = b * f - c * e
    E = a * i - c * c
    F = -a * f + b * c
    I = a * e - b * b

    det = (a * A + b * B + c * C).unsqueeze(1)

    struct_inv = (1.0 / det) * torch.stack([A, B, C, E, F, I], dim=1)

    return struct_inv


def kpts_pt(kpts_world, shape, align_corners=None):
    device = kpts_world.device
    D, H, W = shape

    kpts_pt_ = (
        kpts_world.flip(-1) / (torch.tensor([W, H, D], device=device) - 1)
    ) * 2 - 1
    if not align_corners:
        kpts_pt_ *= (torch.tensor([W, H, D], device=device) - 1) / torch.tensor(
            [W, H, D], device=device
        )

    return kpts_pt_


def kpts_world(kpts_pt, shape, align_corners=None):
    device = kpts_pt.device
    D, H, W = shape

    if not align_corners:
        kpts_pt /= (torch.tensor([W, H, D], device=device) - 1) / torch.tensor(
            [W, H, D], device=device
        )
    kpts_world_ = (
        ((kpts_pt + 1) / 2) * (torch.tensor([W, H, D], device=device) - 1)
    ).flip(-1)

    return kpts_world_


def foerstner_kpts(img, mask, sigma=1.4, d=9, thresh=1e-8):
    _, _, D, H, W = img.shape
    device = img.device

    filt = torch.tensor(
        [1.0 / 12.0, -8.0 / 12.0, 0.0, 8.0 / 12.0, -1.0 / 12.0], device=device
    )
    grad = torch.cat(
        [filter1D(img, filt, 0), filter1D(img, filt, 1), filter1D(img, filt, 2)], dim=1
    )

    struct_inv = invert_structure_tensor(structure_tensor(grad, sigma))

    distinctiveness = 1.0 / (
        struct_inv[:, 0, ...] + struct_inv[:, 3, ...] + struct_inv[:, 5, ...]
    ).unsqueeze(1)

    pad1 = d // 2
    pad2 = d - pad1 - 1

    maxfeat = F.max_pool3d(
        F.pad(distinctiveness, (pad2, pad1, pad2, pad1, pad2, pad1)), d, stride=1
    )

    structure_element = torch.tensor(
        [
            [[0.0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        ],
        device=device,
    )

    mask_eroded = (
        1
        - F.conv3d(
            1 - mask.float(), structure_element.unsqueeze(0).unsqueeze(0), padding=1
        ).clamp_(0, 1)
    ).bool()

    kpts = torch.nonzero(
        mask_eroded & (maxfeat == distinctiveness) & (distinctiveness >= thresh)
    ).unsqueeze(0)[:, :, 2:]

    return kpts_pt(kpts, (D, H, W), align_corners=True)


def mindssc(img, delta=1, sigma=1):
    device = img.device

    # define start and end locations for self-similarity pattern
    six_neighbourhood = torch.tensor(
        [[0, 1, 1], [1, 1, 0], [1, 0, 1], [1, 1, 2], [2, 1, 1], [1, 2, 1]],
        dtype=torch.float,
        device=device,
    )

    # squared distances
    dist = pdist(six_neighbourhood.unsqueeze(0)).squeeze(0)

    # define comparison mask
    x, y = torch.meshgrid(
        torch.arange(6, device=device), torch.arange(6, device=device)
    )
    mask = (x > y).view(-1) & (dist == 2).view(-1)

    # build kernel
    idx_shift1 = (
        six_neighbourhood.unsqueeze(1).repeat(1, 6, 1).view(-1, 3)[mask, :].long()
    )
    idx_shift2 = (
        six_neighbourhood.unsqueeze(0).repeat(6, 1, 1).view(-1, 3)[mask, :].long()
    )
    mshift1 = torch.zeros((12, 1, 3, 3, 3), device=device)
    mshift1.view(-1)[
        torch.arange(12, device=device) * 27
        + idx_shift1[:, 0] * 9
        + idx_shift1[:, 1] * 3
        + idx_shift1[:, 2]
    ] = 1
    mshift2 = torch.zeros((12, 1, 3, 3, 3), device=device)
    mshift2.view(-1)[
        torch.arange(12, device=device) * 27
        + idx_shift2[:, 0] * 9
        + idx_shift2[:, 1] * 3
        + idx_shift2[:, 2]
    ] = 1
    rpad = nn.ReplicationPad3d(delta)

    # compute patch-ssd
    ssd = smooth(
        (
            (
                F.conv3d(rpad(img), mshift1, dilation=delta)
                - F.conv3d(rpad(img), mshift2, dilation=delta)
            )
            ** 2
        ),
        sigma,
    )

    # MIND equation
    mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
    mind_var = torch.mean(mind, 1, keepdim=True)
    mind_var = torch.clamp(mind_var, mind_var.mean() * 0.001, mind_var.mean() * 1000)
    mind /= mind_var
    mind = torch.exp(-mind)

    return mind


def minimum_spanning_tree(dist):
    device = dist.device
    N = dist.shape[1]

    mst = csgraph.minimum_spanning_tree(csr_matrix(dist[0].cpu().numpy()))
    bfo = csgraph.breadth_first_order(mst, 0, directed=False)
    edges = (
        torch.tensor([bfo[1][bfo[0]][1:], bfo[0][1:]], dtype=torch.long, device=device)
        .t()
        .view(1, -1, 2)
    )

    level = torch.zeros((1, N, 1), dtype=torch.long, device=device)
    for i in range(N - 1):
        level[0, edges[0, i, 1], 0] = level[0, edges[0, i, 0], 0] + 1

    idx = edges[0, :, 1].sort()[1]
    edges = edges[:, idx, :]

    return edges, level


def minconv(input):
    device = input.device
    disp_width = input.shape[-1]

    disp1d = torch.linspace(
        -(disp_width // 2), disp_width // 2, disp_width, device=device
    )
    regular1d = (disp1d.view(1, -1) - disp1d.view(-1, 1)) ** 2

    output = torch.min(
        input.view(-1, disp_width, 1, disp_width, disp_width)
        + regular1d.view(1, disp_width, disp_width, 1, 1),
        1,
    )[0]
    output = torch.min(
        output.view(-1, disp_width, disp_width, 1, disp_width)
        + regular1d.view(1, 1, disp_width, disp_width, 1),
        2,
    )[0]
    output = torch.min(
        output.view(-1, disp_width, disp_width, disp_width, 1)
        + regular1d.view(1, 1, 1, disp_width, disp_width),
        3,
    )[0]

    output = output - (torch.min(output.view(-1, disp_width**3), 1)[0]).view(
        -1, 1, 1, 1
    )

    return output.view_as(input)


def tbp(cost, edges, level, dist):
    marginals = cost
    message = torch.zeros_like(marginals)

    for i in range(level.max(), 0, -1):
        child = edges[0, level[0, 1:, 0] == i, 1]
        parent = edges[0, level[0, 1:, 0] == i, 0]
        weight = dist[0, child, parent].view(-1, 1, 1, 1)

        data = marginals[:, child, :, :, :]
        data_reg = minconv(data * weight) / weight

        message[:, child, :, :, :] = data_reg
        marginals = torch.index_add(marginals, 1, parent, data_reg)

    for i in range(1, level.max() + 1):
        child = edges[0, level[0, 1:, 0] == i, 1]
        parent = edges[0, level[0, 1:, 0] == i, 0]
        weight = dist[0, child, parent].view(-1, 1, 1, 1)

        data = (
            marginals[:, parent, :, :, :]
            - message[:, child, :, :, :]
            + message[:, parent, :, :, :]
        )
        data_reg = minconv(data * weight) / weight

        message[:, child, :, :, :] = data_reg

    marginals += message

    return marginals


def mean_filter(img, r):
    device = img.device

    weight = torch.ones((2 * r + 1,), device=device) / (2 * r + 1)

    img = filter1D(img, weight, 0)
    img = filter1D(img, weight, 1)
    img = filter1D(img, weight, 2)

    return img


def pdist(x, p=2):
    if p == 1:
        dist = torch.abs(x.unsqueeze(2) - x.unsqueeze(1)).sum(dim=3)
    elif p == 2:
        xx = (x**2).sum(dim=2).unsqueeze(2)
        yy = xx.permute(0, 2, 1)
        dist = xx + yy - 2.0 * torch.bmm(x, x.permute(0, 2, 1))
        dist[:, torch.arange(dist.shape[1]), torch.arange(dist.shape[2])] = 0
    return dist


def kpts_dist(kpts, img, beta, k=64):
    device = kpts.device
    B, N, _ = kpts.shape
    _, _, D, H, W = img.shape

    dist = pdist(kpts_world(kpts, (D, H, W), align_corners=True)).sqrt()
    dist[:, torch.arange(dist.shape[1]), torch.arange(dist.shape[2])] = 1e15
    dist[dist < 0.1] = 0.1
    img_mean = mean_filter(img, 2)
    kpts_mean = F.grid_sample(
        img_mean,
        kpts.view(1, 1, 1, -1, 3).to(img_mean.dtype),
        mode="nearest",
        align_corners=True,
    ).view(1, -1, 1)
    dist += pdist(kpts_mean, p=1) / beta

    include_self = False
    ind = (-dist).topk(k + (1 - int(include_self)), dim=-1)[1][
        :, :, 1 - int(include_self) :
    ]
    A = torch.zeros((B, N, N), device=device)
    A[:, torch.arange(N).repeat(k), ind[0].t().contiguous().view(-1)] = 1
    A[:, ind[0].t().contiguous().view(-1), torch.arange(N).repeat(k)] = 1
    dist = A * dist

    return dist


def get_patch(patch_step, patch_radius, shape, device):
    D, H, W = shape

    patch = (
        torch.stack(
            torch.meshgrid(
                torch.arange(0, 2 * patch_radius + 1, patch_step, device=device),
                torch.arange(0, 2 * patch_radius + 1, patch_step, device=device),
                torch.arange(0, 2 * patch_radius + 1, patch_step, device=device),
            ),
            dim=3,
        ).view(1, -1, 3)
        - patch_radius
    )
    patch = flow_pt(patch, (D, H, W), align_corners=True)
    return patch


def ssd(
    kpts_fixed,
    feat_fixed,
    feat_moving,
    disp_radius=16,
    disp_step=2,
    patch_radius=3,
    unroll_step_size=2**6,
):
    device = kpts_fixed.device
    N = kpts_fixed.shape[1]
    _, C, D, H, W = feat_fixed.shape

    patch_step = disp_step  # same stride necessary for fast implementation
    patch = get_patch(patch_step, patch_radius, (D, H, W), device=device)
    patch_width = round(patch.shape[1] ** (1.0 / 3))

    pad = [(patch_width - 1) // 2, (patch_width - 1) // 2 + (1 - patch_width % 2)]

    disp = get_disp(
        disp_step, disp_radius + ((pad[0] + pad[1]) / 2), (D, H, W), device=device
    )
    disp_width = disp_radius * 2 + 1

    cost = torch.zeros(1, N, disp_width, disp_width, disp_width, device=device)
    n = math.ceil(N / unroll_step_size)
    for j in range(n):
        j1 = j * unroll_step_size
        j2 = min((j + 1) * unroll_step_size, N)

        feat_fixed_patch = F.grid_sample(
            feat_fixed,
            kpts_fixed[:, j1:j2, :].view(1, -1, 1, 1, 3) + patch.view(1, 1, -1, 1, 3),
            mode="nearest",
            padding_mode="border",
            align_corners=True,
        )
        feat_moving_disp = F.grid_sample(
            feat_moving,
            kpts_fixed[:, j1:j2, :].view(1, -1, 1, 1, 3) + disp.view(1, 1, -1, 1, 3),
            mode="nearest",
            padding_mode="border",
            align_corners=True,
        )

        fixed_sum = (feat_fixed_patch**2).sum(dim=3).view(C, (j2 - j1), 1, 1, 1)
        moving_sum = (patch_width**3) * F.avg_pool3d(
            (feat_moving_disp**2).view(
                C,
                -1,
                disp_width + pad[0] + pad[1],
                disp_width + pad[0] + pad[1],
                disp_width + pad[0] + pad[1],
            ),
            patch_width,
            stride=1,
        ).view(C, (j2 - j1), disp_width, disp_width, disp_width)
        corr = F.conv3d(
            feat_moving_disp.view(
                1,
                -1,
                disp_width + pad[0] + pad[1],
                disp_width + pad[0] + pad[1],
                disp_width + pad[0] + pad[1],
            ),
            feat_fixed_patch.view(-1, 1, patch_width, patch_width, patch_width),
            groups=C * (j2 - j1),
        ).view(C, (j2 - j1), disp_width, disp_width, disp_width)

        cost[0, j1:j2, :, :, :] = (fixed_sum + moving_sum - 2 * corr).sum(dim=0) / (
            patch_width**3
        )

    return cost


def compute_marginals(
    kpts_fix,
    img_fix,
    mind_fix,
    mind_mov,
    alpha,
    beta,
    disp_radius,
    disp_step,
    patch_radius,
):
    cost = alpha * ssd(
        kpts_fix, mind_fix, mind_mov, disp_radius, disp_step, patch_radius
    )

    dist = kpts_dist(kpts_fix, img_fix, beta)
    edges, level = minimum_spanning_tree(dist)
    marginals = tbp(cost, edges, level, dist)

    return marginals


def flow_pt(flow_world, shape, align_corners=None):
    device = flow_world.device
    D, H, W = shape

    flow_pt_ = (flow_world.flip(-1) / (torch.tensor([W, H, D], device=device) - 1)) * 2
    if not align_corners:
        flow_pt_ *= (torch.tensor([W, H, D], device=device) - 1) / torch.tensor(
            [W, H, D], device=device
        )

    return flow_pt_


def get_disp(disp_step, disp_radius, shape, device):
    D, H, W = shape

    disp = torch.stack(
        torch.meshgrid(
            torch.arange(
                -disp_step * disp_radius,
                disp_step * disp_radius + 1,
                disp_step,
                device=device,
            ),
            torch.arange(
                -disp_step * disp_radius,
                disp_step * disp_radius + 1,
                disp_step,
                device=device,
            ),
            torch.arange(
                -disp_step * disp_radius,
                disp_step * disp_radius + 1,
                disp_step,
                device=device,
            ),
        ),
        dim=3,
    ).view(1, -1, 3)

    disp = flow_pt(disp, (D, H, W), align_corners=True)
    return disp


def corrfield(
    img_fix,
    mask_fix,
    img_mov,
    alpha,
    beta,
    gamma,
    delta,
    sigma,
    sigma1,
    L,
    N,
    Q,
    R,
    T,
):
    device = img_fix.device
    _, _, D, H, W = img_fix.shape

    print("Compute fixed MIND features ...", end=" ")
    torch.cuda.synchronize()
    t0 = time.time()
    mind_fix = mindssc(img_fix, delta, sigma1)
    torch.cuda.synchronize()
    t1 = time.time()
    print("finished ({:.2f} s).".format(t1 - t0))

    dense_flow = torch.zeros((1, D, H, W, 3), device=device)
    img_mov_warped = img_mov
    for i in range(len(L)):
        print("Stage {}/{}".format(i + 1, len(L)))
        print("    search radius: {}".format(L[i]))
        print("      cube length: {}".format(N[i]))
        print("     quantisation: {}".format(Q[i]))
        print("     patch radius: {}".format(R[i]))
        print("        transform: {}".format(T[i]))

        disp = get_disp(Q[i], L[i], (D, H, W), device=device)

        print("    Compute moving MIND features ...", end=" ")
        torch.cuda.synchronize()
        t0 = time.time()
        mind_mov = mindssc(img_mov_warped, delta, sigma1)
        torch.cuda.synchronize()
        t1 = time.time()
        print("finished ({:.2f} s).".format(t1 - t0))

        torch.cuda.synchronize()
        t0 = time.time()
        kpts_fix = foerstner_kpts(img_fix, mask_fix, sigma, N[i])
        torch.cuda.synchronize()
        t1 = time.time()
        print(
            "    {} fixed keypoints extracted ({:.2f} s).".format(
                kpts_fix.shape[1], t1 - t0
            )
        )

        print("    Compute forward marginals ...", end=" ")
        torch.cuda.synchronize()
        t0 = time.time()
        marginalsf = compute_marginals(
            kpts_fix, img_fix, mind_fix, mind_mov, alpha, beta, L[i], Q[i], R[i]
        )
        torch.cuda.synchronize()
        t1 = time.time()
        print("finished ({:.2f} s).".format(t1 - t0))

        flow = (
            F.softmax(-gamma * marginalsf.view(1, kpts_fix.shape[1], -1, 1), dim=2)
            * disp.view(1, 1, -1, 3)
        ).sum(2)

        kpts_mov = kpts_fix + flow

        return kpts_mov, kpts_fix
