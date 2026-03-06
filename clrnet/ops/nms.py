# Copyright (c) 2018, Grégoire Payen de La Garanderie, Durham University
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch

try:
    from . import nms_impl
except Exception:
    nms_impl = None


def _lane_overlap(box_a, box_b, threshold, n_strips=71):
    start_a = int(box_a[2].item() * n_strips + 0.5)
    start_b = int(box_b[2].item() * n_strips + 0.5)
    start = max(start_a, start_b)

    len_a = box_a[4].item()
    len_b = box_b[4].item()
    end_a = int(start_a + len_a - 1 + 0.5 - (1 if (len_a - 1) < 0 else 0))
    end_b = int(start_b + len_b - 1 + 0.5 - (1 if (len_b - 1) < 0 else 0))
    end = min(min(end_a, end_b), box_a.shape[0] - 6)

    if end < start:
        return False

    seg_a = box_a[5 + start:5 + end + 1]
    seg_b = box_b[5 + start:5 + end + 1]
    dist = torch.abs(seg_a - seg_b).sum().item()
    return dist < (threshold * (end - start + 1))


def _nms_fallback(boxes, scores, overlap, top_k):
    device = boxes.device
    boxes_num = boxes.shape[0]

    sorted_idx = torch.argsort(scores, descending=True)
    suppressed = torch.zeros(boxes_num, dtype=torch.bool, device=device)
    parent_object_index = torch.zeros(boxes_num, dtype=torch.long, device=device)

    keep_list = []
    for i in range(boxes_num):
        if suppressed[i]:
            continue

        idx_i = sorted_idx[i]
        keep_list.append(idx_i)
        keep_rank = len(keep_list)

        for j in range(i, boxes_num):
            if suppressed[j]:
                continue
            idx_j = sorted_idx[j]
            if _lane_overlap(boxes[idx_i], boxes[idx_j], overlap):
                suppressed[j] = True
                parent_object_index[idx_j] = keep_rank

        if len(keep_list) >= int(top_k):
            break

    keep = torch.zeros(boxes_num, dtype=torch.long, device=device)
    if keep_list:
        keep_vals = torch.stack(keep_list)
        keep[:keep_vals.numel()] = keep_vals

    num_to_keep = torch.tensor(min(int(top_k), len(keep_list)), dtype=torch.long, device=device)
    return keep, num_to_keep, parent_object_index


def nms(boxes, scores, overlap, top_k):
    if nms_impl is not None:
        return nms_impl.nms_forward(boxes, scores, overlap, top_k)
    return _nms_fallback(boxes, scores, overlap, top_k)
