net = dict(type='CLRErNet')

num_classes = 5
ignore_label = 255
bg_weight = 0.4

backbone = dict(
    type='DLAWrapper',
    dla='dla34',
    pretrained=False,
)

neck = dict(
    type='FPN',
    in_channels=[128, 256, 512],
    out_channels=64,
    num_outs=3,
    attention=False,
)

heads = dict(
    type='CLRHead',
    num_priors=192,
    refine_layers=3,
    fc_hidden_dim=64,
    sample_points=36,
)

# Optional CLRErNet refinement knobs
clrernet = dict(
    enable_refine=True,
    refine_last_n=2,
)

num_points = 72
max_lanes = 4
sample_y = range(589, 230, -20)

test_parameters = dict(
    conf_threshold=0.4,
    nms_thres=50,
    nms_topk=max_lanes,
)

img_w = 800
img_h = 320
cut_height = 270
