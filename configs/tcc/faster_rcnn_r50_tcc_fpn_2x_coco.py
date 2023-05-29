_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]
find_unused_parameters = True
model = dict(
    backbone=dict(
        stage_with_tcc=(True, True, True, True),
        norm_cfg = dict(type='SyncBN', requires_grad=True)
    ),
    neck=dict(use_tcc=True), 
)
