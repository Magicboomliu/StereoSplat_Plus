optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'pretrained': dict(lr_mult=0.01),
        }
    ),
    weight_decay=0.01
)

grad_max_norm = 1.0

