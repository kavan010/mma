loss 245.43 | actor 0.039 | critic 490.861 | std 1.003 | maxImp 1300.0
[690] reward: 1.865 | advantages mean: -3.288
  --> impulse advanced to 1400
loss 284.26 | actor -0.001 | critic 568.608 | std 1.003 | maxImp 1400.0
[691] reward: 1.883 | advantages mean: 2.172
loss 202.51 | actor 0.013 | critic 405.074 | std 1.003 | maxImp 1400.0
[692] reward: 1.879 | advantages mean: -0.198
loss 258.24 | actor 0.057 | critic 516.453 | std 1.003 | maxImp 1400.0
[693] reward: 1.873 | advantages mean: -0.467
loss 263.80 | actor -0.000 | critic 527.686 | std 1.003 | maxImp 1400.0
[694] reward: 1.864 | advantages mean: -0.637
loss 250.51 | actor 0.067 | critic 500.962 | std 1.003 | maxImp 1400.0
[695] reward: 1.854 | advantages mean: -1.314
  --> impulse advanced to 1500
loss 279.32 | actor -0.010 | critic 558.755 | std 1.003 | maxImp 1500.0
[696] reward: 1.848 | advantages mean: -1.559
loss 262.82 | actor -0.012 | critic 525.750 | std 1.003 | maxImp 1500.0
[697] reward: 1.849 | advantages mean: -1.139
loss 242.10 | actor -0.011 | critic 484.299 | std 1.003 | maxImp 1500.0
[698] reward: 1.848 | advantages mean: -0.880
loss 254.69 | actor 0.056 | critic 509.349 | std 1.003 | maxImp 1500.0
[699] reward: 1.851 | advantages mean: -0.699
loss 253.20 | actor 0.036 | critic 506.419 | std 1.003 | maxImp 1500.0
[700] reward: 1.850 | advantages mean: 0.136
loss 242.27 | actor 0.005 | critic 484.608 | std 1.003 | maxImp 1500.0
[701] reward: 1.838 | advantages mean: -2.102
loss 273.64 | actor 0.006 | critic 547.350 | std 1.003 | maxImp 1500.0
[702] reward: 1.842 | advantages mean: -1.277
loss 265.69 | actor 0.010 | critic 531.437 | std 1.003 | maxImp 1500.0
[703] reward: 1.839 | advantages mean: -0.306
loss 236.86 | actor 0.042 | critic 473.730 | std 1.003 | maxImp 1500.0
[704] reward: 1.853 | advantages mean: 0.618
loss 226.80 | actor 0.009 | critic 453.662 | std 1.003 | maxImp 1500.0
[705] reward: 1.844 | advantages mean: -0.321
loss 254.43 | actor 0.033 | critic 508.888 | std 1.003 | maxImp 1500.0
[706] reward: 1.836 | advantages mean: -0.697
loss 254.42 | actor 0.036 | critic 508.859 | std 1.003 | maxImp 1500.0
[707] reward: 1.832 | advantages mean: -0.642
loss 257.81 | actor 0.016 | critic 515.668 | std 1.003 | maxImp 1500.0
[708] reward: 1.841 | advantages mean: -0.402
loss 244.78 | actor 0.072 | critic 489.497 | std 1.003 | maxImp 1500.0
[709] reward: 1.829 | advantages mean: -0.690
loss 231.14 | actor 0.042 | critic 462.274 | std 1.003 | maxImp 1500.0
[710] reward: 1.815 | advantages mean: -1.728
loss 287.74 | actor 0.068 | critic 575.434 | std 1.003 | maxImp 1500.0
[711] reward: 1.818 | advantages mean: -2.577
loss 265.55 | actor 0.155 | critic 530.883 | std 1.003 | maxImp 1500.0
[712] reward: 1.801 | advantages mean: -4.298
loss 293.86 | actor 0.059 | critic 587.678 | std 1.003 | maxImp 1500.0
[713] reward: 1.815 | advantages mean: -0.509
loss 253.88 | actor 0.217 | critic 507.414 | std 1.003 | maxImp 1500.0
[714] reward: 1.730 | advantages mean: -7.710
loss 312.72 | actor 0.210 | critic 625.101 | std 1.003 | maxImp 1500.0
[715] reward: 1.724 | advantages mean: -12.457
loss 300.48 | actor 0.346 | critic 600.350 | std 1.003 | maxImp 1500.0
[716] reward: 1.694 | advantages mean: -23.520
loss 231.31 | actor 0.550 | critic 461.610 | std 1.003 | maxImp 1500.0
[717] reward: 1.691 | advantages mean: -11.694
loss 93.90 | actor 0.351 | critic 187.184 | std 1.003 | maxImp 1500.0
[718] reward: 1.635 | advantages mean: -8.028
loss 83.32 | actor 0.339 | critic 166.050 | std 1.003 | maxImp 1500.0
[719] reward: 1.569 | advantages mean: -6.922
Traceback (most recent call last):
  File "/home/kavang/Desktop/mma/src/train.py", line 276, in <module>
    updateModel(impulse_mag)
    ~~~~~~~~~~~^^^^^^^^^^^^^
  File "/home/kavang/Desktop/mma/src/train.py", line 184, in updateModel
    dist, values = model(states[idx])
                   ~~~~~^^^^^^^^^^^^^
  File "/usr/lib/python3.14/site-packages/torch/nn/modules/module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.14/site-packages/torch/nn/modules/module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/kavang/Desktop/mma/src/train.py", line 60, in forward
    dist = Normal(mean, std)
  File "/usr/lib/python3.14/site-packages/torch/distributions/normal.py", line 66, in __init__
    super().__init__(batch_shape, validate_args=validate_args)
    ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.14/site-packages/torch/distributions/distribution.py", line 77, in __init__
    raise ValueError(
    ...<5 lines>...
    )
ValueError: Expected parameter loc (Tensor of shape (5120, 10)) of distribution Normal(loc: torch.Size([5120, 10]), scale: torch.Size([5120, 10])) to satisfy the constraint Real(), but found invalid values:
tensor([[nan, nan, nan,  ..., nan, nan, nan],
        [nan, nan, nan,  ..., nan, nan, nan],
        [nan, nan, nan,  ..., nan, nan, nan],
        ...,
        [nan, nan, nan,  ..., nan, nan, nan],
        [nan, nan, nan,  ..., nan, nan, nan],
        [nan, nan, nan,  ..., nan, nan, nan]], grad_fn=<AddmmBackward0>)
░▒    ~/De/mma/src    main !22 ?8                 1 ✘  14m 37s   16:50:40  ▓▒░