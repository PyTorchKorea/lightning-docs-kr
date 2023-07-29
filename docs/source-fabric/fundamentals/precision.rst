################################
Save memory with mixed precision
################################

.. video:: https://pl-public-data.s3.amazonaws.com/assets_lightning/fabric/animations/precision.mp4
    :width: 800
    :autoplay:
    :loop:
    :muted:
    :nocontrols:


************************
What is Mixed Precision?
************************

Like most deep learning frameworks, PyTorch runs on 32-bit floating-point (FP32) arithmetic by default.
However, many deep learning models do not require this to reach complete accuracy during training.
Mixed precision training delivers significant computational speedup by conducting operations in half-precision while keeping minimum information in single-precision to maintain as much information as possible in crucial areas of the network.
Switching to mixed precision has resulted in considerable training speedups since the introduction of Tensor Cores in the Volta and Turing architectures.
It combines FP32 and lower-bit floating points (such as FP16) to reduce memory footprint and increase performance during model training and evaluation.
It accomplishes this by recognizing the steps that require complete accuracy and employing a 32-bit floating point for those steps only while using a 16-bit floating point for the rest.
Compared to complete precision training, mixed precision training delivers all these benefits while ensuring no task-specific accuracy is lost `[1] <https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html>`_.

This is how you select the precision in Fabric:

.. code-block:: python

    from lightning.fabric import Fabric

    # This is the default
    fabric = Fabric(precision="32-true")

    # Also FP32 (legacy)
    fabric = Fabric(precision=32)

    # FP32 as well (legacy)
    fabric = Fabric(precision="32")

    # Float16 mixed precision
    fabric = Fabric(precision="16-mixed")

    # Float16 true half precision
    fabric = Fabric(precision="16-true")

    # BFloat16 mixed precision (Volta GPUs and later)
    fabric = Fabric(precision="bf16-mixed")

    # BFloat16 true half precision (Volta GPUs and later)
    fabric = Fabric(precision="bf16-true")

    # 8-bit mixed precision via TransformerEngine (Hopper GPUs and later)
    fabric = Fabric(precision="transformer-engine")

    # Double precision
    fabric = Fabric(precision="64-true")

    # Or (legacy)
    fabric = Fabric(precision="64")

    # Or (legacy)
    fabric = Fabric(precision=64)


The same values can also be set through the :doc:`command line interface <launch>`:

.. code-block:: bash

    lightning run model train.py --precision=bf16-mixed


.. note::

    In some cases, it is essential to remain in FP32 for numerical stability, so keep this in mind when using mixed precision.
    For example, when running scatter operations during the forward (such as torchpoint3d), the computation must remain in FP32.


----


********************
FP16 Mixed Precision
********************

In most cases, mixed precision uses FP16.
Supported `PyTorch operations <https://pytorch.org/docs/stable/amp.html#op-specific-behavior>`_ automatically run in FP16, saving memory and improving throughput on the supported accelerators.
Since computation happens in FP16, which has a very limited "dynamic range", there is a chance of numerical instability during training.
This is handled internally by a dynamic grad scaler which skips invalid steps and adjusts the scaler to ensure subsequent steps fall within a finite range.
For more information `see the autocast docs <https://pytorch.org/docs/stable/amp.html#gradient-scaling>`_.

This is how you enable FP16 in Fabric:

.. code-block:: python

    # Select FP16 mixed precision
    fabric = Fabric(precision="16-mixed")

.. note::

    When using TPUs, setting ``precision="16-mixed"`` will enable bfloat16 based mixed precision, the only supported half-precision type on TPUs.


----


************************
BFloat16 Mixed Precision
************************

BFloat16 Mixed precision is similar to FP16 mixed precision. However, it maintains more of the "dynamic range" that FP32 offers.
This means it can improve numerical stability than FP16 mixed precision.
For more information, see `this TPU performance blog post <https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus>`_.

.. code-block:: python

    # Select BF16 precision
    fabric = Fabric(precision="bf16-mixed")


Under the hood, we use `torch.autocast <https://pytorch.org/docs/stable/amp.html>`__ with the dtype set to ``bfloat16``, with no gradient scaling.
It is also possible to use BFloat16 mixed precision on the CPU, relying on MKLDNN.

.. note::

    BFloat16 may not provide significant speedups or memory improvements, offering better numerical stability.
    For GPUs, the most significant benefits require `Ampere <https://en.wikipedia.org/wiki/Ampere_(microarchitecture)>`_ based GPUs or newer, such as A100s or 3090s.


----


*****************************************************
Float8 Mixed Precision via Nvidia's TransformerEngine
*****************************************************

`Transformer Engine <https://github.com/NVIDIA/TransformerEngine>`__ (TE) is a library for accelerating models on the
latest NVIDIA GPUs using 8-bit floating point (FP8) precision on Hopper GPUs, to provide better performance with lower
memory utilization in both training and inference. It offers improved performance over half precision with no degradation in accuracy.

Using TE requires replacing some of the layers in your model. Fabric automatically replaces the :class:`torch.nn.Linear`
and :class:`torch.nn.LayerNorm` layers in your model with their TE alternatives, however, TE also offers
`fused layers <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/pytorch.html>`__
to squeeze out all the possible performance. If Fabric detects that any layer has been replaced already, automatic
replacement is not done.

This plugin is a mix of "mixed" and "true" precision. The computation is downcasted to FP8 precision on the fly, but
the model and inputs can be kept in true full or half precision.

.. code-block:: python

    # Select 8bit mixed precision via TransformerEngine
    fabric = Fabric(precision="transformer-engine")

    # Customize the fp8 recipe or set a different base precision:
    from lightning.fabric.plugins.precision import TransformerEnginePrecision

    recipe = {"fp8_format": "HYBRID", "amax_history_len": 16, "amax_compute_algo": "max"}
    precision = TransformerEnginePrecision(dtype=torch.bfloat16, recipe=recipe)
    fabric = Fabric(plugins=precision)


Under the hood, we use `transformer_engine.pytorch.fp8_autocast <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/pytorch.html#transformer_engine.pytorch.fp8_autocast>`__ with the default fp8 recipe.

.. note::

    This requires `Hopper <https://en.wikipedia.org/wiki/Hopper_(microarchitecture)>`_ based GPUs or newer, such the H100.


----


*******************
True Half Precision
*******************

As mentioned before, for numerical stability mixed precision keeps the model weights in full float32 precision while casting only supported operations to lower bit precision.
However, in some cases it is indeed possible to train completely in half precision. Similarly, for inference the model weights can often be cast to half precision without a loss in accuracy (even when trained with mixed precision).

.. code-block:: python

    # Select FP16 precision
    fabric = Fabric(precision="16-true")
    model = MyModel()
    model = fabric.setup(model)  # model gets cast to torch.float16

    # Select BF16 precision
    fabric = Fabric(precision="bf16-true")
    model = MyModel()
    model = fabric.setup(model)  # model gets cast to torch.bfloat16

Tip: For faster initialization, you can create model parameters with the desired dtype directly on the device:

.. code-block:: python

    fabric = Fabric(precision="bf16-true")

    # init the model directly on the device and with parameters in half-precision
    with fabric.init_module():
        model = MyModel()

    model = fabric.setup(model)


----


************************************
Control where precision gets applied
************************************

Fabric automatically casts the data type and operations in the ``forward`` of your model:

.. code-block:: python

    fabric = Fabric(precision="bf16-mixed")

    model = ...
    optimizer = ...

    # Here, Fabric sets up the `model.forward` for precision auto-casting
    model, optimizer = fabric.setup(model, optimizer)

    # Precision casting gets handled in your forward, no code changes required
    output = model.forward(input)

    # Precision does NOT get applied here (only in forward)
    loss = loss_function(output, target)

If you want to enable operations in lower bit-precision **outside** your model's ``forward()``, you can use the :meth:`~lightning.fabric.fabric.Fabric.autocast` context manager:

.. code-block:: python

    # Precision now gets also handled in this part of the code:
    with fabric.autocast():
        loss = loss_function(output, target)
