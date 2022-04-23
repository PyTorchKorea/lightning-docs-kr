.. _model-parallel:

Train 1 trillion+ parameter models
==================================

When training large models, fitting larger batch sizes, or trying to increase throughput using multi-GPU compute, Lightning provides advanced optimized distributed training strategies to support these cases and offer substantial improvements in memory usage.

In many cases these strategies are some flavour of model parallelism however we only introduce concepts at a high level to get you started. Refer to the `FairScale documentation <https://fairscale.readthedocs.io/en/latest/deep_dive/oss_sdp_fsdp.html>`_ for more information about model parallelism.

Note that some of the extreme memory saving configurations will affect the speed of training. This Speed/Memory trade-off in most cases can be adjusted.

Some of these memory-efficient strategies rely on offloading onto other forms of memory, such as CPU RAM or NVMe. This means you can even see memory benefits on a **single GPU**, using a strategy such as :ref:`deepspeed-zero-stage-3-offload`.

Check out this amazing video explaining model parallelism and how it works behind the scenes:

.. raw:: html

    <iframe width="540" height="300" src="https://www.youtube.com/embed/w_CKzh5C1K4" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


Choosing an Advanced Distributed GPU Strategy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you would like to stick with PyTorch DDP, see :ref:`ddp-optimizations`.

Unlike :class:`~torch.nn.parallel.DistributedDataParallel` (DDP) where the maximum trainable model size and batch size do not change with respect to the number of GPUs, memory-optimized strategies can accommodate bigger models and larger batches as more GPUs are used. This means as you scale up the number of GPUs, you can reach the number of model parameters you'd like to train.

There are many considerations when choosing a strategy as described below. In addition, check out the visualization of various strategy benchmarks using `minGPT <https://github.com/SeanNaren/minGPT>`__ `here <https://share.streamlit.io/seannaren/mingpt/streamlit/app.py>`__.

Pre-training vs Fine-tuning
"""""""""""""""""""""""""""

When fine-tuning, we often use a magnitude less data compared to pre-training a model. This is important when choosing a distributed strategy as usually for pre-training, **we are compute-bound**.
This means we cannot sacrifice throughput as much as if we were fine-tuning, because in fine-tuning the data requirement is smaller.

Overall:

* When **fine-tuning** a model, use advanced memory efficient strategies such as :ref:`deepspeed-zero-stage-3` or :ref:`deepspeed-zero-stage-3-offload`, allowing you to fine-tune larger models if you are limited on compute
* When **pre-training** a model, use simpler optimizations such :ref:`sharded-training`, :ref:`deepspeed-zero-stage-2` or :ref:`fully-sharded-training`, scaling the number of GPUs to reach larger parameter sizes
* For both fine-tuning and pre-training, use :ref:`deepspeed-activation-checkpointing` or :ref:`fairscale-activation-checkpointing` as the throughput degradation is not significant

For example when using 128 GPUs, you can **pre-train** large 10 to 20 Billion parameter models using :ref:`deepspeed-zero-stage-2` without having to take a performance hit with more advanced optimized multi-gpu strategy.

But for **fine-tuning** a model, you can reach 10 to 20 Billion parameter models using :ref:`deepspeed-zero-stage-3-offload` on a **single GPU**. This does come with a significant throughput hit, which needs to be weighed accordingly.

When Shouldn't I use an Optimized Distributed Strategy?
"""""""""""""""""""""""""""""""""""""""""""""""""""""""

Sharding techniques help when model sizes are fairly large; roughly 500M+ parameters is where we've seen benefits. However, in the following cases, we recommend sticking to ordinary distributed strategies
* When your model is small (ResNet50 of around 80M Parameters), unless you are using unusually large batch sizes or inputs.
* Due to high distributed communication between devices, if running on a slow network/interconnect, the training might be much slower than expected and then it's up to you to determince the tradeoff here.

----------

.. _sharded-training:

Sharded Training
^^^^^^^^^^^^^^^^
Lightning integration of optimizer sharded training provided by `FairScale <https://github.com/facebookresearch/fairscale>`_.
The technique can be found within `DeepSpeed ZeRO <https://arxiv.org/abs/1910.02054>`_ and
`ZeRO-2 <https://www.microsoft.com/en-us/research/blog/zero-2-deepspeed-shattering-barriers-of-deep-learning-speed-scale/>`_,
however the implementation is built from the ground up to be PyTorch compatible and standalone.
Sharded Training allows you to maintain GPU scaling efficiency, whilst reducing memory overhead drastically. In short, expect near-normal linear scaling (if your network allows), and significantly reduced memory usage when training large models.

Sharded Training still utilizes Data Parallel Training under the hood, except optimizer states and gradients are sharded across GPUs.
This means the memory overhead per GPU is lower, as each GPU only has to maintain a partition of your optimizer state and gradients.

The benefits vary by model and parameter sizes, but we've recorded up to a 63% memory reduction per GPU allowing us to double our model sizes. Because of efficient communication,
these benefits in multi-GPU setups are almost free and throughput scales well with multi-node setups.

It is highly recommended to use Sharded Training in multi-GPU environments where memory is limited, or where training larger models are beneficial (500M+ parameter models).
A technical note: as batch size scales, storing activations for the backwards pass becomes the bottleneck in training. As a result, sharding optimizer state and gradients becomes less impactful.
Use :ref:`fairscale-activation-checkpointing` to see even more benefit at the cost of some throughput.

To use Sharded Training, you need to first install FairScale using the command below.

.. code-block:: bash

    pip install fairscale


.. code-block:: python

    # train using Sharded DDP
    trainer = Trainer(strategy="ddp_sharded")

Sharded Training can work across all DDP variants by adding the additional ``--strategy ddp_sharded`` flag via command line using a PyTorch Lightning script.

Internally we re-initialize your optimizers and shard them across your machines and processes. We handle all communication using PyTorch distributed, so no code changes are required.

----------

.. _fully-sharded-training:

Fully Sharded Training
^^^^^^^^^^^^^^^^^^^^^^

.. warning::
    Fully Sharded Training is in beta and the API is subject to change. Please create an `issue <https://github.com/PyTorchLightning/pytorch-lightning/issues>`_ if you run into any issues.

`Fully Sharded <https://fairscale.readthedocs.io/en/latest/api/nn/fsdp.html>`__ shards optimizer state, gradients and parameters across data parallel workers. This allows you to fit much larger models onto multiple GPUs into memory.

Fully Sharded Training alleviates the need to worry about balancing layers onto specific devices using some form of pipe parallelism, and optimizes for distributed communication with minimal effort.

Shard Parameters to Reach 10+ Billion Parameters
""""""""""""""""""""""""""""""""""""""""""""""""

To reach larger parameter sizes and be memory efficient, we have to shard parameters. There are various ways to enable this.

.. note::
    Currently Fully Sharded Training relies on the user to wrap the model with Fully Sharded within the ``LightningModule``.
    This means you must create a single model that is treated as a ``torch.nn.Module`` within the ``LightningModule``.
    This is a limitation of Fully Sharded Training that will be resolved in the future.

Enabling Module Sharding for Maximum Memory Efficiency
""""""""""""""""""""""""""""""""""""""""""""""""""""""

To activate parameter sharding, you must wrap your model using provided ``wrap`` or ``auto_wrap`` functions as described below. Internally in Lightning, we enable a context manager around the ``configure_sharded_model`` function to make sure the ``wrap`` and ``auto_wrap`` parameters are passed correctly.

When not using Fully Sharded these wrap functions are a no-op. This means once the changes have been made, there is no need to remove the changes for other strategies.

``auto_wrap`` will recursively wrap :class:`~torch.nn.Module` within the ``LightningModule`` with nested Fully Sharded Wrappers,
signalling that we'd like to partition these modules across data parallel devices, discarding the full weights when not required (information :class:`here <fairscale.nn.fsdp>`).

``auto_wrap`` can have varying level of success based on the complexity of your model. **Auto Wrap does not support models with shared parameters**.

``wrap`` will simply wrap the module with a Fully Sharded Parallel class with the correct parameters from the Lightning context manager.

Below is an example of using both ``wrap`` and ``auto_wrap`` to create your model.

.. code-block:: python

    import torch
    import torch.nn as nn
    import pytorch_lightning as pl
    from pytorch_lightning import Trainer
    from fairscale.nn import checkpoint_wrapper, auto_wrap, wrap


    class MyModel(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.linear_layer = nn.Linear(32, 32)
            self.block = nn.Sequential(nn.Linear(32, 32), nn.ReLU())
            self.final_block = nn.Sequential(nn.Linear(32, 32), nn.ReLU())

        def configure_sharded_model(self):
            # modules are sharded across processes
            # as soon as they are wrapped with ``wrap`` or ``auto_wrap``.
            # During the forward/backward passes, weights get synced across processes
            # and de-allocated once computation is complete, saving memory.

            # Wraps the layer in a Fully Sharded Wrapper automatically
            linear_layer = wrap(self.linear_layer)

            # Wraps the module recursively
            # based on a minimum number of parameters (default 100M parameters)
            block = auto_wrap(self.block)

            # For best memory efficiency,
            # add FairScale activation checkpointing
            final_block = auto_wrap(checkpoint_wrapper(self.final_block))
            self.model = nn.Sequential(linear_layer, nn.ReLU(), block, final_block)

        def configure_optimizers(self):
            return torch.optim.AdamW(self.model.parameters())


    model = MyModel()
    trainer = Trainer(accelerator="gpu", devices=4, strategy="fsdp", precision=16)
    trainer.fit(model)

    trainer.test()
    trainer.predict()


----------

.. _fairscale-activation-checkpointing:

FairScale Activation Checkpointing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Activation checkpointing frees activations from memory as soon as they are not needed during the forward pass. They are then re-computed for the backwards pass as needed. Activation checkpointing is very useful when you have intermediate layers that produce large activations.

FairScales' checkpointing wrapper also handles batch norm layers correctly unlike the PyTorch implementation, ensuring stats are tracked correctly due to the multiple forward passes.

This saves memory when training larger models however requires wrapping modules you'd like to use activation checkpointing on. See :class:`here <fairscale.nn.checkpoint.checkpoint_wrapper>` for more information.

.. warning::

    Ensure to not wrap the entire model with activation checkpointing. This is not the intended usage of activation checkpointing, and will lead to failures as seen in `this discussion <https://github.com/PyTorchLightning/pytorch-lightning/discussions/9144>`__.

.. code-block:: python

    from pytorch_lightning import Trainer
    from fairscale.nn import checkpoint_wrapper


    class MyModel(pl.LightningModule):
        def __init__(self):
            super().__init__()
            # Wrap layers using checkpoint_wrapper
            self.block_1 = checkpoint_wrapper(nn.Sequential(nn.Linear(32, 32), nn.ReLU()))
            self.block_2 = nn.Linear(32, 2)


.. _deepspeed_advanced:

DeepSpeed
^^^^^^^^^

.. note::
    The DeepSpeed strategy is in beta and the API is subject to change. Please create an `issue <https://github.com/PyTorchLightning/pytorch-lightning/issues>`_ if you run into any issues.

`DeepSpeed <https://github.com/microsoft/DeepSpeed>`__ is a deep learning training optimization library, providing the means to train massive billion parameter models at scale.
Using the DeepSpeed strategy, we were able to **train model sizes of 10 Billion parameters and above**, with a lot of useful information in this `benchmark <https://github.com/huggingface/transformers/issues/9996>`_ and the `DeepSpeed docs <https://www.deepspeed.ai/tutorials/megatron/>`__.
DeepSpeed also offers lower level training optimizations, and efficient optimizers such as `1-bit Adam <https://www.deepspeed.ai/tutorials/onebit-adam/>`_. We recommend using DeepSpeed in environments where speed and memory optimizations are important (such as training large billion parameter models).

Below is a summary of all the configurations of DeepSpeed.

* :ref:`deepspeed-zero-stage-1` - **Shard optimizer states**, remains at speed parity with DDP whilst providing memory improvement

* :ref:`deepspeed-zero-stage-2` - **Shard optimizer states and gradients**, remains at speed parity with DDP whilst providing even more memory improvement

* :ref:`deepspeed-zero-stage-2-offload` - **Offload optimizer states and gradients to CPU**. Increases distributed communication volume and GPU-CPU device transfer, but provides significant memory improvement

* :ref:`deepspeed-zero-stage-3` - **Shard optimizer states, gradients, parameters and optionally activations**. Increases distributed communication volume, but provides even more memory improvement

* :ref:`deepspeed-zero-stage-3-offload` - **Offload optimizer states, gradients, parameters and optionally activations to CPU**. Increases distributed communication volume and GPU-CPU device transfer, but even more significant memory improvement.

* :ref:`deepspeed-activation-checkpointing` - **Free activations after forward pass**. Increases computation, but provides memory improvement for all stages.

To use DeepSpeed, you first need to install DeepSpeed using the commands below.

.. code-block:: bash

    pip install deepspeed

If you run into an issue with the install or later in training, ensure that the CUDA version of the PyTorch you've installed matches your locally installed CUDA (you can see which one has been recognized by running ``nvcc --version``).

.. note::

    DeepSpeed currently only supports single optimizer, single scheduler within the training loop.

    When saving a checkpoint we rely on DeepSpeed which saves a directory containing the model and various components.


.. _deepspeed-zero-stage-1:

DeepSpeed ZeRO Stage 1
""""""""""""""""""""""

`DeepSpeed ZeRO Stage 1 <https://www.deepspeed.ai/tutorials/zero/#zero-overview>`_ partitions your optimizer states (Stage 1) across your GPUs to reduce memory.

It is recommended to skip Stage 1 and use Stage 2, which comes with larger memory improvements and still remains efficient. Stage 1 is useful to pair with certain optimizations such as `Torch ORT <https://github.com/pytorch/ort>`__.

.. code-block:: python

    from pytorch_lightning import Trainer

    model = MyModel()
    trainer = Trainer(accelerator="gpu", devices=4, strategy="deepspeed_stage_1", precision=16)
    trainer.fit(model)


.. _deepspeed-zero-stage-2:

DeepSpeed ZeRO Stage 2
""""""""""""""""""""""

`DeepSpeed ZeRO Stage 2 <https://www.deepspeed.ai/tutorials/zero/#zero-overview>`_ partitions your optimizer states (Stage 1) and your gradients (Stage 2) across your GPUs to reduce memory. In most cases, this is more efficient or at parity with DDP, primarily due to the optimized custom communications written by the DeepSpeed team.
As a result, benefits can also be seen on a single GPU. Do note that the default bucket sizes allocate around ``3.6GB`` of VRAM to use during distributed communications, which can be tweaked when instantiating the strategy described in a few sections below.

.. code-block:: python

    from pytorch_lightning import Trainer

    model = MyModel()
    trainer = Trainer(accelerator="gpu", devices=4, strategy="deepspeed_stage_2", precision=16)
    trainer.fit(model)

.. code-block:: bash

    python train.py --strategy deepspeed_stage_2 --precision 16 --accelerator 'gpu' --devices 4


.. _deepspeed-zero-stage-2-offload:

DeepSpeed ZeRO Stage 2 Offload
""""""""""""""""""""""""""""""

Below we show an example of running `ZeRO-Offload <https://www.deepspeed.ai/tutorials/zero-offload/>`_. ZeRO-Offload leverages the host CPU to offload optimizer memory/computation, reducing the overall memory consumption.

.. code-block:: python

    from pytorch_lightning import Trainer

    model = MyModel()
    trainer = Trainer(accelerator="gpu", devices=4, strategy="deepspeed_stage_2_offload", precision=16)
    trainer.fit(model)


This can also be done via the command line using a PyTorch Lightning script:

.. code-block:: bash

    python train.py --strategy deepspeed_stage_2_offload --precision 16 --accelerator 'gpu' --devices 4


You can also modify the ZeRO-Offload parameters via the strategy as below.

.. code-block:: python

    from pytorch_lightning import Trainer
    from pytorch_lightning.strategies import DeepSpeedStrategy

    model = MyModel()
    trainer = Trainer(
        accelerator="gpu",
        devices=4,
        strategy=DeepSpeedStrategy(offload_optimizer=True, allgather_bucket_size=5e8, reduce_bucket_size=5e8),
        precision=16,
    )
    trainer.fit(model)


.. note::
    We suggest tuning the ``allgather_bucket_size`` parameter and ``reduce_bucket_size`` parameter to find optimum parameters based on your model size.
    These control how large a buffer we limit the model to using when reducing gradients/gathering updated parameters. Smaller values will result in less memory, but tradeoff with speed.

    DeepSpeed allocates a reduce buffer size `multiplied by 1.5x <https://github.com/microsoft/DeepSpeed/blob/fead387f7837200fefbaba3a7b14709072d8d2cb/deepspeed/runtime/zero/stage_1_and_2.py#L2188>`_ so take that into consideration when tweaking the parameters.

    The strategy sets a reasonable default of ``2e8``, which should work for most low VRAM GPUs (less than ``7GB``), allocating roughly ``3.6GB`` of VRAM as buffer. Higher VRAM GPUs should aim for values around ``5e8``.

For even more speed benefit, DeepSpeed offers an optimized CPU version of ADAM called `DeepSpeedCPUAdam <https://deepspeed.readthedocs.io/en/latest/optimizers.html#adam-cpu>`_ to run the offloaded computation, which is faster than the standard PyTorch implementation.

.. code-block:: python

    import pytorch_lightning
    from pytorch_lightning import Trainer
    from deepspeed.ops.adam import DeepSpeedCPUAdam


    class MyModel(pl.LightningModule):
        ...

        def configure_optimizers(self):
            # DeepSpeedCPUAdam provides 5x to 7x speedup over torch.optim.adam(w)
            return DeepSpeedCPUAdam(self.parameters())


    model = MyModel()
    trainer = Trainer(accelerator="gpu", devices=4, strategy="deepspeed_stage_2_offload", precision=16)
    trainer.fit(model)


.. _deepspeed-zero-stage-3:

DeepSpeed ZeRO Stage 3
""""""""""""""""""""""

DeepSpeed ZeRO Stage 3 shards the optimizer states, gradients and the model parameters (also optionally activations). Sharding model parameters and activations comes with an increase in distributed communication, however allows you to scale your models massively from one GPU to multiple GPUs.
**The DeepSpeed team report the ability to fine-tune models with over 40B parameters on a single GPU and over 2 Trillion parameters on 512 GPUs.** For more information we suggest checking the `DeepSpeed ZeRO-3 Offload documentation <https://www.deepspeed.ai/news/2021/03/07/zero3-offload.html>`__.

We've ran benchmarks for all these features and given a simple example of how all these features work in Lightning, which you can see at `minGPT <https://github.com/SeanNaren/minGPT/tree/stage3>`_.

To reach the highest memory efficiency or model size, you must:

1. Use the DeepSpeed strategy with the stage 3 parameter
2. Use CPU Offloading to offload weights to CPU, plus have a reasonable amount of CPU RAM to offload onto
3. Use DeepSpeed Activation Checkpointing to shard activations

Below we describe how to enable all of these to see benefit. **With all these improvements we reached 45 Billion parameters training a GPT model on 8 GPUs with ~1TB of CPU RAM available**.

Also please have a look at our :ref:`deepspeed-zero-stage-3-tips` which contains a lot of helpful information when configuring your own models.

.. note::

    When saving a model using DeepSpeed and Stage 3, model states and optimizer states will be saved in separate sharded states (based on the world size). See :ref:`deepspeed-zero-stage-3-single-file` to obtain a single checkpoint file.

.. code-block:: python

    from pytorch_lightning import Trainer
    from deepspeed.ops.adam import FusedAdam


    class MyModel(pl.LightningModule):
        ...

        def configure_optimizers(self):
            return FusedAdam(self.parameters())


    model = MyModel()
    trainer = Trainer(accelerator="gpu", devices=4, strategy="deepspeed_stage_3", precision=16)
    trainer.fit(model)

    trainer.test()
    trainer.predict()


You can also use the Lightning Trainer to run predict or evaluate with DeepSpeed once the model has been trained.

.. code-block:: python

    from pytorch_lightning import Trainer


    class MyModel(pl.LightningModule):
        ...


    model = MyModel()
    trainer = Trainer(accelerator="gpu", devices=4, strategy="deepspeed_stage_3", precision=16)
    trainer.test(ckpt_path="my_saved_deepspeed_checkpoint.ckpt")


Shard Model Instantly to Reduce Initialization Time/Memory
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

When instantiating really large models, it is sometimes necessary to shard the model layers instantly.

This is the case if layers may not fit on one single machines CPU or GPU memory, but would fit once sharded across multiple machines.
We expose a hook that layers initialized within the hook will be sharded instantly on a per layer basis, allowing you to instantly shard models.

This reduces the time taken to initialize very large models, as well as ensure we do not run out of memory when instantiating larger models. For more information you can refer to the DeepSpeed docs for `Constructing Massive Models <https://deepspeed.readthedocs.io/en/latest/zero3.html>`_.

.. code-block:: python

    import torch.nn as nn
    from pytorch_lightning import Trainer
    from deepspeed.ops.adam import FusedAdam


    class MyModel(pl.LightningModule):
        ...

        def configure_sharded_model(self):
            # Created within sharded model context, modules are instantly sharded across processes
            # as soon as they are made.
            self.block = nn.Sequential(nn.Linear(32, 32), nn.ReLU())

        def configure_optimizers(self):
            return FusedAdam(self.parameters())


    model = MyModel()
    trainer = Trainer(accelerator="gpu", devices=4, strategy="deepspeed_stage_3", precision=16)
    trainer.fit(model)

    trainer.test()
    trainer.predict()


.. _deepspeed-zero-stage-3-offload:

DeepSpeed ZeRO Stage 3 Offload
""""""""""""""""""""""""""""""

DeepSpeed ZeRO Stage 3 Offloads optimizer state, gradients to the host CPU to reduce memory usage as ZeRO Stage 2 does, however additionally allows you to offload the parameters as well for even more memory saving.

.. note::

    When saving a model using DeepSpeed and Stage 3, model states and optimizer states will be saved in separate sharded states (based on the world size). See :ref:`deepspeed-zero-stage-3-single-file` to obtain a single checkpoint file.

.. code-block:: python

    from pytorch_lightning import Trainer
    from pytorch_lightning.strategies import DeepSpeedStrategy

    # Enable CPU Offloading
    model = MyModel()
    trainer = Trainer(accelerator="gpu", devices=4, strategy="deepspeed_stage_3_offload", precision=16)
    trainer.fit(model)

    # Enable CPU Offloading, and offload parameters to CPU
    model = MyModel()
    trainer = Trainer(
        accelerator="gpu",
        devices=4,
        strategy=DeepSpeedStrategy(
            stage=3,
            offload_optimizer=True,
            offload_parameters=True,
        ),
        precision=16,
    )
    trainer.fit(model)


DeepSpeed Infinity (NVMe Offloading)
""""""""""""""""""""""""""""""""""""

Additionally, DeepSpeed supports offloading to NVMe drives for even larger models, utilizing the large memory space found in NVMes. DeepSpeed `reports <https://www.microsoft.com/en-us/research/blog/zero-infinity-and-deepspeed-unlocking-unprecedented-model-scale-for-deep-learning-training/>`__ the ability to fine-tune 1 Trillion+ parameters using NVMe Offloading on one 8 GPU machine. Below shows how to enable this, assuming the NVMe drive is mounted in a directory called ``/local_nvme``.

.. code-block:: python

    from pytorch_lightning import Trainer
    from pytorch_lightning.strategies import DeepSpeedStrategy

    # Enable CPU Offloading
    model = MyModel()
    trainer = Trainer(accelerator="gpu", devices=4, strategy="deepspeed_stage_3_offload", precision=16)
    trainer.fit(model)

    # Enable CPU Offloading, and offload parameters to CPU
    model = MyModel()
    trainer = Trainer(
        accelerator="gpu",
        devices=4,
        strategy=DeepSpeedStrategy(
            stage=3,
            offload_optimizer=True,
            offload_parameters=True,
            remote_device="nvme",
            offload_params_device="nvme",
            offload_optimizer_device="nvme",
            nvme_path="/local_nvme",
        ),
        precision=16,
    )
    trainer.fit(model)

When offloading to NVMe you may notice that the speed is slow. There are parameters that need to be tuned based on the drives that you are using. Running the `aio_bench_perf_sweep.py <https://github.com/microsoft/DeepSpeed/blob/master/csrc/aio/py_test/aio_bench_perf_sweep.py>`__ script can help you to find optimum parameters. See the `issue <https://github.com/microsoft/DeepSpeed/issues/998>`__ for more information on how to parse the information.

.. _deepspeed-activation-checkpointing:

DeepSpeed Activation Checkpointing
""""""""""""""""""""""""""""""""""

Activation checkpointing frees activations from memory as soon as they are not needed during the forward pass.
They are then re-computed for the backwards pass as needed.

Activation checkpointing is very useful when you have intermediate layers that produce large activations.

This saves memory when training larger models, however requires using a checkpoint function to run modules as shown below.

.. warning::

    Ensure to not wrap the entire model with activation checkpointing. This is not the intended usage of activation checkpointing, and will lead to failures as seen in `this discussion <https://github.com/PyTorchLightning/pytorch-lightning/discussions/9144>`__.

.. code-block:: python

    from pytorch_lightning import Trainer
    import deepspeed


    class MyModel(LightningModule):
        ...

        def __init__(self):
            super().__init__()
            self.block_1 = nn.Sequential(nn.Linear(32, 32), nn.ReLU())
            self.block_2 = torch.nn.Linear(32, 2)

        def forward(self, x):
            # Use the DeepSpeed checkpointing function instead of calling the module directly
            # checkpointing self.block_1 means the activations are deleted after use,
            # and re-calculated during the backward passes
            x = deepspeed.checkpointing.checkpoint(self.block_1, x)
            return self.block_2(x)


.. code-block:: python

    from pytorch_lightning import Trainer
    from pytorch_lightning.strategies import DeepSpeedStrategy
    import deepspeed


    class MyModel(pl.LightningModule):
        ...

        def configure_sharded_model(self):
            self.block_1 = nn.Sequential(nn.Linear(32, 32), nn.ReLU())
            self.block_2 = torch.nn.Linear(32, 2)

        def forward(self, x):
            # Use the DeepSpeed checkpointing function instead of calling the module directly
            x = deepspeed.checkpointing.checkpoint(self.block_1, x)
            return self.block_2(x)


    model = MyModel()

    trainer = Trainer(accelerator="gpu", devices=4, strategy="deepspeed_stage_3_offload", precision=16)

    # Enable CPU Activation Checkpointing
    trainer = Trainer(
        accelerator="gpu",
        devices=4,
        strategy=DeepSpeedStrategy(
            stage=3,
            offload_optimizer=True,  # Enable CPU Offloading
            cpu_checkpointing=True,  # (Optional) offload activations to CPU
        ),
        precision=16,
    )
    trainer.fit(model)


.. _deepspeed-zero-stage-3-tips:

DeepSpeed ZeRO Stage 3 Tips
"""""""""""""""""""""""""""

Here is some helpful information when setting up DeepSpeed ZeRO Stage 3 with Lightning.

* If you're using Adam or AdamW, ensure to use FusedAdam or DeepSpeedCPUAdam (for CPU Offloading) rather than the default torch optimizers as they come with large speed benefits
* Treat your GPU/CPU memory as one large pool. In some cases, you may not want to offload certain things (like activations) to provide even more space to offload model parameters
* When offloading to the CPU, make sure to bump up the batch size as GPU memory will be freed
* We also support sharded checkpointing. By passing ``save_full_weights=False`` to the ``DeepSpeedStrategy``, we'll save shards of the model which allows you to save extremely large models. However to load the model and run test/validation/predict you must use the Trainer object.

.. _deepspeed-zero-stage-3-single-file:

Collating Single File Checkpoint for DeepSpeed ZeRO Stage 3
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

After training using ZeRO Stage 3, you'll notice that your checkpoints are a directory of sharded model and optimizer states. If you'd like to collate a single file from the checkpoint directory please use the below command, which handles all the Lightning states additionally when collating the file.

.. code-block:: python

    from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

    # lightning deepspeed has saved a directory instead of a file
    save_path = "lightning_logs/version_0/checkpoints/epoch=0-step=0.ckpt/"
    output_path = "lightning_model.pt"
    convert_zero_checkpoint_to_fp32_state_dict(save_path, output_path)


.. warning::

    This single file checkpoint does not include the optimizer/lr-scheduler states. This means we cannot restore training via the ``trainer.fit(ckpt_path=)`` call. Ensure to keep the sharded checkpoint directory if this is required.

Custom DeepSpeed Config
"""""""""""""""""""""""

In some cases you may want to define your own DeepSpeed Config, to access all parameters defined. We've exposed most of the important parameters, however, there may be debugging parameters to enable. Also, DeepSpeed allows the use of custom DeepSpeed optimizers and schedulers defined within a config file that is supported.

.. note::
    All strategy default parameters will be ignored when a config object is passed.
    All compatible arguments can be seen in the `DeepSpeed docs <https://www.deepspeed.ai/docs/config-json/>`_.

.. code-block:: python

    from pytorch_lightning import Trainer
    from pytorch_lightning.strategies import DeepSpeedStrategy

    deepspeed_config = {
        "zero_allow_untested_optimizer": True,
        "optimizer": {
            "type": "OneBitAdam",
            "params": {
                "lr": 3e-5,
                "betas": [0.998, 0.999],
                "eps": 1e-5,
                "weight_decay": 1e-9,
                "cuda_aware": True,
            },
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "last_batch_iteration": -1,
                "warmup_min_lr": 0,
                "warmup_max_lr": 3e-5,
                "warmup_num_steps": 100,
            },
        },
        "zero_optimization": {
            "stage": 2,  # Enable Stage 2 ZeRO (Optimizer/Gradient state partitioning)
            "offload_optimizer": True,  # Enable Offloading optimizer state/calculation to the host CPU
            "contiguous_gradients": True,  # Reduce gradient fragmentation.
            "overlap_comm": True,  # Overlap reduce/backward operation of gradients for speed.
            "allgather_bucket_size": 2e8,  # Number of elements to all gather at once.
            "reduce_bucket_size": 2e8,  # Number of elements we reduce/allreduce at once.
        },
    }

    model = MyModel()
    trainer = Trainer(accelerator="gpu", devices=4, strategy=DeepSpeedStrategy(config=deepspeed_config), precision=16)
    trainer.fit(model)


We support taking the config as a json formatted file:

.. code-block:: python

    from pytorch_lightning import Trainer
    from pytorch_lightning.strategies import DeepSpeedStrategy

    model = MyModel()
    trainer = Trainer(
        accelerator="gpu", devices=4, strategy=DeepSpeedStrategy(config="/path/to/deepspeed_config.json"), precision=16
    )
    trainer.fit(model)


You can use also use an environment variable via your PyTorch Lightning script:

.. code-block:: bash

    PL_DEEPSPEED_CONFIG_PATH=/path/to/deepspeed_config.json python train.py --strategy deepspeed

----------

.. _ddp-optimizations:

DDP Optimizations
^^^^^^^^^^^^^^^^^


When Using DDP Strategies, Set find_unused_parameters=False
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

By default, we have set ``find_unused_parameters=True`` for compatibility reasons that have been observed in the past (refer to the `discussion <https://github.com/PyTorchLightning/pytorch-lightning/discussions/6219>`_ for more details).
When enabled, it can result in a performance hit and can be disabled in most cases. Read more about it `here <https://pytorch.org/docs/stable/notes/ddp.html#internal-design>`_.

.. tip::
    It applies to all DDP strategies that support ``find_unused_parameters`` as input.

.. code-block:: python

    from pytorch_lightning.strategies import DDPStrategy

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=2,
        strategy=DDPStrategy(find_unused_parameters=False),
    )

.. code-block:: python

    from pytorch_lightning.strategies import DDPSpawnStrategy

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=2,
        strategy=DDPSpawnStrategy(find_unused_parameters=False),
    )


DDP Static Graph
""""""""""""""""

`DDP static graph <https://pytorch.org/blog/pytorch-1.11-released/#stable-ddp-static-graph>`__ assumes that your model
employs the same set of used/unused parameters in every iteration, so that it can deterministically know the flow of
training and apply special optimizations during runtime.

.. note::
    DDP static graph support requires PyTorch>=1.11.0

.. code-block:: python

    from pytorch_lightning import Trainer
    from pytorch_lightning.strategies import DDPStrategy

    trainer = Trainer(devices=4, strategy=DDPStrategy(static_graph=True))


When Using DDP on a Multi-node Cluster, Set NCCL Parameters
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

`NCCL <https://developer.nvidia.com/nccl>`__ is the NVIDIA Collective Communications Library that is used by PyTorch to handle communication across nodes and GPUs. There are reported benefits in terms of speedups when adjusting NCCL parameters as seen in this `issue <https://github.com/PyTorchLightning/pytorch-lightning/issues/7179>`__. In the issue, we see a 30% speed improvement when training the Transformer XLM-RoBERTa and a 15% improvement in training with Detectron2.

NCCL parameters can be adjusted via environment variables.

.. note::

    AWS and GCP already set default values for these on their clusters. This is typically useful for custom cluster setups.

* `NCCL_NSOCKS_PERTHREAD <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-nsocks-perthread>`__
* `NCCL_SOCKET_NTHREADS <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-socket-nthreads>`__
* `NCCL_MIN_NCHANNELS <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-min-nchannels>`__

.. code-block:: bash

    export NCCL_NSOCKS_PERTHREAD=4
    export NCCL_SOCKET_NTHREADS=2


Gradients as Bucket View
""""""""""""""""""""""""

Enabling ``gradient_as_bucket_view=True`` in the ``DDPStrategy`` will make gradients views point to different offsets of the ``allreduce`` communication buckets. See :class:`~torch.nn.parallel.DistributedDataParallel` for more information.

This can reduce peak memory usage and throughput as saved memory will be equal to the total gradient memory + removes the need to copy gradients to the ``allreduce`` communication buckets.

.. note::

    When ``gradient_as_bucket_view=True`` you cannot call ``detach_()`` on gradients. If hitting such errors, please fix it by referring to the :meth:`~torch.optim.Optimizer.zero_grad` function in ``torch/optim/optimizer.py`` as a solution (`source <https://pytorch.org/docs/master/_modules/torch/nn/parallel/distributed.html#DistributedDataParallel>`__).

.. code-block:: python

    from pytorch_lightning import Trainer
    from pytorch_lightning.strategies import DDPStrategy

    model = MyModel()
    trainer = Trainer(accelerator="gpu", devices=4, strategy=DDPStrategy(gradient_as_bucket_view=True))
    trainer.fit(model)

DDP Communication Hooks
"""""""""""""""""""""""

DDP Communication hooks is an interface to control how gradients are communicated across workers, overriding the standard allreduce in DistributedDataParallel. This allows you to enable performance improving communication hooks when using multiple nodes.

Enable `FP16 Compress Hook for multi-node throughput improvement <https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook>`__:

.. code-block:: python

    from pytorch_lightning import Trainer
    from pytorch_lightning.strategies import DDPStrategy
    from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as default

    model = MyModel()
    trainer = Trainer(accelerator="gpu", devices=4, strategy=DDPStrategy(ddp_comm_hook=default.fp16_compress_hook))
    trainer.fit(model)

Enable `PowerSGD for multi-node throughput improvement <https://pytorch.org/docs/stable/ddp_comm_hooks.html#powersgd-communication-hook>`__:

.. note::

    PowerSGD typically requires extra memory of the same size as the model’s gradients to enable error feedback, which can compensate for biased compressed communication and improve accuracy (`source <https://pytorch.org/docs/stable/ddp_comm_hooks.html#powersgd-hooks>`__).

.. code-block:: python

    from pytorch_lightning import Trainer
    from pytorch_lightning.strategies import DDPStrategy
    from torch.distributed.algorithms.ddp_comm_hooks import powerSGD_hook as powerSGD

    model = MyModel()
    trainer = Trainer(
        accelerator="gpu",
        devices=4,
        strategy=DDPStrategy(
            ddp_comm_state=powerSGD.PowerSGDState(
                process_group=None,
                matrix_approximation_rank=1,
                start_powerSGD_iter=5000,
            ),
            ddp_comm_hook=powerSGD.powerSGD_hook,
        ),
    )
    trainer.fit(model)


Combine hooks for accumulated benefit:

.. note::
    DDP communication wrappers support requires PyTorch>=1.9.0

.. code-block:: python

    from pytorch_lightning import Trainer
    from pytorch_lightning.strategies import DDPStrategy
    from torch.distributed.algorithms.ddp_comm_hooks import (
        default_hooks as default,
        powerSGD_hook as powerSGD,
    )

    model = MyModel()
    trainer = Trainer(
        accelerator="gpu",
        devices=4,
        strategy=DDPStrategy(
            ddp_comm_state=powerSGD.PowerSGDState(
                process_group=None,
                matrix_approximation_rank=1,
                start_powerSGD_iter=5000,
            ),
            ddp_comm_hook=powerSGD.powerSGD_hook,
            ddp_comm_wrapper=default.fp16_compress_wrapper,
        ),
    )
    trainer.fit(model)


When using Post-localSGD, you must also pass ``model_averaging_period`` to allow for model parameter averaging:

.. note::
    Post-localSGD support requires PyTorch>=1.10.0

.. code-block:: python

    from pytorch_lightning import Trainer
    from pytorch_lightning.strategies import DDPStrategy
    from torch.distributed.algorithms.ddp_comm_hooks import post_localSGD_hook as post_localSGD

    model = MyModel()
    trainer = Trainer(
        accelerator="gpu",
        devices=4,
        strategy=DDPStrategy(
            ddp_comm_state=post_localSGD.PostLocalSGDState(
                process_group=None,
                subgroup=None,
                start_localSGD_iter=8,
            ),
            ddp_comm_hook=post_localSGD.post_localSGD_hook,
            model_averaging_period=4,
        ),
    )
    trainer.fit(model)
