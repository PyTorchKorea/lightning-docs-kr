###########################################
LightningLite (Lightning을 위한 디딤돌)
###########################################


:class:`~pytorch_lightning.lite.LightningLite` 는 PyTorch 사용자들이 기존 반복(loop) / 최적화 로직을
완벽하게 제어하면서 기존 코드를 모든 종류의 장치에서 사용 가능하도록 확장할 수 있도록 합니다.

.. image:: https://pl-public-data.s3.amazonaws.com/docs/static/images/lite/lightning_lite.gif
    :alt: PyTorch 코드를 LightningLite로 변환하는 방법을 보여주는 애니메이션.
    :width: 500
    :align: center

|

아래 설명들 중 하나에 해당한다면 :class:`~pytorch_lightning.lite.LightningLite` 가 바로 적합한 도구입니다:

- 기존 코드에 최소한의 변경만으로 여러 장치로 빠르게 확장하고 싶습니다.
- 기존 코드를 Lightning API로 변환하고 싶지만, Lightning으로의 완벽한 전환 과정(full path)이 다소 복잡할 것 같습니다.
  전환하는 동안 재현성(reproducibility)을 보장하기 위한 디딤돌(stepping stone)을 찾고 있습니다.


.. warning:: :class:`~pytorch_lightning.lite.LightningLite` 은 현재 beta 기능입니다. 사용자 피드백에 따라 API가 변경될 수 있습니다.


----------

****************
예제로 배우기
****************


기존 PyTorch 코드
========================

``run`` 함수는 ``MyModel`` 학습을 위해 ``MyDataset`` 을 ``num_epochs`` 에폭(epoch)만큼 반복하는 사용자 정의 학습 루프(loop)를 포함하고 있습니다.

.. code-block:: python

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset


    class MyModel(nn.Module):
        ...


    class MyDataset(Dataset):
        ...


    def run(args):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = MyModel(...).to(device)
        optimizer = torch.optim.SGD(model.parameters(), ...)

        dataloader = DataLoader(MyDataset(...), ...)

        model.train()
        for epoch in range(args.num_epochs):
            for batch in dataloader:
                batch = batch.to(device)
                optimizer.zero_grad()
                loss = model(batch)
                loss.backward()
                optimizer.step()


    run(args)

----------


LightningLite로 변환하기
==========================

:class:`~pytorch_lightning.lite.LightningLite` 로 변환하기 위해 필요한 다섯 단계는 다음과 같습니다.

1. :class:`~pytorch_lightning.lite.LightningLite` 를 상속(subclass)받아 :meth:`~pytorch_lightning.lite.LightningLite.run` 메소드를 재정의합니다.
2. 기존 ``run`` 함수의 내용을 :class:`~pytorch_lightning.lite.LightningLite` 의 ``run`` 메소드로 이동합니다.
3. ``.to(...)``, ``.cuda()`` 등과 같은 모든 호출을 제거합니다. :class:`~pytorch_lightning.lite.LightningLite` 가 자동으로 이를 처리할 것입니다.
4. 각 모델과 옵티마이저(optimizer) 쌍에는 :meth:`~pytorch_lightning.lite.LightningLite.setup` 을, 모든 데이터로더(dataloader)에는 :meth:`~pytorch_lightning.lite.LightningLite.setup_dataloaders` 을 적용하고, ``loss.backward()`` 를 ``self.backward(loss)`` 로 변경합니다.
5. :class:`~pytorch_lightning.lite.LightningLite` 를 상속받은 서브클래스를 객체화(instantiate)한 뒤 :meth:`~pytorch_lightning.lite.LightningLite.run` 메소드를 호출합니다.

|

.. code-block:: python

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from pytorch_lightning.lite import LightningLite


    class MyModel(nn.Module):
        ...


    class MyDataset(Dataset):
        ...


    class Lite(LightningLite):
        def run(self, args):

            model = MyModel(...)
            optimizer = torch.optim.SGD(model.parameters(), ...)
            model, optimizer = self.setup(model, optimizer)  # 모델 / 옵티마이저(optimizer) 확장

            dataloader = DataLoader(MyDataset(...), ...)
            dataloader = self.setup_dataloaders(dataloader)  # 데이터로더(dataloader) 확장

            model.train()
            for epoch in range(args.num_epochs):
                for batch in dataloader:
                    optimizer.zero_grad()
                    loss = model(batch)
                    self.backward(loss)  # loss.backward() 대체
                    optimizer.step()


    Lite(...).run(args)


이게 전부입니다. 이제 모든 종류의 장치에서 학습하고 확장할 수 있습니다. LightningLite를 사용한 전체 MNIST 학습 예제는 `여기 <https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/basic_examples/mnist_examples/image_classifier_2_lite.py>`_ 에서 확인할 수 있습니다.

:class:`~pytorch_lightning.lite.LightningLite` 가 장치를 관리하므로, 사용자가 관리하지 않아도 됩니다.
코드 내에 특정 장치용 로직이 있다면 삭제해야 합니다.

다음은 8개의 GPU에서 `torch.bfloat16 <https://pytorch.org/docs/1.10.0/generated/torch.Tensor.bfloat16.html>`_ 정밀도(precision)로 학습을 하는 방법을 보여줍니다:

.. code-block:: python

    Lite(strategy="ddp", devices=8, accelerator="gpu", precision="bf16").run(10)

`DeepSpeed Zero3 <https://www.deepspeed.ai/news/2021/03/07/zero3-offload.html>`_ 를 사용하여 8개의 GPU와 정밀도 16으로 학습하는 방법은 다음과 같습니다:

.. code-block:: python

    Lite(strategy="deepspeed", devices=8, accelerator="gpu", precision=16).run(10)

나아가 :class:`~pytorch_lightning.lite.LightningLite` 가 알아서 해주기도 합니다!

.. code-block:: python

    Lite(devices="auto", accelerator="auto", precision=16).run(10)

필요한 경우 분산-집합(distributed collectives)을 사용할 수도 있습니다.
다음은 (8개의 GPU x 32개 노드의) GPU 256개에서 실행하는 예제입니다.

.. code-block:: python

    class Lite(LightningLite):
        def run(self):

            # Transfer and concatenate tensors across processes
            self.all_gather(...)

            # Transfer an object from one process to all the others
            self.broadcast(..., src=...)

            # The total number of processes running across all devices and nodes.
            self.world_size

            # The global index of the current process across all devices and nodes.
            self.global_rank

            # The index of the current process among the processes running on the local node.
            self.local_rank

            # The index of the current node.
            self.node_rank

            # Wether this global rank is rank zero.
            if self.is_global_zero:
                # do something on rank 0
                ...

            # Wait for all processes to enter this call.
            self.barrier()


    Lite(strategy="ddp", devices=8, num_nodes=32, accelerator="gpu").run()


사용자 지정 데이터 또는 모델에 장치 할당이 필요한 경우, 데이터에는 ``self.setup_dataloaders(..., move_to_device=False)`` 를 하고
모델에는 ``self.setup(..., move_to_device=False)`` 를 함으로써 :class:`~pytorch_lightning.lite.LightningLite` 의 자동 배치를
비활성화할 수 있습니다.
뿐만 아니라, ``self.device`` 로 현재 장치에 접근하거나 :meth:`~pytorch_lightning.lite.LightningLite.to_device` 를 사용하여
객체를 현재 장치로 이동할 수 있습니다.


.. note:: 큰 모델들은 out-of-memory(메모리 부족) 에러가 발생하므로 :meth:`~pytorch_lightning.lite.LightningLite.run` 에서 모델을 생성(instantiate)하는 것을 권장합니다.

.. tip::

    :meth:`~pytorch_lightning.lite.LightningLite.run` 함수 내에 수백에서 수천 라인의 코드가 있고 이에 대해 확신이 서지 않는다면,
    적절한 느낌입니다. 2019년에 :class:`~pytorch_lightning.core.lightning.LightningModule` 이 점점 커지면서 개발자들 또한 같은 느낌을 받았고,
    이에 따라 단순성(simplicity)과 상호운용성(interoperability), 표준화(standardization)를 위해 코드를 구성하기 시작했습니다.
    이러한 느낌은 코드 리팩토링(refactoring)과 함께 / 또는 :class:`~pytorch_lightning.core.lightning.LightningModule` 으로 완전히 전환하는 것을
    고려해봐야 한다는 좋은 신호입니다.


----------


분산 학습 시의 함정(pitfall)
=============================

:class:`~pytorch_lightning.lite.LightningLite` 는 학습을 확장할 수 있는 도구들을 제공하지만, 직면해야 할 몇 가지 주요한 과제들도 있습니다:


.. list-table::
   :widths: 50 50
   :header-rows: 0

   * - 프로세스 발산(Processes divergence)
     - 이전 파일 또는 다른 이유에서 서로 다른 if/else 조건, 경쟁 조건(race condition)으로 프로세스가 코드의 다른 부분(section) 실행하여 멈출(hanging) 때 발생합니다.
   * - 프로세스 간 리듀스(Cross processes reduction)
     - 리듀스 과정(reduction)에서의 오류로 메트릭(metric) 또는 변화도(gradient)가 잘못 계산되었습니다.
   * - 대규모의 샤딩된 모델(Large sharded models)
     - 대규모 모델의 생성(instantiation)과 구현(materialization), 상태 관리(state management).
   * - 순서가 0뿐인 작업(Rank 0 only actions)
     - 로깅(logging), 프로파일링(profiling) 등.
   * - 체크포인팅 / 조기 중단 / 콜백 / 로깅 (Checkpointing / Early stopping / Callbacks / Logging)
     - 학습 과정을 쉽게 사용자 정의하고 상태를 관리할 수 있는 기능.
   * - 결함-감내 학습(Fault-tolerant training)
     - 오류 발생 시에 마치 오류가 없었던 것처럼 재개(resume)하는 기능.


위와 같은 과제들 중 하나를 맞이했다면, 이제 :class:`~pytorch_lightning.lite.LightningLite` 의 한계를 마주한 것입니다.
이러한 걱정을 할 필요가 없는 :doc:`Lightning <../starter/introduction>` 으로 변환하는 것을 추천합니다.

----------

Lightning으로의 변환
======================

:class:`~pytorch_lightning.lite.LightningLite` 은 수백가지 기능을 갖는 Lightning API로의 완전한 전환을 위한 디딤돌입니다.

:class:`~pytorch_lightning.lite.LightningLite` 클래스 자체를 :class:`~pytorch_lightning.core.lightning.LightningModule` 의 개선된 버전(future)로 볼 수도 있으므로,
해당 API로 코드를 천천히 재구성(refactor)해보겠습니다.
아래에는 :meth:`~pytorch_lightning.core.lightning.LightningModule.training_step` 와 :meth:`~pytorch_lightning.core.lightning.LightningModule.forward`,
:meth:`~pytorch_lightning.core.lightning.LightningModule.configure_optimizers`, :meth:`~pytorch_lightning.core.lightning.LightningModule.train_dataloader` 메서드들이
구현되어 있습니다.


.. code-block:: python

    class Lite(LightningLite):

        # 1. 이 부분은 LightningModule의 `__init__` 함수가 됩니다.
        def run(self, args):
            self.args = args

            self.model = MyModel(...)

            self.fit()  # 이는 Lightning Trainer에 의해 자동화됩니다.

        # 2. Lightning이 자체적인 학습 루프(fitting loop)를 생성하고,
        # 모델, 옵티마이저, 데이터로더 등을 설정하므로 이 코드는 완전히 제거해도 됩니다.
        def fit(self):
            # 필요한 것들을 설정
            optimizer = self.configure_optimizers()
            self.model, optimizer = self.setup(self.model, optimizer)
            dataloader = self.setup_dataloaders(self.train_dataloader())

            # 학습(fitting) 시작
            self.model.train()
            for epoch in range(num_epochs):
                for batch in enumerate(dataloader):
                    optimizer.zero_grad()
                    loss = self.training_step(batch, batch_idx)
                    self.backward(loss)
                    optimizer.step()

        # 3. 이는 LightningModule에 속하므로 그대로 둡니다.
        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx):
            return self.forward(batch)

        def configure_optimizers(self):
            return torch.optim.SGD(self.model.parameters(), ...)

        # 4. [선택사항] 이는 그대로 두거나, LightningDataModule이 더 높은 결합성(composability)을 갖도록 따로 분리(extract)할 수도 있습니다.
        def train_dataloader(self):
            return DataLoader(MyDataset(...), ...)


    Lite(...).run(args)


마지막으로, :meth:`~pytorch_lightning.lite.LightningLite.run` 을 :meth:`~pytorch_lightning.core.lightning.LightningModule.__init__` 으로
바꾸고, 내부의 ``fit`` 호출 부분을 삭제합니다.

.. code-block:: python

    from pytorch_lightning import LightningDataModule, LightningModule, Trainer


    class LightningModel(LightningModule):
        def __init__(self, args):
            super().__init__()
            self.model = MyModel(...)

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx):
            loss = self(batch)
            self.log("train_loss", loss)
            return loss

        def configure_optimizers(self):
            return torch.optim.SGD(self.model.parameters(), lr=0.001)


    class BoringDataModule(LightningDataModule):
        def train_dataloader(self):
            return DataLoader(MyDataset(...), ...)


    trainer = Trainer(max_epochs=10)
    trainer.fit(LightningModel(), datamodule=BoringDataModule())


이제 수백가지 기능들의 이점을 누릴 수 있는 PyTorch Lightning으로의 변환을 성공적으로 완료하였습니다!

----------

********************************
Lightning Lite 매개변수(flag)
********************************

Lite는 가속화된 분산 학습 및 추론(inference)에 특화되어 있습니다. 이는 장치 및 통신 전략을 손쉽게 구성하고,
다른 방식으로의 원활하게 전환할 수 있는 편리한 방법을 제공합니다. 용어(terminology) 및 사용법이 Lightning과
동일하므로, 변환을 결심했을 때 변환에 드는 노력을 최소화할 수 있습니다.


accelerator (가속기 종류)
==========================

``"cpu"``, ``"gpu"``, ``"tpu"``, ``"auto"`` 중 하나를 선택합니다 (IPU는 곧 제공 예정입니다).

.. code-block:: python

    # CPU 가속기
    lite = Lite(accelerator="cpu")

    # 2개의 GPU 가속기에서 실행
    lite = Lite(devices=2, accelerator="gpu")

    # 8개의 TPU 가속기에서 실행
    lite = Lite(devices=8, accelerator="tpu")

    # DistributedDataParallel(ddp) 전략으로 GPU 가속기에서 실행
    lite = Lite(devices=4, accelerator="gpu", strategy="ddp")

``"auto"`` 옵션은 사용 중인 기기를 인식하고 사용 가능한 가속기를 선택합니다.

.. code-block:: python

    # 기기에 GPU가 있으면, GPU 가속기를 사용합니다.
    lite = Lite(devices=2, accelerator="auto")


strategy (학습 전략)
======================

학습 전략을 선택합니다: ``"dp"``, ``"ddp"``, ``"ddp_spawn"``, ``"tpu_spawn"``, ``"deepspeed"``, ``"ddp_sharded"``, 또는 ``"ddp_sharded_spawn"``

.. code-block:: python

    # 4개의 GPU에서 DistributedDataParallel 전략 사용
    lite = Lite(strategy="ddp", accelerator="gpu", devices=4)

    # 4개의 CPU에서 DDP Spawn 전략 사용
    lite = Lite(strategy="ddp_spawn", accelerator="cpu", devices=4)


또한, 몇몇 매개변수를 추가로 설정해서 사용자 지정 전략을 사용할 수 있습니다.

.. code-block:: python

    from pytorch_lightning.strategies import DeepSpeedStrategy

    lite = Lite(strategy=DeepSpeedStrategy(stage=2), accelerator="gpu", devices=2)


Horovoard 및 Full Sharded 학습 전략은 곧 지원될 예정입니다.


device (장치)
==============

실행할 장치를 설정합니다. 아래와 같은 자료형일 수 있습니다:

- int: 학습할 장치(예. GPU)의 개수
- list of int: 학습할 장치의 인덱스(예. GPU ID, 0-indexed)
- str: 위 중 하나의 문자열 표현

.. code-block:: python

    # Lite에서 사용하는 기본 값, CPU에서 실행
    lite = Lite(devices=None)

    # 위와 동일
    lite = Lite(devices=0)

    # int: 2개의 GPU에서 실행
    lite = Lite(devices=2, accelerator="gpu")

    # list: GPU 1, 4에서 실행 (버스 순서에 따름)
    lite = Lite(devices=[1, 4], accelerator="gpu")
    lite = Lite(devices="1, 4", accelerator="gpu")  # 위와 동일

    # -1: 모든 GPU에서 실행
    lite = Lite(devices=-1, accelerator="gpu")
    lite = Lite(devices="-1", accelerator="gpu")  # 위와 동일



gpus (사용하지 않음)
=======================

.. warning:: ``gpus=x`` 는 v1.7에서 더 이상 사용하지 않으며(deprecated), v2.0에서 제거될 예정입니다.
    대신에 ``accelerator='gpu'`` 및 ``devices=x`` 을 사용하십시오.

``devices=X`` 및 ``accelerator="gpu"`` 의 약어(shorthand).

.. code-block:: python

    # 2개의 GPU에서 실행
    lite = Lite(accelerator="gpu", devices=2)

    # 위와 동일
    lite = Lite(devices=2, accelerator="gpu")


tpu_cores (사용하지 않음)
============================

.. warning:: ``tpu_cores=x`` 는 v1.7에서 더 이상 사용하지 않으며(deprecated), v2.0에서 제거될 예정입니다.
    대신에 ``accelerator='tpu'`` 및 ``devices=x`` 을 사용하십시오.

``devices=X`` 및 ``accelerator="tpu"`` 의 약어.

.. code-block:: python

    # 8개의 TPU에서 실행
    lite = Lite(accelerator="tpu", devices=8)

    # 위와 동일
    lite = Lite(devices=8, accelerator="tpu")


num_nodes (노드의 수)
====================================

분산 작업 시의 클러스터 노드의 수.

.. code-block:: python

    # Lite에서 사용하는 기본값
    lite = Lite(num_nodes=1)

    # 8개의 노드에서 실행
    lite = Lite(num_nodes=8)


클러스터에서의 분산 다중 노드 학습에 대해서는 :doc:`이 문서 <../clouds/cluster>` 에서 자세히 알아볼 수 있습니다.


precision (정밀도)
=====================

Lightning Lite는 배정밀도(double precision; 64), 단정밀도(full precision; 32), 또는 반정밀도(half precision; 16) 연산(`bfloat16 <https://pytorch.org/docs/1.10.0/generated/torch.Tensor.bfloat16.html>`_ 포함)을 지원합니다.
반정밀도 또는 혼합 정밀도(mixed precision)는 32비트 정밀도와 16비트 정밀도를 합쳐서 사용하여 모델 학습 시의 메모리 공간(footprint)을 줄입니다.
그 결과 성능이 향상되어 최신 GPU에서 눈에 띄게 성능이 향상됩니다.

.. code-block:: python

    # Lite에서 사용하는 기본값
    lite = Lite(precision=32, devices=1)

    # 16-비트 (혼합) 정밀도
    lite = Lite(precision=16, devices=1)

    # 16-비트 bfloat 정밀도
    lite = Lite(precision="bf16", devices=1)

    # 64-비트 (배(double)) 정밀도
    lite = Lite(precision=64, devices=1)


plugins (플러그인)
=====================

:ref:`Plugins` 을 사용하여 임의의 백엔드(backend), 정밀도 라이브러리, 클러스터 등을 연결할 수 있습니다.
예: 임의의 동작을 정의하고 싶으면 관련 클래스를 상속받아 전달하면 됩니다. 다음은 직접 만든
:class:`~pytorch_lightning.plugins.environments.ClusterEnvironment` 를 연결하는 예시입니다.

.. code-block:: python

    from pytorch_lightning.plugins.environments import ClusterEnvironment


    class MyCluster(ClusterEnvironment):
        @property
        def main_address(self):
            return your_main_address

        @property
        def main_port(self):
            return your_main_port

        def world_size(self):
            return the_world_size


    lite = Lite(plugins=[MyCluster()], ...)


----------


**********************
Lightning Lite 메소드
**********************


run
====

run 메소드는 2가지 용도로 사용합니다:

1.  :class:`~pytorch_lightning.lite.lite.LightningLite` 클래스에서 이 메시드를 재정의(override)하고
    학습(또는 추론) 코드를 내부에 넣습니다.
2.  run 메소드를 호출하여 학습 절차를 시작합니다. Lite는 분산 백엔드 설정을 처리합니다.

선택적으로 run 메소드에 인자(예를 들어 모델의 하이퍼파라매터나 백엔드)를 전달할 수 있습니다.

.. code-block:: python

    from pytorch_lightning.lite import LightningLite


    class Lite(LightningLite):

        # 입력 인자는 선택 사항입니다; 필요 시에 넣으세요.
        def run(self, learning_rate, num_layers):
            """여기에 학습 과정이 들어갑니다."""


    lite = Lite(accelerator="gpu", devices=2)
    lite.run(learning_rate=0.01, num_layers=12)


setup
======

모델 및 해당하는 옵티마이저(들)을 설정합니다. 여러 모델을 설정해야 하는 경우, 각각에 대해서 ``setup()`` 을 호출하십시오.
모델과 옵티마이저는 적절한 장치로 자동으로 이동합니다.

.. code-block:: python

    model = nn.Linear(32, 64)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    # 가속화된 학습을 위해 모델 및 옵티마이저 설정
    model, optimizer = self.setup(model, optimizer)

    # Lite가 장치를 설정하는 것을 원치 않는 경우
    model, optimizer = self.setup(model, optimizer, move_to_device=False)


setup 메소드는 선택한 정밀도로 모델을 준비하여 ``forward()`` 중 연산들이 자동으로 변환(cast)되도록 합니다.

setup_dataloaders
=================

가속화된 연산을 위해 하나 이상의 데이터로더를 설정합니다. 분산 전략(예. DDP)을 사용하는 경우, Lite는 자동으로 샘플러(sampler)를
대체합니다. 또한, 데이터로더는 반환된 데이터 텐서를 적절한 장치로 자동으로 이동하도록 설정됩니다.

.. code-block:: python

    train_data = torch.utils.DataLoader(train_dataset, ...)
    test_data = torch.utils.DataLoader(test_dataset, ...)

    train_data, test_data = self.setup_dataloaders(train_data, test_data)

    # Lite가 데이터를 자동으로 장치로 이동시키는 것을 원치 않는 경우
    train_data, test_data = self.setup_dataloaders(train_data, test_data, move_to_device=False)

    # Lite가 분산 학습 도중 샘플러를 대체하기를 원치 않는 경우
    train_data, test_data = self.setup_dataloaders(train_data, test_data, replace_sampler=False)


backward
===========

``loss.backward()`` 을 대체하여 정밀도와 가속기 코드를 신경쓰지 않도록(agnostic) 합니다.

.. code-block:: python

    output = model(input)
    loss = loss_fn(output, target)

    # loss.backward()
    self.backward(loss)


to_device
=========

:meth:`~pytorch_lightning.lite.lite.LightningLite.to_device` 를 사용하여 모델 또는 텐서, 텐서 컬렉션을 현재 장치로 이동합니다.
기본적으로 :meth:`~pytorch_lightning.lite.lite.LightningLite.setup` 및 :meth:`~pytorch_lightning.lite.lite.LightningLite.setup_dataloaders` 가
모델과 데이터를 적절한 장치로 이동했으므로, 이 메소드는 수동 작업이 필요할 때만 사용합니다.

.. code-block:: python

    data = torch.load("dataset.pt")
    data = self.to_device(data)


seed_everything
===============

run의 시작 부분에 이 메소드를 호출하여 코드를 재현 가능하도록 합니다.

.. code-block:: python

    # `torch.manual_seed(...)` 대신 다음을 호출:
    self.seed_everything(1234)


이는 PyTorch 및 NumPy, Python 난수 생성기를 포괄합니다. 또한, Lite는 데이터로더 워커(worker) 프로세서의 시드(seed)를 적절히 초기화합니다.
(``workers=False`` 를 전달하여 이 기능을 끌 수 있습니다.)


autocast
========

정밀도 백엔드가 autocast 컨텍스트 매니저 내부의 코드 블록을 자동으로 캐스팅하도록 합니다. 이는 선택사항이며, Lite가
(모델이 :meth:`~pytorch_lightning.lite.lite.LightningLite.setup` 될 때) 이미 모델의 forward 메소드에 적용하였습니다
모델 forward 메소드 외부의 추가 연산들에 대해 자동으로 캐스팅하려는 경우에만 사용합니다:

.. code-block:: python

    model, optimizer = self.setup(model, optimizer)

    # Lite가 모델의 정밀도를 자동으로 처리합니다
    output = model(inputs)

    with self.autocast():  # 선택 사항
        loss = loss_function(output, target)

    self.backward(loss)
    ...


print
=====

내장 print 함수를 통해 콘솔에 출력하지만, 메인 프로세스(main process)에서만 가능합니다.
이는 여러 장치/노드에서 실행할 때 과도한 출력 및 로그를 방지합니다.


.. code-block:: python

    # 메인 프로세스에서만 출력
    self.print(f"{epoch}/{num_epochs}| Train Epoch Loss: {loss}")


save
====

체크포인트(checkpoint)에 내용을 저장합니다. 기존의 ``torch.save(...)`` 를 모두 대체합니다. Lite는 단일 장치나 다중 장치,
다중 노드 중 어디에서 실행하던지 잘 저장될 수 있도록 처리합니다.

.. code-block:: python

    # `torch.save(...)` 대신 다음을 호출:
    self.save(model.state_dict(), "path/to/checkpoint.ckpt")


load
====

파일로부터 체크포인트 내용을 불러옵니다. 기존의 ``torch.load(...)`` 를 모두 대체합니다. Lite는 단일 장치나 다중 장치,
다중 노드 중 어디에서 실행하던지 잘 불러올 수 있도록 처리합니다.

.. code-block:: python

    # `torch.load(...)` 대신 다음을 호출:
    self.load("path/to/checkpoint.ckpt")


barrier
=======

모든 프로세스들이 대기해였다가 동기화되길 원할 때 사용합니다. 모든 프로세스가 barrier 호출에 진입하면, 그 때 계속 실행합니다.
예를 들어 한 프로세스가 데이터를 다운로드해서 디스크에 쓰는 동안 다른 모든 프로세스들이 대기하도록 할 때 유용합니다.

.. code-block:: python

    # 한 프로세스에서만 데이터 다운로드
    if self.global_rank == 0:
        download_data("http://...")

    # 모든 프로세스가 여기서 만날 때까지 대기
    self.barrier()

    # 이제 모든 프로세스가 데이터를 읽을 수 있음
