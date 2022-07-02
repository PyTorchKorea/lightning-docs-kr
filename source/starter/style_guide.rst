################
스타일 가이드
################
PyTorch Lightning의 주요한 목표는 가독성(readability)과 재현성(reproducibility)을 향상시키는 것입니다. GitHub 저장소나 리서치 프로젝트에서
:class:`~pytorch_lightning.core.lightning.LightningModule` 을 발견하고, 필요한 부분이 어디에 있는지 찾아보기 위해 정확히 어디를 봐야 하는지
알 수 있다고 생각해보세요.

이 스타일 가이드의 목표는 Lightning의 코드가 유사하게 구성되도록 권장하기 위함입니다.

--------------

*****************
LightningModule
*****************

:class:`~pytorch_lightning.core.lightning.LightningModule` 클래스를 구성하는 모범 사례가 있습니다:

시스템(System)과 모델(Model)
==============================

.. figure:: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/model_system.png
    :width: 400

LightningModule의 주요한 원칙은 전체 시스템은 반드시 독립적(self-contained)이어야 한다는 것입니다.
Lightning에서는 시스템(system)과 모델(model)을 구분합니다.

모델은 ResNet18, RNN 등과 같은 것입니다.

시스템은 모델들(a collection of models)이 사용자가 정의한 학습/검증 로직을 사용하여 어떻게 상호작용하는지에 대해 정의합니다.
이에 대한 예시는 다음과 같습니다:

* GAN
* Seq2Seq
* BERT
* 그 외

LightningModule은 시스템과 모델 모두를 정의할 수 있습니다:

다음은 LightningModule로 시스템을 정의하는 것으로, 모범 사례로 권장하는 구조입니다. 모델을 시스템으로부터 분리하면 모듈성(modularity)이
향상되어, 더 나은 테스팅에 도움이 되고 시스템에 의존성이 줄어들어 리팩토링이 더 쉬워집니다.

.. testcode::

    class Encoder(nn.Module):
        ...


    class Decoder(nn.Module):
        ...


    class AutoEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Encoder()
            self.decoder = Decoder()

        def forward(self, x):
            return self.encoder(x)


    class AutoEncoderSystem(LightningModule):
        def __init__(self):
            super().__init__()
            self.auto_encoder = AutoEncoder()


빠른 프로토타이핑을 위해서는 모든 연산을 LightningModule 내에 정의하는 것이 유용합니다. 재현성과 확장성을
위해서는 관련된 백본(backbone)에 전달하는 것이 더 나을 수 있습니다.

다음은 LightningModule로 모델을 정의하는 것이지만, 아래 예시처럼 모델을 정의하는 것은 권장하지 않습니다.

.. testcode::

    class LitModel(LightningModule):
        def __init__(self):
            super().__init__()
            self.layer_1 = nn.Linear()
            self.layer_2 = nn.Linear()
            self.layer_3 = nn.Linear()


독립성(self-contained)
===========================

LightningModule은 반드시 독립적(self-contained)이어야 합니다. 모델이 독립적인지를 확인해보는 좋은 방법 중에 하나는,
다음과 같이 스스로에게 물어보는 것입니다:

"누군가 내부에 대해서 전혀 모르는 상태에서 이 파일을 Trainer에 사용(drop)할 수 있을까?"

예를 들어, 주요한 모델들은 특정 옵티마이저(optimizer)와 학습율 스케쥴러(learning rate scheduler)에서 잘 동작하기 때문에
옵티마이저는 모델과 결합(couple)합니다.

초기화(init)
=================

LightningModule이 독립적이지 않게 되는 첫번째 위치는 초기화(init) 부분입니다. 사용자가 추측할 필요가 없도록 초기화 부분에
모든 관련된 적절한 기본값(sensible defaults)을 정의해주세요.

다음은 이 LightningModule이 어떻게 초기화되었는지 알아보기 위해 사용자가 파일을 찾아봐야만 하는 예시입니다.

.. testcode::

    class LitModel(LightningModule):
        def __init__(self, params):
            self.lr = params.lr
            self.coef_x = params.coef_x

이렇게 정의된 모델은 많은 궁금증들, 예를 들면 ``coef_x`` 는 무엇인지? 문자열인지? 실수(float)인지? 범위는 어떻게 되는지? 를
갖게 합니다. 이렇게 하는 대신, 명시적으로 초기화를 하는 것이 좋습니다.

.. testcode::

    class LitModel(LightningModule):
        def __init__(self, encoder: nn.Module, coef_x: float = 0.2, lr: float = 1e-3):
            ...

이제 사용자는 추측할 필요가 없습니다. 값의 타입(type) 뿐만 아니라, 모델에는 사용자가 즉시 확인할 수 있는 적절한 기본값도
존재합니다.


메소드 순서
============
LightningModule에서 필요로 하는 메소드들은 다음의 것들 뿐입니다:

* init
* training_step
* configure_optimizers

하지만, 다른 선택적인 메소드들을 구현하기로 마음먹었다면, 권장하는 순서는 다음과 같습니다:

* 모델/시스템 정의 (초기화)
* 추론(inerence)을 한다면, forward 정의
* 학습용 훅들(training hooks)
* 검증용 훅들(validation hooks)
* 테스트용 훅들(test hooks)
* 예측용 훅들(predict hooks)
* 옵티마이저 설정(configure_optimizers)
* 다른 훅(hook)들

실제 코드는 다음과 같습니다:

.. code-block::

    class LitModel(pl.LightningModule):

        def __init__(...):

        def forward(...):

        def training_step(...):

        def training_step_end(...):

        def training_epoch_end(...):

        def validation_step(...):

        def validation_step_end(...):

        def validation_epoch_end(...):

        def test_step(...):

        def test_step_end(...):

        def test_epoch_end(...):

        def configure_optimizers(...):

        def any_extra_hook(...):


forward와 training_step
========================

:meth:`~pytorch_lightning.core.lightning.LightningModule.forward` 는 추론/예측을 위해 사용하고,
:meth:`~pytorch_lightning.core.lightning.LightningModule.training_step` 를 독립적으로 유지하는 것을 추천합니다.

.. code-block:: python

    def forward(self, x):
        embeddings = self.encoder(x)
        return embeddings


    def training_step(self, batch, batch_idx):
        x, _ = batch
        z = self.encoder(x)
        pred = self.decoder(z)
        ...


--------------

************
데이터
************

데이터를 다루는 모범 사례입니다.

DataLoader
==============

Lightning은 :class:`~torch.utils.data.DataLoader` 를 사용해서 시스템 전반의 모든 데이터 흐름을 다룹니다. DataLoader를 구성할 때는
최대의 효율을 위해 워커(worker)의 수를 반드시 적절하게 조절해야 합니다.

.. warning:: 코드가 병목을 일으킬 수 있으므로 DataLoader에서 ``Trainer(strategy="ddp_spawn")`` 를 ``num_workers>0`` 로 사용하지 않도록 주의하세요.

DataModule
==============

:class:`~pytorch_lightning.core.datamodule.LightningDataModule` 은 데이터 관련된 훅들을 :class:`~pytorch_lightning.core.lightning.LightningModule` 로부터
분리하도록 설계되어 데이터셋에 구애받지 않는 데이터셋을 만들 수 있습니다. 이렇게 하면 모델이 서로 다른 데이터셋을 사용하도록 언제든지 교체(hot swap)할 수 있어,
여러 분야(domain)에서 테스트와 벤치마킹을 할 수 있습니다. 또한 프로젝트들 간에 정확한 데이터 분할(split)과 변환(transform)을 공유하고 재사용 할 수 있게 합니다.

Lightning에서의 데이터 관리 방법과 모범 사례는 :ref:`data` 문서를 참고하세요.

* 어떠한 데이터 분할(split) 방법이 사용되었나요?
* 전체와 분할된 데이터셋 각각에는 몇 개의 샘플이 있나요?
* 어떠한 변환(transform) 방법이 사용되었나요?

이러한 이유들 때문에 DataModule을 사용하기 권하고 있습니다. 이는 협업할 때 팀의 시간을 많이 절약할 수 있기에 특히 중요합니다.

사용자들은 DataModule을 Trainer에 던져놓기만 하고, 데이터에 어떠한 작업이 수행되는지는 신경쓰지 않아도 됩니다.

이는 데이터의 정제(cleaning)와 특정 목적의 작업(ad-hoc instruction) 때문에 아이디어를 반복하는 과정이 느려지는 학계(academic)나 기업(corporate) 모두에
해당됩니다.

- 직접 손으로 따라해볼 수 있는 예제들입니다:
- `Introduction to PyTorch Lightning <https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/mnist-hello-world.html>`_
- `Introduction to DataModules <https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/datamodules.html>`_
