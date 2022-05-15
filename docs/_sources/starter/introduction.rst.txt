:orphan:

###############################
Lightning 15분 만에 배워보기
###############################

**필요한 배경지식:** 없음

**목표:** 이 문서에서는 일반적인 Lightning 워크플로우의 주요한 7단계를 안내합니다.

PyTorch Lightning(파이토치 라이트닝)은 대규모로 엄청 빠른 성능을 요구하면서 최대한의 유연성을 필요로 하는
전문적인 AI 연구자들과 머신러닝 엔지니어들을 위한 "배터리가 포함된(batteries included)" 딥러닝 프레임워크입니다.

.. join_slack::
   :align: left
   :margin: 20

Lightning(라이트닝)은 반복적으로 사용하는 코드(boilerplate)를 제거하고 확장성(scalability)을 확보하도록 PyTorch 코드를 재구성합니다.

.. raw:: html

    <video width="100%" max-width="800px" controls autoplay muted playsinline
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/pl_docs_animation_final.m4v"></video>

|

PyTorch 코드를 재구성함으로써, Lightning에서는 이런 것들이 가능해집니다:

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. Add callout items below this line

.. displayitem::
   :header: 완전한 유연성
   :description: 반복되는 코드 없이 PyTorch를 그대로 사용하여 아이디어를 구현합니다.
   :col_css: col-md-3
   :image_center: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/card_full_control.png
   :height: 290

.. displayitem::
   :header: 재현성 + 가독성
   :description: 연구용 코드와 엔지니어링 코드를 분리하여 재현성을 갖추고 더 나은 가독성을 제공합니다.
   :col_css: col-md-3
   :image_center: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/card_no_boilerplate.png
   :height: 290

.. displayitem::
   :header: 간단한 다중 GPU 학습
   :description: 코드 변경 없이 여러개의 GPU/TPU/HPU 등을 사용합니다.
   :col_css: col-md-3
   :image_center: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/card_hardware.png
   :height: 290

.. displayitem::
   :header: 테스트 완료
   :description: 이미 모든 테스트를 완료하여 직접 테스트 할 필요없습니다.
   :col_css: col-md-3
   :image_center: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/card_testing.png
   :height: 290

.. raw:: html

        </div>
    </div>

.. End of callout item section

----

******************************
1: PyTorch Lightning 설치하기
******************************
.. raw:: html

   <div class="row" style='font-size: 16px'>
      <div class='col-md-6'>

`pip <https://pypi.org/project/pytorch-lightning/>`_ 사용자라면,

.. code-block:: bash

    pip install pytorch-lightning

.. raw:: html

      </div>
      <div class='col-md-6'>

`conda <https://anaconda.org/conda-forge/pytorch-lightning>`_ 사용자라면,

.. code-block:: bash

    conda install pytorch-lightning -c conda-forge

.. raw:: html

      </div>
   </div>

또는 `advanced install guide <installation.html>`_ 를 참조하세요.

----

.. _new_project:

*****************************
2: LightningModule 정의하기
*****************************

LightningModule을 사용하여 PyTorch nn.Module이 training_step (뿐만 아니라 validation_step이나 test_step) 내에서 복잡한 방식으로 함께 동작할 수 있도록 합니다.

.. testcode::

    import os
    from torch import optim, nn, utils, Tensor
    from tests.helpers.datasets import MNIST
    import pytorch_lightning as pl

    # 원하는만큼의 nn.Module (또는 기존 모델)을 정의합니다.
    encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
    decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

    # LightningModule을 정의합니다.
    class LitAutoEncoder(pl.LightningModule):
        def __init__(self, encoder, decoder):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder

        def training_step(self, batch, batch_idx):
            # training_step defines the train loop.
            # it is independent of forward
            x, y = batch
            x = x.view(x.size(0), -1)
            z = self.encoder(x)
            x_hat = self.decoder(z)
            loss = nn.functional.mse_loss(x_hat, x)
            # Logging to TensorBoard by default
            self.log("train_loss", loss)
            return loss

        def configure_optimizers(self):
            optimizer = optim.Adam(self.parameters(), lr=1e-3)
            return optimizer


    # 오토인코더(autoencoder)를 초기화합니다.
    autoencoder = LitAutoEncoder(encoder, decoder)

----

**********************
3: 데이터셋 정의하기
**********************

Lightning은 *어떠한* 순회 가능한 객체(iterable; :class:`~torch.utils.data.DataLoader`, numpy 등...)도 학습/검증/테스트/예측용으로 나누어 사용할 수 있습니다.

.. code-block:: python

    # 데이터를 설정합니다.
    dataset = MNIST(os.getcwd(), download=True)
    train_loader = utils.data.DataLoader(dataset)

----

******************
4: 모델 학습하기
******************

Lightning :doc:`Trainer <../common/trainer>` 는 모든 :doc:`LightningModule <../common/lightning_module>` 과 데이터셋을 "함께(mix)" 학습할 수 있으며,
확장에 필요한 모든 엔지니어링적 복잡성들을 추상화(abstract)합니다.

.. code-block:: python

    # 모델을 학습합니다 (힌트: 빠른 아이디어 반복에 도움이 되는 Trainer의 인자들을 참고하세요)
    trainer = pl.Trainer(limit_train_batches=100, max_epochs=1)
    trainer.fit(model=autoencoder, train_dataloaders=train_loader)

Lightning :doc:`Trainer <../common/trainer>` 는 아래 예시들을 포함하여 `40종류 이상의 기법들 <../common/trainer.html#trainer-flags>`_ 을 자동화합니다:

* 에폭(epoch) 및 배치(batch) 반복
* ``optimizer.step()``, ``loss.backward()``, ``optimizer.zero_grad()`` 호출
* 평가(evaluation) 도중 경사도(grads) 활성화/비활성화를 위한 ``model.eval()`` 호출
* :doc:`체크포인트(checkpoint) 저장하기 및 불러오기 <../common/checkpointing>`
* 텐서보드(tensorboard) (:doc:`loggers <../visualize/loggers>` 옵션 참조)
* :doc:`Multi-GPU <../accelerators/gpu>` 지원
* :doc:`TPU <../accelerators/tpu>`
* :ref:`16비트 정밀도(precision) AMP <speed-amp>` 지원

----


******************
5: 모델 사용하기
******************

모델을 학습한 뒤에는 ONNX, TorchScript로 내보내기(export)하여 상용 환경에 포함하거나 단순히 가중치를 불러오고 예측을 실행할 수 있습니다.

.. code:: python

    # 체크포인트(checkpoint)를 불러옵니다.
    checkpoint = "./lightning_logs/version_0/checkpoints/epoch=0-step=100.ckpt"
    autoencoder = LitAutoEncoder.load_from_checkpoint(checkpoint, encoder=encoder, decoder=decoder)

    # 학습한 nn.Module을 선택합니다.
    encoder = autoencoder.encoder
    encoder.eval()

    # 4개의 가짜 이미지로 예측(embed)합니다!
    fake_image_batch = Tensor(4, 28 * 28)
    embeddings = encoder(fake_image_batch)
    print("⚡" * 20, "\nPredictions (4 image embeddings):\n", embeddings, "\n", "⚡" * 20)

----

*********************
6: 학습 시각화하기
*********************

Lightning에는 *많은* 배터리가 포함되어 있습니다. 실험을 시각화하는데 사용하는 텐서보드(Tensorboard)도 유용한 도구 중 하나입니다.

명령줄(commandline)에서 아래를 실행하고 브라우저에서 **http://localhost:6006/** 을 열어보세요.

.. code:: bash

    tensorboard --logdir .

----

*************************
7: 엄청 빠르게 학습하기
*************************

Trainer에 인자(argument)를 사용하여 고급 학습 기능을 사용할 수 있습니다. 이는 다른 코드를 변경하지 않으면서 학습 단계(train loop)에 자동으로 통합할 수 있도록 하는 최신(state-of-the-art)의 기술입니다.

.. code::

   # 4개의 GPU에서 학습
   trainer = Trainer(
       devices=4,
       accelerator="gpu",
    )

   # Deepspeed/FSDP를 사용하여 1TB 이상의 매개변수를 갖는 모델 학습
   trainer = Trainer(
       devices=4,
       accelerator="gpu",
       strategy="deepspeed_stage_2",
       precision=16
    )

   # 빠른 아이디어 반복을 위한 20개 이상의 유용한 플래그(flag)
   trainer = Trainer(
       max_epochs=10,
       min_epochs=5,
       overfit_batches=1
    )

   # 최신 기술을 사용
   trainer = Trainer(callbacks=[StochasticWeightAveraging(...)])

----

********************
유연성 극대화하기
********************

Lightning의 핵심 원칙은 **PyTorch의 어떠한 부분도 숨기지 않으면서** 언제나 최대한의 유연성을 제공하는 것입니다.

Lightning은 프로젝트의 복잡도에 따라 *추가적인* 5단계의 유연성을 제공합니다.

----

학습 단계(loop) 사용자 정의하기
==================================

.. image:: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/custom_loop.png
    :width: 600
    :alt: Injecting custom code in a training loop

LightningModule에서 사용할 수 있는 20개 이상의 메소드 (:ref:`lightning_hooks`) 중 일부를 사용하여 훈련 단계 어디에든 사용자 정의 코드를 삽입할 수 있습니다.

.. testcode::

    class LitAutoEncoder(pl.LightningModule):
        def backward(self, loss, optimizer, optimizer_idx):
            loss.backward()

----

Trainer 확장하기
==================

.. raw:: html

    <video width="100%" max-width="800px" controls autoplay muted playsinline
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/cb.m4v"></video>

유사한 기능을 하는 여러줄의 코드가 있는 경우, 콜백(callback)을 사용하여 손쉽게 그룹으로 묶어서 해당하는 코드들을 동시에 켜거나 끌 수 있습니다.

.. code::

   trainer = Trainer(callbacks=[AWSCheckpoints()])

----

PyTorch 자체의 반복(loop) 사용하기
===================================

최첨단 연구 시 특정 유형의 작업들을 위해, Lightning은 전문가들이 다양한 방식으로 학습 단계를 완전히 제어할 수 있는 기능을 제공합니다.

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. Add callout items below this line

.. displayitem::
   :header: 직접 최적화(manual optimization)
   :description: 자동화된 학습 단계에서 최적화 단계는 사용자가 직접 관여합니다.
   :col_css: col-md-4
   :image_center: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/manual_opt.png
   :button_link: ../model/build_model_advanced.html#manual-optimization
   :image_height: 220px
   :height: 320

.. displayitem::
   :header: Lightning Lite(라이트닝 라이트)
   :description: 복잡한 PyTorch 프로젝트를 이관하기 위한 반복 단계를 완벽히 제어합니다.
   :col_css: col-md-4
   :image_center: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/lite.png
   :button_link: ../model/build_model_expert.html
   :image_height: 220px
   :height: 320

.. displayitem::
   :header: 반복(Loop)
   :description: 메타학습(meta-learning), 강화학습(reinforcement learning), GAN을 완벽히 제어합니다.
   :col_css: col-md-4
   :image_center: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/loops.png
   :button_link: ../extensions/loops.html
   :image_height: 220px
   :height: 320

.. raw:: html

        </div>
    </div>

.. End of callout item section

----

**********
다음 단계
**********

사용 사례에 따라, 아래 내용들 중 하나를 다음 단계로 살펴보세요.

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. Add callout items below this line

.. displayitem::
   :header: Level 2: Add a validation and test set
   :description: Add validation and test sets to avoid over/underfitting.
   :button_link: ../levels/basic_level_2.html
   :col_css: col-md-3
   :height: 180
   :tag: basic

.. displayitem::
   :header: See more examples
   :description: See examples across computer vision, NLP, RL, etc...
   :col_css: col-md-3
   :button_link: ../tutorials.html
   :height: 180
   :tag: basic

.. displayitem::
   :header: I need my raw PyTorch Loop
   :description: Expert-level control for researchers working on the bleeding-edge
   :col_css: col-md-3
   :button_link: ../model/build_model_expert.html
   :height: 180
   :tag: expert

.. displayitem::
   :header: Deploy your model
   :description: Learn how to predict or put your model into production
   :col_css: col-md-3
   :button_link: ../deploy/production.html
   :height: 180
   :tag: basic

.. raw:: html

        </div>
    </div>
