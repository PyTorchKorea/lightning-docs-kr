.. _converting:

######################################
PyTorch를 Lightning으로 구성하기
######################################

아래와 같이 PyTorch를 Lightning(라이트닝)으로 구성할 수 있습니다.

--------

******************************
1. 연산 코드 가져오기
******************************

일반적인 nn.Module 구조를 가져옵니다

.. testcode::

    import pytorch_lightning as pl
    import torch
    import torch.nn as nn
    import torch.nn.functional as F


    class LitModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_1 = nn.Linear(28 * 28, 128)
            self.layer_2 = nn.Linear(128, 10)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = self.layer_1(x)
            x = F.relu(x)
            x = self.layer_2(x)
            return x

--------

***************************
2. 학습 로직 구성하기
***************************
LightningModule의 training_step에 학습 데이터를 묶음(batch)으로 가져와 학습하는 과정을 구성합니다:

.. testcode::

    class LitModel(pl.LightningModule):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self.encoder(x)
            loss = F.cross_entropy(y_hat, y)
            return loss

.. note:: 기존 프로젝트가 복잡해서 기존의 학습 루프를 직접 구성해야 하면 :doc:`Own your loop <../model/own_your_loop>` 를 참조하세요.

----

****************************************
3. 옵티마이저와 LR스케줄러 이동하기
****************************************
옵티마이저(들)를 :meth:`~pytorch_lightning.core.lightning.LightningModule.configure_optimizers` 훅(hook)으로 이동합니다.

.. testcode::

    class LitModel(pl.LightningModule):
        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.encoder.parameters(), lr=1e-3)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            return [optimizer], [lr_scheduler]

--------

***************************************
4. (선택사항) 검증 로직 구성하기
***************************************
검증(validation) 루프가 필요하면, 검증 데이터를 묶음(batch)으로 가져와 검증하는 과정을 구성합니다:

.. testcode::

    class LitModel(pl.LightningModule):
        def validation_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self.encoder(x)
            val_loss = F.cross_entropy(y_hat, y)
            self.log("val_loss", val_loss)

.. tip:: 학습(fit) 중 체크포인트 기능이 켜진 경우 ``trainer.validate()`` 가 자동으로 최적의 체크포인트를 불러옵니다.

--------

************************************
5. (선택사항) 테스트 로직 구성하기
************************************
테스트(test) 루프가 필요하면, 테스트 데이터를 묶음(batch)으로 가져와 테스트하는 과정을 구성합니다:

.. testcode::

    class LitModel(pl.LightningModule):
        def test_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self.encoder(x)
            test_loss = F.cross_entropy(y_hat, y)
            self.log("test_loss", test_loss)

--------

****************************************
6. (선택사항) 예측 로직 구성하기
****************************************
예측(prediction) 루프가 필요하면, 테스트 데이터를 묶음(batch)으로 가져와 예측하는 과정을 구성합니다:

.. testcode::

    class LitModel(LightningModule):
        def predict_step(self, batch, batch_idx):
            x, y = batch
            pred = self.encoder(x)
            return pred

--------

******************************************
7. .cuda() 또는 .to(device) 호출 제거하기
******************************************

:doc:`LightningModule <../common/lightning_module>` 은 어떠한 하드웨어에서도 자동으로 실행됩니다!

``LightningModule.__init__`` 내에서 초기화된 :class:`~torch.nn.Module` 인스턴스들과 :class:`~torch.utils.data.DataLoader` 에서 가져온 데이터는
Lightning이 자동으로 해당 장치로 이동해서 실행하므로, 기존에 명시적으로 ``.cuda()`` 또는 ``.to(device)`` 을 호출하는 부분은 제거해도 됩니다.

그럼에도 장치(device)에 직접 접근해야 할 필요가 있다면, ``LightningModule`` 내부에서 (``__init__`` 과 ``setup`` 메소드를 제외하고) 아무데서나
``self.device`` 를 사용하면 됩니다.

.. testcode::

    class LitModel(LightningModule):
        def training_step(self, batch, batch_idx):
            z = torch.randn(4, 5, device=self.device)
            ...

Hint: ``LightningModule.__init__`` 메소드 내에서 :class:`~torch.Tensor` 를 초기화하면서 자동으로 장치(device)로 이동하려면
:meth:`~torch.nn.Module.register_buffer` 를 호출하여 매개변수로 등록하면 됩니다.

.. testcode::

    class LitModel(LightningModule):
        def __init__(self):
            super().__init__()
            self.register_buffer("running_mean", torch.zeros(num_features))

--------

*************************
8. 기존 데이터 사용하기
*************************
일반적인 PyTorch DataLoader는 Lightning에서 동작합니다. 더 모듈화되고 확장 가능한 데이터셋들은 :doc:`LightningDataModule <../data/datamodule>` 를
참고하세요.

----

************
더 알아두기
************

추가로, :meth:`~pytorch_lightning.trainer.trainer.Trainer.validate` 메소드를 사용하면 검증(validation) 루프만 실행할 수 있습니다.

.. code-block:: python

    model = LitModel()
    trainer.validate(model)

.. note:: ``model.eval()`` 와 ``torch.no_grad()`` 는 검증 시에 자동으로 호출됩니다.


테스트 루프(test loop)는 :meth:`~pytorch_lightning.trainer.trainer.Trainer.fit` 에서 사용되지 않으므로, 필요 시 명시적으로
:meth:`~pytorch_lightning.trainer.trainer.Trainer.test` 을 호출해야 합니다.

.. code-block:: python

    model = LitModel()
    trainer.test(model)

.. note:: ``model.eval()`` 와 ``torch.no_grad()`` 는 테스트 시에 자동으로 호출됩니다.

.. tip:: 체크포인트 기능이 켜진 경우, ``trainer.test()`` 는 자동으로 최적의 체크포인트(best checkpoint)를 불러옵니다.


예측 루프(prediction look)는 :meth:`~pytorch_lightning.trainer.trainer.Trainer.predict` 을 호출하기 전에는 사용되지 않습니다.

.. code-block:: python

    model = LitModel()
    trainer.predict(model)

.. note:: ``model.eval()`` 과 ``torch.no_grad()`` 는 예측 시에 자동으로 호출됩니다.

.. tip:: 체크포인트 기능이 켜진 경우, ``trainer.predict()`` 는 자동으로 최적의 체크포인트를 불러옵니다.
