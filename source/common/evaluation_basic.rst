:orphan:

#################################
모델 검증하고 테스트하기 (기본)
#################################
**예상 독자**: 과적합(overfit)을 방지하기 위해 검증 단계를 추가하려는 사용자

----

*********************
테스트 단계 추가하기
*********************
모델이 학습하지 않은(unseen) 데이터셋에서도 일반화(generalize)되는지(예: 논문을 게재하거나 프로덕션 환경에서) 확인하기 위해, 일반적으로 데이터셋을 *학습용* 과 *테스트용* 의 두 부분으로 분할합니다.

테스트셋은 학습에 사용하지 **않으며**, 학습된 모델이 실세계에서 얼마나 잘 동작하는지를 **평가하는데만** 사용합니다.

----

학습용과 테스트용 찾아보기
=================================
데이터셋은 두 부분으로 나뉘어 있습니다. 데이터셋 문서를 참고하여 *학습용* 과 *테스트용* 분할(split)을 찾아보세요.

.. code-block:: python

   import torch.utils.data as data
   from torchvision import datasets

   # 데이터셋
   train_set = datasets.MNIST(root="MNIST", download=True, train=True)
   test_set = datasets.MNIST(root="MNIST", download=True, train=False)

----

테스트 단계(loop) 정의하기
============================
LightningModule의 **test_step** 메소드를 구현하여 테스트 단계(loop)를 추가합니다.

.. code:: python

    class LitAutoEncoder(pl.LightningModule):
        def training_step(self, batch, batch_idx):
            ...

        def test_step(self, batch, batch_idx):
            # 여기가 테스트하는 부분입니다.
            x, y = batch
            x = x.view(x.size(0), -1)
            z = self.encoder(x)
            x_hat = self.decoder(z)
            test_loss = F.mse_loss(x_hat, x)
            self.log("test_loss", test_loss)

----

테스트 단계를 포함하여 학습하기
==================================
모델이 학습을 끝내고 난 뒤에, **.test** 를 호출합니다.

.. code-block:: python

   from torch.utils.data import DataLoader

   # Trainer 초기화하기
   trainer = Trainer()

   # 모델 테스트하기
   trainer.test(model, dataloaders=DataLoader(test_set))

----

*********************
검증 단계 추가하기
*********************
학습 중, 모델이 학습을 완료하는 시점을 판단하기 위해 학습 데이터의 작은 부분을 사용하는 것이 일반적입니다.

----

학습용 데이터 분할하기
=======================
일반적으로 학습 데이터셋의 20% 가량을 **검증용셋** 으로 사용합니다. 이 숫자는 데이터셋에 따라 달라집니다.

.. code-block:: python

   # 학습용 데이터의 20%를 검증용으로 사용합니다.
   train_set_size = int(len(train_set) * 0.8)
   valid_set_size = len(train_set) - train_set_size

   # 학습용 세트를 2개로 나눕니다.
   seed = torch.Generator().manual_seed(42)
   train_set, valid_set = data.random_split(train_set, [train_set_size, valid_set_size], generator=seed)

----

검증 단계 정의하기
==========================
LightningModule의 **validation_step** 메소드를 구현하여 검증 단계(loop)를 추가합니다.

.. code:: python

    class LitAutoEncoder(pl.LightningModule):
        def training_step(self, batch, batch_idx):
            ...

        def validation_step(self, batch, batch_idx):
            # 여기가 검증하는 부분입니다
            x, y = batch
            x = x.view(x.size(0), -1)
            z = self.encoder(x)
            x_hat = self.decoder(z)
            test_loss = F.mse_loss(x_hat, x)
            self.log("val_loss", test_loss)

----

검증 단계를 포함하여 학습하기
==============================
검증 단계를 실행하기 위해, **.fit** 호출 시에 검증용 데이터를 함께 전달합니다.

.. code-block:: python

   from torch.utils.data import DataLoader

   train_set = DataLoader(train_set)
   val_set = DataLoader(val_set)

   # 학습 데이터와 검증 데이터 모두를 사용하여 학습합니다.
   trainer = Trainer()
   trainer.fit(model, train_set, val_set)
