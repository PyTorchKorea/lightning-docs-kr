:orphan:

#####################
모델 학습하기 (기본)
#####################
**예상 독자**: 자체 학습 루프(loop)를 작성하지 않고 모델을 학습할 필요가 있는 사용자.

----

***********
불러오기
***********
파일의 가장 윗 부분에 관련 호출(import)을 추가합니다

.. code:: python

    import os
    import torch
    from torch import nn
    import torch.nn.functional as F
    from torchvision import transforms
    from torchvision.datasets import MNIST
    from torch.utils.data import DataLoader, random_split
    import pytorch_lightning as pl

----

***************************************
파이토치(PyTorch) nn.Modules 정의하기
***************************************

.. code:: python

    class Encoder(nn.Module):
        def __init__(self):
            self.l1 = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))

        def forward(self, x):
            return self.l1(x)


    class Decoder(nn.Module):
        def __init__(self):
            self.l1 = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

        def forward(self, x):
            return self.l1(x)

----

**************************
LightningModule 정의하기
**************************
LightningModule은 nn.Module이 어떻게 동작할지 정의할 수 있는 완벽한 **비결(recipe)** 입니다.

- **training_step** 은 *nn.Modules* 과 어떻게 상호 작용할 것인지 정의합니다.
- **configure_optimizers** 에서는 모델에서 사용할 옵티마이저(들)을 정의합니다.

.. code:: python

    class LitAutoEncoder(pl.LightningModule):
        def __init__(self, encoder, decoder):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder

        def training_step(self, batch, batch_idx):
            # training_step은 학습 루프를 정의합니다.
            x, y = batch
            x = x.view(x.size(0), -1)
            z = self.encoder(x)
            x_hat = self.decoder(z)
            loss = F.mse_loss(x_hat, x)
            return loss

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
            return optimizer

----

***************************
학습 데이터셋 정의하기
***************************
학습 데이터셋을 포함하고 있는 PyTorch :class:`~torch.utils.data.DataLoader` 를 정의합니다.

.. code-block:: python

    dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(dataset)

----

***************
모델 학습하기
***************
모델 학습을 위해서는 Lightning :doc:`Trainer <../common/trainer>` 를 사용합니다. 이는 규모 확장 시에 필요한 모든 복잡성을 추상화하고 각종 엔지니어링을 담당합니다.

.. code-block:: python

    # model
    autoencoder = LitAutoEncoder(Encoder(), Decoder())

    # train model
    trainer = pl.Trainer()
    trainer.fit(model=autoencoder, train_dataloaders=train_loader)

----

***************************
학습 루프 제거하기
***************************
사용자를 대신하여 Lightning Trainer가 내부적으로 아래와 같은 학습 루프를 실행합니다.

.. code:: python

    autoencoder = LitAutoEncoder(encoder, decoder)
    optimizer = autoencoder.configure_optimizers()

    for batch, batch_idx in enumerate(train_loader):
        loss = autoencoder(batch, batch_idx)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

Lightning의 진가는 검증/테스트 분할(validation/test split), 스케줄러, 분산 학습 및 최신 SOTA 테크닉들을 추가하면서 학습 과정이 복잡해질 때 나타납니다.

Lightning을 사용하면 매번 새로운 학습 루프를 작성할 필요없이 이러한 테크닉들을 모두 사용할 수 있습니다.
