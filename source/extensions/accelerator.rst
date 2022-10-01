.. _accelerator:

###########
Accelerator
###########

Accelerator(가속기)는 Lightning Trainer를 임의 하드웨어 (CPUs, GPUs, TPUs, IPUs 등...)에 연결합니다.
현재 다음과 같은 Accelerator가 있습니다:

- CPU
- :doc:`GPU <../accelerators/gpu>`
- :doc:`TPU <../accelerators/tpu>`
- :doc:`IPU <../accelerators/ipu>`
- :doc:`HPU <../accelerators/hpu>`

Accelerator는 여러 장치 (분산 통신) 간의 통신을 관리하는 Strategy의 일부입니다.
Trainer, loops 또는 Lightning의 다른 구성 요소가 하드웨어와 통신해야할 때마다 Strategy를 호출하고 Strategy는 Accelerator를 호출합니다.

.. image:: https://pl-public-data.s3.amazonaws.com/docs/static/images/strategies/overview.jpeg
    :alt: Illustration of the Strategy as a composition of the Accelerator and several plugins

We expose Accelerators and Strategies mainly for expert users who want to extend Lightning to work with new
hardware and distributed training or clusters.
주로 새로운 하드웨어 및 분산 학습 또는 클러스터와 함께 작동하도록 Lightning을 확장하려는 전문가 사용자를 위해 Accelerators 및 Strategies를 제공합니다.


----------

사용자 지정 Accelerator 만들기
---------------------------

다음은 새로운 Accelerator를 만드는 방법입니다.
가상의 XPU 가속기를 통합하고 ``xpulib`` 라이브러리를 통해 하드웨어에 액세스할 수 있다고 가정해 보겠습니다.
.. code-block:: python

    import xpulib


    class XPUAccelerator(Accelerator):
        """대규모 기계 학습에 최적화된 XPU에 대한 실험 지원."""

        @staticmethod
        def parse_devices(devices: Any) -> Any:
            # 장치가 `devices` 인수를 통해 Trainer에게 전달 될 수있는 방법 작성
            return devices

        @staticmethod
        def get_parallel_devices(devices: Any) -> Any:
            # 장치 인덱스를 실제 장치 객체로 변환
            return [torch.device("xpu", idx) for idx in devices]

        @staticmethod
        def auto_device_count() -> int:
            # `Trainer(devices="auto")`일 때 auto-device 선택값 반환
            return xpulib.available_devices()

        @staticmethod
        def is_available() -> bool:
            return xpulib.is_available()

        def get_device_stats(self, device: Union[str, torch.device]) -> Dict[str, Any]:
            # Return optional device statistics for loggers
            # loggers에 대한 선택적인 장치 통계 반환
            return {}


마지막으로 Trainer에 XPUAccelerator를 추가합니다:

.. code-block:: python

    from pytorch_lightning import Trainer

    accelerator = XPUAccelerator()
    trainer = Trainer(accelerator=accelerator, devices=2)


:doc:Strategies 및 Strategies가 Accelerator와 어떻게 상호 작용하는지에 대해 `더 알아보기. <../extensions/strategy>`


----------

Accelerators 등록
------------------------

코드 변경 없이 CLI에서 사용자 지정 가속기로 전환하려면 다음과 같이 새 가속기를 약식 이름으로 등록하는 :meth:`~pytorch_lightning.accelerators.accelerator.Accelerator.register_accelerators` 클래스 메서드를 구현하면 됩니다:

.. code-block:: python

    class XPUAccelerator(Accelerator):
        ...

        @classmethod
        def register_accelerators(cls, accelerator_registry):
            accelerator_registry.register(
                "xpu",
                cls,
                description=f"XPU Accelerator - optimized for large-scale machine learning.",
            )

이제 다음과 같이 사용이 가능합니다:

.. code-block:: python

    trainer = Trainer(accelerator="xpu")

또는 Lightning CLI를 사용하는 경우 예를 들어 다음과 같습니다:

.. code-block:: bash

    python train.py fit --trainer.accelerator=xpu --trainer.devices=2


----------

Accelerator API
---------------

.. currentmodule:: pytorch_lightning.accelerators

.. autosummary::
    :nosignatures:
    :template: classtemplate.rst

    Accelerator
    CPUAccelerator
    GPUAccelerator
    HPUAccelerator
    IPUAccelerator
    TPUAccelerator
