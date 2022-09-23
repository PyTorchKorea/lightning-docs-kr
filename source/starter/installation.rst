:orphan:

.. _installation:

############
설치하기
############

--------------

*********************
pip를 사용하여 설치
*********************

`파이토치(PyTorch) 설치하기 페이지 <https://pytorch.kr/get-started/locally/#start-locally>` _ 에서 PyTorch를 설치한 뒤,
아래 명령어로 `pip <https://pypi.org/project/pytorch-lightning/>`_ 를 사용하여 설치할 수 있습니다:

.. code-block:: bash

    pip install pytorch-lightning

--------------

***********************
Conda를 사용하여 설치
***********************

만약 conda를 아직 설치하지 않았다면, `Conda 설치 가이드 <https://docs.conda.io/projects/conda/en/latest/user-guide/install>`_ 를 참고하세요.
Lightning은 아래 명령어로 `conda <https://anaconda.org/conda-forge/pytorch-lightning>`_  를 사용하여 설치할 수 있습니다:

.. code-block:: bash

    conda install pytorch-lightning -c conda-forge

`Conda 가상환경(Environments) <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_ 을 사용할 수도 있습니다:

.. code-block:: bash

    conda activate my_env
    conda install pytorch-lightning -c conda-forge

--------------

************************
소스 코드로 설치
************************

소스 코드로 최신 버전(nightly)을 설치합니다. 아직 배포되지 않은 버그 수정(bug fix)과 새롭게 출시할 기능들이
포함되어 있습니다. 미검증·불안정 최신 기능(bleeding edge)이므로, 신중하게 사용하세요.

.. code-block:: bash

    pip install https://github.com/PyTorchLightning/pytorch-lightning/archive/master.zip

향후 공개될 개선 버전(patch release)를 소스 코드로부터 설치합니다. 개선 버전은 가장 최근의 주요 버전(major release)에 대한 버그 수정만
포함되어 있습니다.

.. code-block:: bash

    pip install https://github.com/PyTorchLightning/pytorch-lightning/archive/refs/heads/release/1.5.x.zip

--------------

************************************
Lightning 커버리지(Coverage)
************************************

파이토치 라이트닝(PyTorch Lightning)은 다양한 Python과 PyTorch 버전에서 유지 보수 및 테스트되고 있습니다.

더 자세한 정보는 `CI Coverage <https://github.com/PyTorchLightning/pytorch-lightning#continuous-integration>`_ 를 참고하세요.

다양한 GPU와 TPU, CPU, IPU에서 엄격하게 테스트되었습니다. GPU 테스트는 2개의 NVIDIA P100에서 실행됩니다. TPU 테스트는 Google GKE TPUv2/3에서
실행됩니다. TPU py3.7은 Colab 및 Kaggle 환경을 지원함을 뜻합니다. IPU 테스트는 MK1 IPU 장비에서 실행됩니다.
