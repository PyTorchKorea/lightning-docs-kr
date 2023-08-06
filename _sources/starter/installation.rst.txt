:orphan:

.. _installation:

############
설치하기
############

*********************
pip를 사용하여 설치
*********************

라이트닝(lightning)을 가상환경이나 conda 환경에서 pip로 설치할 수 있습니다

.. code-block:: bash

    python -m pip install lightning

--------------

***********************
Conda를 사용하여 설치
***********************

만약 conda를 아직 설치하지 않았다면, `Conda 설치 가이드 <https://docs.conda.io/projects/conda/en/latest/user-guide/install>`_ 를 참고하세요.
Lightning은 아래 명령어로 `conda <https://anaconda.org/conda-forge/pytorch-lightning>`_  를 사용하여 설치할 수 있습니다:

.. code-block:: bash

    conda install lightning -c conda-forge

`Conda 가상환경(Environments) <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_ 을 사용할 수도 있습니다:

.. code-block:: bash

    conda activate my_env
    conda install lightning -c conda-forge

----

GRPC 패키지를 가져오는 데 어려움이 있는 경우 `이 글 <https://stackoverflow.com/questions/66640705/how-can-i-install-grpcio-on-an-apple-m1-silicon-laptop>`_ 을 따라해보세요.



----

**********************
소스 코드에서 설치
**********************

소스 코드로 최신 버전(nightly)을 설치합니다. 아직 배포되지 않은 버그 수정(bug fix)과 새롭게 출시할 기능들이
포함되어 있습니다. 미검증·불안정 최신 기능(bleeding edge)이므로, 신중하게 사용하세요.

.. code-block:: bash

    pip install https://github.com/Lightning-AI/lightning/archive/refs/heads/master.zip -U

향후 공개될 개선 버전(patch release)를 소스 코드로부터 설치합니다. 개선 버전은 가장 최근의 주요 버전(major release)에 대한 버그 수정만
포함되어 있습니다.

.. code-block:: bash

    pip install https://github.com/Lightning-AI/lightning/archive/refs/heads/release/stable.zip -U

----

*******************************
모델 개발에 최적화된 버전 설치
*******************************
이미 Lightning으로 개발한 모델을 배포하기 위해 최소한의 의존성만을 필요로 하는 경우, 최적화된 `lightning[pytorch]` 패키지를 설치하세요:

.. code-block:: bash

    pip install 'lightning[pytorch]'

^^^^^^^^^^^^^^^^^^^^^^^^^^^
PyTorch 버전 지정하기
^^^^^^^^^^^^^^^^^^^^^^^^^^^
특정한 PyTorch 버전을 사용하려면 `PyTorch 설치 페이지 <https://pytorch.kr/get-started/locally/#start-locally>`_ 를 참고하세요.

----


*****************************************************
ML 워크플로우에 최적화된 버전 설치하기 (lightning Apps)
*****************************************************
이미 Lightning으로 개발한 워크플로우를 배포하기 위해 최소한의 의존성만을 필요로 하는 경우, 최적화된 `lightning[apps]` 패키지를 설치하세요:

.. code-block:: bash

    pip install lightning-app
