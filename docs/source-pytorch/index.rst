⚡ PyTorch Lightning에 오신 것을 환영합니다!
========================================

.. twocolumns::
   :left:
      .. image:: _static/fetched-s3-assets/mov.gif
         :alt: Animation showing how to convert standard training code to Lightning
   :right:
      PyTorch Lightning(파이토치 라이트닝))은 대규모에서 성능을 포기하지 않으면서 최대한의 유연성을 필요로 하는 전문적인 AI 연구자들과 머신러닝 엔지니어들을 위한 딥러닝 프레임워크입니다.
      Lightning(라이트닝)은 프로젝트가 생각으로부터 문서 / 제품화에 이르는 동안 함께 발전합니다.

.. raw:: html

   <div class="row" style='font-size: 14px'>
      <div class='col-md-6'>
      </div>
      <div class='col-md-6'>

.. raw:: html

      </div>
   </div>


.. raw:: html

   <hr class="docutils" style="margin: 50px 0 50px 0">


Lightning 설치하기
----------------------


.. raw:: html

   <div class="row" style='font-size: 16px'>
      <div class='col-md-6'>

Pip 사용자라면,

.. code-block:: bash

    pip install lightning

.. raw:: html

      </div>
      <div class='col-md-6'>

Conda 사용자라면,

.. code-block:: bash

    conda install lightning -c conda-forge

.. raw:: html

      </div>
   </div>

또는 `advanced install guide <starter/installation.html>`_ 참조하세요.

지원하는 PyTorch 버전은 :ref:`compatibility matrix <versioning:Compatibility matrix>` 에서 확인할 수 있습니다.

.. raw:: html

   <hr class="docutils" style="margin: 50px 0 50px 0">

처음이신가요?
-----------

.. raw:: html

    <div class="tutorials-callout-container">
        <div class="row">

.. Add callout items below this line

.. customcalloutitem::
   :description: Learn the 7 key steps of a typical Lightning workflow.
   :header: Lightning in 15 minutes
   :button_link:  starter/introduction.html

.. customcalloutitem::
   :description: Learn how to benchmark PyTorch Lightning.
   :header: Benchmarking
   :button_link: benchmarking/benchmarks.html

.. raw:: html

        </div>
    </div>

.. End of callout item section

.. raw:: html

   <hr class="docutils" style="margin: 50px 0 50px 0">

이미 Lightning 사용자라면?
-----------------------

.. raw:: html

    <div class="tutorials-callout-container">
        <div class="row">

.. Add callout items below this line

.. customcalloutitem::
   :description: Learn Lightning in small bites at 4 levels of expertise: Introductory, intermediate, advanced and expert.
   :header: Level Up!
   :button_link:  expertise_levels.html

.. customcalloutitem::
   :description: Detailed description of API each package. Assumes you already have basic Lightning knowledge.
   :header: API Reference
   :button_link: api_references.html

.. customcalloutitem::
   :description: From NLP, Computer vision to RL and meta learning - see how to use Lightning in ALL research areas.
   :header: Hands-on Examples
   :button_link: tutorials.html

.. customcalloutitem::
   :description: Learn how to do everything from hyper-parameters sweeps to cloud training to Pruning and Quantization with Lightning.
   :header: Common Workflows
   :button_link: common_usecases.html

.. customcalloutitem::
   :description: Convert your current code to Lightning
   :header: Convert code to PyTorch Lightning
   :button_link: starter/converting.html


.. raw:: html

        </div>
    </div>

.. End of callout item section

.. raw:: html

   <div style="display:none">

.. toctree::
   :maxdepth: 1
   :name: start
   :caption: Home

   starter/introduction
   Install <starter/installation>
   upgrade/migration_guide


.. toctree::
   :maxdepth: 2
   :name: levels
   :caption: Level Up

   levels/core_skills
   levels/intermediate
   levels/advanced
   levels/expert

.. toctree::
   :maxdepth: 1
   :name: pl_docs
   :caption: Core API

   common/lightning_module
   common/trainer

.. toctree::
   :maxdepth: 1
   :name: api
   :caption: Optional API

   api_references

.. toctree::
   :maxdepth: 1
   :name: More
   :caption: More

   Community <community/index>
   Examples <tutorials>
   Glossary <glossary/index>
   How to <common/index>


.. raw:: html

   </div>

.. PyTorch-Lightning documentation master file, created by
   sphinx-quickstart on Fri Nov 15 07:48:22 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
