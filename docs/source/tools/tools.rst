NonlinearRegression Tools and Utilities
=======================================

:py:mod:`NonlinearRegression.tools.analyze_run_data` and :py:mod:`NonlienarRegression.tools.bilinearHelper`
contain many functions which are useful for data preparation, calculation, and visualization.


:py:mod:`analyze_run_data` Module
---------------------------------

.. automodule:: NonlinearRegression.tools.analyze_run_data
   :members: get_opt_name, organize_run_data, read_minimum_loss, update_links, write_default_params, write_default_params, write_summary


.. autoclass:: NonlinearRegression.tools.analyze_run_data.ModelComparison
   :members:

   .. automethod:: __init__


:py:mod:`nlr_exceptions` Module
-------------------------------

.. automodule:: NonlinearRegression.tools.nlr_exceptions
   :members: loadTemplate, checkFileExists


.. autoclass:: NonlinearRegression.tools.nlr_exceptions.FileNotFound  
   :members: 

   .. automethod:: __init__


.. autoclass:: NonlinearRegression.tools.nlr_exceptions.TemplateNotFound 
   :members: 

   .. automethod:: __init__


:py:mod:`checkSource` Module
----------------------------

.. automodule:: NonlinearRegression.tensorflow.timedelay.checkSource
   :members: checkSource


.. autoclass:: NonlinearRegression.tensorflow.timedelay.checkSource.SourceError
   :members: 

   .. automethod:: __init__


:py:mod:`preprocessing` Module
------------------------------

.. automodule:: NonlinearRegression.tools.preprocessing
   :members: coarseGrainWrap

