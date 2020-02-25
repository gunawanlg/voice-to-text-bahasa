Welcome to Gurih documentation!
===============================

**Build Status & Coverage**

.. image:: https://travis-ci.com/Arc-rendezvous/voice-to-text-bahasa.svg?token=5tiyHPA8cpz3o58hhBPz&branch=development
   :alt: Build Status

.. image:: https://codecov.io/gh/Arc-rendezvous/voice-to-text-bahasa/branch/development/graph/badge.svg?token=iHS19McW5N
   :alt: Code Coverage

.. automodule:: gurih 

----

**gurih** is an automatic speech recognition (ASR) python module, crafted
specifically for recognizing and transcribing speech spoken in Bahasa. Given an
audio input, **gurih** will perform the transcription of the input.

For example, given an input of ``30.240s``-long audio recording of Kancil story
**gurih** will output the following transcription:

.. code-block:: text

    Pada suatu hari, terjadilah kelaparan di sebuah pulau yang penduduknya 
    kebanyakan di huni oleh para Harimau. Mereka sangat kelaparan, 
    karena semakin hari tidak ada hewan yang dapat mereka mangsa. 
    Akhirnya, Raja Harimau mengutus Panglima dan para Prajuritnya untuk pergi 
    ke pulau kecil di sebrang dan kembali dengan membawa banyak makanan.

How to Install
--------------
Currently ``gurih`` can only be installed by first cloning it from the repository
hosted on GitHub. The following is how you can install the module, from the command-line:

#. ``git clone https://github.com/Arc-rendezvous/voice-to-text-bahasa.git``
#. ``cd voice-to-text-bahasa``
#. ``pip install -e .``

Usage
-----

**gurih** can be used as a **Python package** inside third-party code directly
as the `.py` script or embedded to `.ipynb` notebooks.

Implemented Functionality
-------------------------
**gurih** provides the following functionality commonly used in the automatic
speech recognition task. Currently, gurih contains the following functionalities:

* ``gurih.models`` : contains the models used to perform the speech recognition task
* ``gurih.data`` : contains all the scripts needed for the data preprocessing of the audio files and transcription
* ``gurih.features`` : contains all the scripts needed for the feature extraction process

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Documentation

   api_cc

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Additional Information

   relevant_knowledge
   faq
   about
   changelog