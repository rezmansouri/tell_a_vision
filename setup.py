from setuptools import setup

setup(
    name='tell_a_vision',
    version='0.1.7',
    description='TV (tell_a_vision) is an inference engine for object detection that provides explanatory analysis for such tasks in computer vision.',
    url='https://github.com/rezmansouri/tell_a_vision',
    author='Reza Mansouri',
    author_email='std_reza_mansouri@khu.ac.ir',
    license='MIT',
    install_requires=['numpy >= 1.21.0',
                      'gTTs >= 2.2.4'
                      ],
    long_description="TV (tell_a_vision) is an inference that takes object detection algorithms' (such as YOLO) output as input and provides explanatory data such as the location, number, and size (distance) of the objects found with respect to their classification in a descriptive format along with audio output. You can try it using this colab notebook demo: https://colab.research.google.com/drive/1o6MgntmIb1qLLpGXxsDw0W6SMH2kheCZ?usp=sharing."
)
