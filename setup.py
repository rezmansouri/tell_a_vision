from setuptools import setup

setup(
    name='tell_a_vision',
    version='0.1.0',
    description='TV (tell_a_vision) is a tool that provides explanatory analysis for object detection tasks in computer vision.',
    url='https://github.com/rezmansouri/tell_a_vision',
    author='Reza Mansouri',
    author_email='std_reza_mansouri@khu.ac.ir',
    license='MIT',
    packages=['tell_a_vision'],
    install_requires=['numpy >= 1.21.0',
                      'gTTs >= 2.2.4'
                      ],
)
