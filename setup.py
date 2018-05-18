from setuptools import setup, find_packages

setup(
    name='obstacle-env',
    version='1.0.dev0',
    description='An environment for obstacle avoidance tasks',
    url='https://github.com/eleurent/obstacle-env',
    author='Edouard Leurent',
    author_email='eleurent@gmail.com',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Researchers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
    ],

    keywords='environment reinforcement learning',
    packages=find_packages(exclude=['docs', 'test*']),
    install_requires=['gym', 'numpy', 'scipy', 'pygame'],
    tests_require=['pytest'],
    extras_require={
        'deploy': ['pytest-runner']
    },
    entry_points={
        'console_scripts': [],
    },
)

