import setuptools


setuptools.setup(
    name='ostrichrl',
    description='OstrichRL',
    url='https://github.com/vittorione94/ostrichrl',
    version='0.1.0',
    author='Vittorio La Barbera, Fabio Pardo',
    author_email='vlabarbera@rvc.ac.uk, f.pardo@imperial.ac.uk',
    install_requires=['dm_control', 'lxml', 'numpy'],
    packages=setuptools.find_packages(exclude=("data")),
    license='MIT',
    python_requires='>=3.6',
    keywords=['ostrichrl', 'musculoskeletal', 'biomechanics',
              'reinforcement learning'])
