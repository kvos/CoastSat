from distutils.core import setup

setup(
    name='CoastSatDHI',
    version='0.1',
    packages=['coastsat'],
    url='',
    license='',
    author='anbr',
    author_email='',
    description='DHI Fork of CoastSAT',
    install_requires = ['numpy=1.16.3',
                        'matplotlib=3.0.3',
                        'earthengine-api=0.1.173',
                        'gdal=2.3.3',
                        'pandas=0.24.2',
                        'geopandas=0.4.1',
                        'pytz=2019.1',
                        'scikit-image=0.15.0',
                        'scikit-learn=0.20.3',
                        'shapely=1.6.4',
                        'scipy=1.2.1',
                        'spyder=3.3.4',
                        'notebook=5.7.8'])
