from setuptools import setup

setup(
    name='gammaProf',
    version='0.2',
    description='Dark matter halo shear profile fitting code assuming an NFW form',
    url='https://github.com/jhollowed/gammaProf',
    author='Joe Hollowed',
    author_email='jphollowed@anl.gov',
    license='BSD',
    packages=['gammaProf'],
    python_requires='>=3.5',
    test_suite='nose.collector',
    tests_require=['nose', 'halotools', 'cluster-lensing', 'lenstronomy']
    zip_safe=False
)
