from setuptools import setup


setup(
    name='cldfbench_haspelmathindefpro',
    py_modules=['cldfbench_haspelmathindefpro'],
    include_package_data=True,
    zip_safe=False,
    entry_points={
        'cldfbench.dataset': [
            'haspelmathindefpro=cldfbench_haspelmathindefpro:Dataset',
        ]
    },
    install_requires=[
        'cldfbench[glottolog,excel]',
        'simplepybtex',
    ],
    extras_require={
        'test': [
            'pytest-cldf',
        ],
    },
)
