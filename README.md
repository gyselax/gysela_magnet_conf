## gysmc package ##
Reading and generating magnetic configurations for GYSELA and Gysela-X++

We currently support:

- Circular geometry
- Culham geometry
- CHEASE
- GEQDSK file
- GVEC

We provide a `pyproject.toml` that you can use to install the library in a virtual env. To do so, clone the
repository and execute `pip install .`. There are optional dependencies if you want to use more geometries:
- `geqdsk_magnetconfig.py`: `pip install .[geqdsk]`,
- `gvec_magnetconfig.py`: `pip install .[gvec]`.
You can also install both with `pip install .[geqdsk,gvec]`. Beware that you need the dependencies of `gvec`
for the installation to succeed.

We also provide a helper `Makefile` that you can use with

```shell
make install
```

or to install both `geqdsk` and `gvec`

```shell
make install_full
```

In order to set up the environment on CEA machines, please source ```setup_env.sh```


Examples are given in the ```examples``` folder. Before running the notebooks in the examples please execute:


```shell
make install
make example
```

This will download the necessary input files for the magnetic configurations that are needed in the examples.
