## gysmc package ##
Reading and generating magnetic configurations for GYSELA and Gysela-X++

We currently support:

- Circular geometry
- Culham geometry
- CHEASE
- GEQDSK file
- GVEC



To use, please clone the repository. You can either add the folder to ```PYTHONPATH``` or install it using

```shell
make install
```

In order to set up the environment on CEA machines, please source ```setup_env.sh```


Examples are given in the ```examples``` folder. Before running the notebooks in the examples please execute:


```shell
make install
make example
```

This will download the necessary input files for the magnetic configurations that are needed in the examples.
