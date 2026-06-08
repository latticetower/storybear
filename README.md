# storybear

## Howto run

First, you need to install it as a package
```
git clone -b dev https://github.com/latticetower/storybear.git
pip install -e storybear
```
After that, `storytell` command should be available via shell. 

Select your csv file (in the example below it is named `smth.csv`) with tabular data and run
```
storytell smth.csv --tempdir imgdir
```
In the example above, `imgdir` is a path to a directory where the plot files will be located.

