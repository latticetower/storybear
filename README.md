# storybear

## Howto run

First, you need to install it as a package
```
git clone -b dev https://github.com/latticetower/storybear.git
pip install -e storybear
```
After that, `storytell` command should be available via shell. 

To load sample data file, run the command
```
storygen smth.csv 
```
as a result, there will appear file named `smth.csv` in the current directory.

Select your csv file (in the example below it is named `smth.csv` and located in the current directory) with tabular data and run
```
storytell smth.csv --tempdir imgdir
```
In the example above, `imgdir` is a path to a directory where the plot files, including temporary ones, will be located.

