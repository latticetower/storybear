# STORYBEAR.

![project logo](https://github.com/latticetower/storybear/blob/dev/assets/storybear-logo.png)

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

## Pipeline scheme

![project schematics](https://github.com/latticetower/storybear/blob/dev/assets/storybear-scheme.jpg)

The names of the classes representing each particular part of the pipeline are selected based on their function. 

0. The pipeline accepts .csv file with columns of different types as an input
1. The columns are processed by the class which I named `DataGal`. As a result of processing, I have a lot of plots produced from dataset columns (let's call each of them P) in the temporary folder. 
2. For each P and corresponding statistical info (represented as a dictionary of key and value pairs, differs for different plotter classes) I use class named `Captionist` - to generate text caption C with LLM. 
3. Each pair of (plot P, text caption C) is processed by `Foodie` class - which is also powered by LLM and returns ranking R. It could have been named Critique, but I've decided to keep it simple in case if I'll decide to add other filtering steps in the pipeline and call them Critique.
4. Next class `Secretary` in the pipeline accepts all the triplets (plot P, text caption C, ranking R) and pick top N (N is a fixed parameter) samples with the highest rankings R. Technically, it is little filter in the beginning of `Editor` execution - I didn't included it in the scheme image.
5. The results of the previous step are processed by `Editor` class. `Editor` uses all the data from the previous step, to summarize them to report lead L (which is an analogue to lead in a news article) and a catchy header H. Header is a more or less exaggerrated depending on some  parameter.
6. The top N triplets (P, C, R) from step 4 with the report lead L, header H from the step 5 are collected and given to the class `Junior`. Basically, in this step the LLM does routine work: look at the editor's idea of the paper and reorder all the selected triplets (P, C, R) to make the overall story look more convincing.
7. The data produced by step 7 are given to the `Artist` class, which modifies the plots in place using its own artistic vision with LLM and keeps everything else as it was.
8. The last class is `Typography` - it collects the data from the previous step and converts the report (for simplicity, made with `python-docx` package).

## TODO & ideas
- [ ] add other methods of reporting, i.e., probably replace docx with pdf generation or even make video (slides with generated images+tts)
- [ ] 

## References
TBA

## Our Team

- @latticetower (Tanya Malygina)
- @nofate (Michael Gamov)