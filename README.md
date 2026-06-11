# STORYBEAR: from science to fairytale via agent-assisted storytelling.

![project logo](https://github.com/latticetower/storybear/blob/dev/assets/storybear-logo.png)

## Features & Limitations
* The project is an autoEDA framework with the rich user experience.
* Any non-standard plots appearing in the report is a courtesy of authors mind.
* In the end it should process some particular bioinformatics data formats (protein sequences, SMILES strings), because I myself am too lazy to do explicit exploratorials by hand every time.
* Some of the captions might be hard-coded. 
* The report header is designed to add some amount of exaggeration.
* The plots are generated 'as is', after that they are processed with LLMs. LLMs might hallucinate. I also might hallucinate during plots coding. This means that any results might not correctly represent the input data. **Use at your own risk**.

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

### Command line

Select your csv file (in the example below it is named `smth.csv` and located in the current directory) with tabular data and run
```
storytell smth.csv --tempdir imgdir
```
In the example above, `imgdir` is a path to a directory where the plot files, including temporary ones, will be located.

This produces the report in docx format.

### Interactive (gradio demo)
There is also an option to run demo with gradio. This can be done by running
```
gradio app.py
```

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
- [ ] Remove columns with "_id", "Id" and "identifier" from the consideration
- [ ] add default text gen (both with and without LLMs, no data)
- [ ] remove very similar plots based on their descriptions
- [ ] don't build plots for the highly correlated columns
- [x] For string columns: draw length distributions
- [ ] Proteins: compute embeddings with esm2 8m + draw scatterplots
- [ ] SMILES: chemberta | mol descriptors

## References
1. https://github.com/py-pdf/fpdf2 candidate package for report creation
2. https://arxiv.org/abs/2605.14163 possible candidate method for pipeline inprovement
3. https://arxiv.org/abs/2508.16757 paper on reranking strategies. I use basic and slow approach at the moment - pair reranking of plot descriptions, scoring based on this reranking, selection of top N plots (N=5).


## Our Team

- [@latticetower](https://github.com/latticetower) Tatiana Malygina
- [@nofate](https://github.com/nofate) Michael Gamov