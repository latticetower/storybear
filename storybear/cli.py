import numpy as np
import pandas as pd
import click

from .pipeline import StorybearPipeline


@click.command()
@click.argument('csv', type=click.File('w'))
def generate_data_cli(csv):
    # we can generate the data, but for simplicity let's start with something I already have and know, yet more complex
    df = pd.read_csv("hf://datasets/latticetower/nrps_modules_asdb4.0/nrps_modules_info_cleaned.csv")
    df.to_csv(csv, index=None)


@click.command()
@click.argument('csv', type=click.Path(readable=True))
@click.option('--tempdir', default="test", type=click.Path(writable=True))
def main_cli(csv, tempdir):
    """Processes tabular file in .csv format and saves plots to the provided directory"""
    print(csv, tempdir)
    pipeline = StorybearPipeline(csv, tempdir)
    pipeline.run()
    

if __name__ == '__main__':
    main_cli()

