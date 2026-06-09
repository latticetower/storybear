import numpy as np
import pandas as pd
import click

from storybear.local_inference import flux_i2i_func, it2t_summary_func

from storybear.pipeline import StorybearPipeline
from storybear.data import generate_data

@click.command()
@click.argument('csv', type=click.File())
def generate_data_cli(csv):
    # we can generate the data, but for simplicity let's start with something I already have and know, yet more complex
    generate_data(csv)


@click.command()
@click.argument('csv', type=click.Path(readable=True))
@click.option('--tempdir', default="test", type=click.Path(writable=True))
def main_cli(csv, tempdir):
    """Processes tabular file in .csv format and saves plots to the provided directory"""
    print(csv, tempdir)
    pipeline = StorybearPipeline(
        csv, 
        tempdir,
        captionist_it2t_func=it2t_summary_func,
    )
    pipeline.run()
    

if __name__ == '__main__':
    main_cli()

