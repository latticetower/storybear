import numpy as np
import pandas as pd
import click

from storybear.local_inference import flux_i2i_func, it2t_summary_func

from storybear.pipeline import StorybearPipeline
from storybear.data import generate_data, download_example_data

@click.command()
@click.argument('csv', type=click.Path())
@click.argument('--i', default=0, type=int)
def generate_data_cli(csv, i):
    # we can generate the data, but for simplicity let's start with something I already have and know, yet more complex
    download_example_data(csv, i)


@click.command()
@click.argument('csv', type=click.Path(readable=True))
@click.option('--tempdir', default="test", type=click.Path(writable=True))
@click.option('--captionist', is_flag=True)
@click.option('--foodie', is_flag=True)
@click.option('--editor', is_flag=True)
@click.option('--artist', is_flag=True)
def main_cli(csv, tempdir, captionist, foodie, editor, artist):
    """Processes tabular file in .csv format and saves plots to the provided directory"""
    capt_func = it2t_summary_func if captionist else None
    foodie_func = it2t_summary_func if foodie else None
    editor_func = it2t_summary_func if editor else None
    artist_func = flux_i2i_func if artist else None
    pipeline = StorybearPipeline(
        csv, 
        tempdir,
        captionist_it2t_func=capt_func,
        foodie_it2t_func=foodie_func,
        editor_it2t_func=editor_func,
        artist_i2i_func=artist_func,
    )
    pipeline.run()
    

if __name__ == '__main__':
    main_cli()

