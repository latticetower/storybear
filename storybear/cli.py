import numpy as np
import pandas as pd
import click

from storybear.pipeline import StorybearPipeline
from storybear.data import generate_data


def _load_backends(remote: bool):
    """Pick the local (in-process) or remote (Modal HTTP) inference backend."""
    if remote:
        from storybear.remote_inference import flux_i2i_func, it2t_summary_func
    else:
        from storybear.local_inference import flux_i2i_func, it2t_summary_func
    return it2t_summary_func, flux_i2i_func

@click.command()
@click.argument('csv', type=click.Path())
def generate_data_cli(csv):
    # we can generate the data, but for simplicity let's start with something I already have and know, yet more complex
    generate_data(csv)


@click.command()
@click.argument('csv', type=click.Path(readable=True))
@click.option('--tempdir', default="test", type=click.Path(writable=True))
@click.option('--captionist', is_flag=True)
@click.option('--foodie', is_flag=True)
@click.option('--editor', is_flag=True)
@click.option('--artist', is_flag=True)
@click.option('--remote', is_flag=True, help="Use the Modal-hosted services instead of local models.")
def main_cli(csv, tempdir, captionist, foodie, editor, artist, remote):
    """Processes tabular file in .csv format and saves plots to the provided directory"""
    it2t_summary_func, flux_i2i_func = _load_backends(remote)
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

