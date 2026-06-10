import gradio as gr
import pandas as pd
from pathlib import Path
from storybear.pipeline import StorybearPipeline
from storybear.data import generate_data

from storybear.local_inference import flux_i2i_func, it2t_summary_func


def create_app():
    with gr.Blocks(title="storybear") as demo:
        num_plots = gr.State(0)
        
        gr.Markdown("## STORYBEAR: from science to fairytale via agent-assisted storytelling")
        gr.Markdown("Click the button below to launch on demo dataset:")
        demo_button = gr.Button("Use demo csv file")

        @gr.render(inputs=[demo_button])
        def click_demo(count):
            print(count)
            tempdir = Path("temp")
            tempdir.mkdir(exist_ok=True)
            csv = tempdir / "smth.csv"
            generate_data(csv)

            pipeline = StorybearPipeline(
                csv, 
                tempdir, 
                captionist_it2t_func=it2t_summary_func,
                foodie_it2t_func=it2t_summary_func,
                editor_it2t_func=it2t_summary_func,
                # artist_i2i_func=flux_i2i_func,
            )
            run_result, docx_path = pipeline.run()
            print(run_result)
            blocks = [
                gr.Markdown(run_result.header),
                gr.Markdown(run_result.lead),
            ]
            for i, record in enumerate(run_result.plot_record_list):
                text = gr.Label(record.caption)
                im = gr.Image(record.plot_path)
                blocks.append(gr.Column(f"Row_{i}", [text, im]))
            # return blocks
            # new_blocks = 
        # demo_button.click(click_demo, [], [])
        container = gr.Row("Parent container")

    return demo