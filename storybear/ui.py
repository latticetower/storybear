import gradio as gr
import pandas as pd
from pathlib import Path
from storybear.pipeline import StorybearPipeline
from storybear.cli import generate_data_cli


def greet(name, intensity):
    return "Hello, " + name + "!" * int(intensity)


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
            # generate_data_cli(csv)
            pipeline = StorybearPipeline(csv, tempdir)
            run_result, docx_path = pipeline.run()
            print(run_result)
            blocks = [
                gr.Markdown(run_result.header),
                gr.Markdown(run_result.lead),
            ]
            for i, record in enumerate(run_result.plot_record_list):
                im = gr.Image(record.plot_path)
                text = gr.Label(record.caption)
                blocks.append(gr.Row(f"Row_{i}", [im, text]))
            # return blocks
            # new_blocks = 
        # demo_button.click(click_demo, [], [])
        container = gr.Row("Parent container")
    return demo