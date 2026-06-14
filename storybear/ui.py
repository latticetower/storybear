import gradio as gr
import pandas as pd
from pathlib import Path
from time import sleep
import numpy as np

from storybear.data import download_example_data
from storybear.data_structures import ReportRecord, PlotRecord
from storybear.pipeline import StorybearPipeline


DATA_EXAMPLES = [
    ["hf://datasets/phihung/titanic/train.csv"],
    ["hf://datasets/latticetower/nrps_modules_asdb4.0/nrps_modules_info_cleaned.csv"]
]


instances = {}

def initialize_instance(request: gr.Request):
    tempdir = Path("temp")
    tempdir.mkdir(exist_ok=True)
    csv = tempdir / "smth.csv"
    download_example_data(csv)
    df = pd.read_csv(csv)
    instances[request.session_hash] = {'data_frame': df, 'report': None}
    return "Session initialized!"


def cleanup_instance(request: gr.Request):
    if request.session_hash in instances:
        del instances[request.session_hash]


# def increment_counter(request: gr.Request):
#     if request.session_hash in instances:
#         instance = instances[request.session_hash]
#         return instance+1
#     return "Error: Session not initialized"

def call_clear_checkboxes(*input):
    print("call clear", len(input))
    return [False]*len(input)


class StageProcessor:
    def __init__(self, named_stages_list):
        self.named_stages_list = named_stages_list
        #self.stage_name = stage_name
        #self.stage_func = stage_func
    def __len__(self):
        return len(self.named_stages_list)

    def __getitem__(self, i):
        stage_name, stage_func = self.named_stages_list[i]

        def process_step(input_block, request: gr.Request, progress=gr.Progress()):
            if not input_block:
                return False
            # stage_name, stage_func = stage_info
            # increment_btn.click(increment_counter, inputs=None, outputs=counter)
            # print("before:", self.stage_name, self.current_value)
            # self.current_value = self.stage_func(self.current_value)
            
            if request.session_hash in instances:
                # Process data
                #instance = instances[request.session_hash]
                #instances[request.session_hash] = stage_func(instance)
                data = instances[request.session_hash]
                report = data['report']
                if report is None:
                    new_report = stage_func(data['data_frame'])
                else:
                    new_report = stage_func(report)
                instances[request.session_hash]['report'] = new_report
            # print("after:", self.stage_name, self.current_value)
            print(input_block)
            text = ""
            for i in progress.tqdm(np.arange(5)):
                text += f"{i}"
                sleep(0.01)
            return True
    
        def restart_pipeline(request: gr.Request, progress=gr.Progress()):
            print("restart")
            # clear_checkboxes_button.click()
            return process_step(True, request, progress=progress)
        
        if i == 0:
            return restart_pipeline
        return process_step

    def finish_job(self, input_block, request: gr.Request):
        # update state of associated components based on session
        print(input_block)
        if request.session_hash in instances:
            report = instances[request.session_hash]['report']
            if report is None:
                print("Report is None")
                return "Report is None"
            print("Finished! Session hash is ok, didn't make it to the report yet")
            # TODO: actual processing of the results should be here
            return "Finished! Session hash is ok, didn't make it to the report yet"

        return f"Finished! Session hash is not available, no data"



def create_app(use_llm=True, use_vlm=False, remote=False):

    # Choose the inference backend: remote (Modal HTTP services) or local models.
    inference = "storybear.remote_inference" if remote else "storybear.local_inference"

    if use_llm:
        import importlib
        inf = importlib.import_module(inference)
        it2t_summary_func = inf.it2t_summary_func
        # Foodie uses the optimized pairwise compare call when the backend has one.
        it2t_compare_func = getattr(inf, "it2t_compare_func", it2t_summary_func)
        captionist_it2t_func = it2t_summary_func
        foodie_it2t_func = it2t_compare_func
        editor_it2t_func = it2t_summary_func
    else:
        captionist_it2t_func = None
        foodie_it2t_func = None
        editor_it2t_func = None

    if use_vlm:
        import importlib
        flux_i2i_func = importlib.import_module(inference).flux_i2i_func
        artist_i2i_func = flux_i2i_func
    else:
        artist_i2i_func = None

    pipeline = StorybearPipeline(
        captionist_it2t_func=captionist_it2t_func,
        foodie_it2t_func=foodie_it2t_func,
        editor_it2t_func=editor_it2t_func,
        artist_i2i_func=artist_i2i_func,
    )
    named_stages_list = pipeline.get_stages()

    # current_value = gr.State([0])
    num_stages = len(named_stages_list)
    pipeline_blocks = []
    
    with gr.Blocks(title="storybear") as demo:
        gr.Markdown("## STORYBEAR: from science to fairytale via agent-assisted storytelling")
        status_output = gr.Textbox(label="Status")
        
        # demo_button.click(click_demo, [], pipeline_blocks[0])
        with gr.Row("Parent container"):
            with gr.Column():
                df_hf_path = gr.Textbox(label="Select data table", lines=3, value=DATA_EXAMPLES[0])
                demo_button = gr.Button("Click the button to launch on demo dataset")
                gr.Examples(DATA_EXAMPLES, df_hf_path, label="Select dataset")
                
            with gr.Column():
                with gr.Accordion("Pipeline status dashboard"):
                    gr.Markdown((
                        "Each of the checkboxes correspond to one of the pipeline stages. "
                        "The checkbox is checked when the stage is finished."
                        "Current stage is executed with the progress bar. \n"
                        "To rerun the pipeline, first click on 'Clear button' below"
                    ))
                    for i, (step_name, step_func) in enumerate(named_stages_list):
                        # for i in range(num_steps):
                        pipeline_blocks.append(
                            gr.Checkbox(value=False, label=f"Stage {i}: {step_name}", interactive=False)
                        )
                    clear_checkboxes_button = gr.ClearButton()

        with gr.Column("Parent container") as container:
            @gr.render(inputs=[pipeline_blocks[-1]])
            def show_demo_view(count, request: gr.Request):
                print(count)
                if request.session_hash in instances:
                    report = instances[request.session_hash]['report']
                    with gr.Row("Header line"):
                        blocks = [
                            gr.Markdown(f"# {report.header}"),
                            gr.Markdown(f"## {report.lead}"),
                        ]
                    for i, record in enumerate(report.plot_record_list):
                        text = gr.Label(record.caption)
                        im = gr.Image(record.plot_path)
                        blocks.append(gr.Row(f"Row_{i}", [text, im]))
            

        clear_checkboxes_button.click(call_clear_checkboxes, pipeline_blocks, pipeline_blocks)
        stage_processor = StageProcessor(named_stages_list)

        for i in range(num_stages - 1):
            input_block = pipeline_blocks[i]
            output_block = pipeline_blocks[i+1]
            # stage_name, stage_func = named_step_list[i+1]
            process_step = stage_processor[i+1]
            input_block.change(process_step, [input_block], [output_block])

        output_block.change(stage_processor.finish_job, [output_block], [status_output])
        output_block.change(show_demo_view, [output_block], [])

        #stage_name, stage_func = named_step_list[0]
        restart_pipeline = stage_processor[0]
        #restart_pipeline = StageProcessor(stage_name, stage_func).restart_pipeline
        demo_button.click(restart_pipeline, [], [pipeline_blocks[0]])

        demo.load(initialize_instance, inputs=None, outputs=status_output)    
        # Clean up instance when page is closed/refreshed
        demo.unload(cleanup_instance)   

    return demo
