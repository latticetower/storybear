import gradio as gr


def greet(name, intensity):
    return "Hello, " + name + "!" * int(intensity)


def create_app():
    demo = gr.Interface(
        fn=greet,
        inputs=["text", "slider"],
        outputs=["text"],
        api_name="predict"
    )
    return demo