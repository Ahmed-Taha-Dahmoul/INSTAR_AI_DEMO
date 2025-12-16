import gradio as gr
import rembg


def create_demo(instant_mesh):
    def preprocess(image, remove_bg):
        if remove_bg:
            image = rembg.remove(image)
        return image

    with gr.Blocks() as demo:
        img = gr.Image(type="pil", label="Input")
        out_img = gr.Image(label="Views")
        out_vid = gr.Video(label="Video")
        out_obj = gr.Model3D(label="OBJ")

        steps = gr.Slider(30, 75, value=75)
        seed = gr.Number(value=42)
        remove_bg = gr.Checkbox(value=True)

        btn = gr.Button("Generate")

        btn.click(
            fn=preprocess,
            inputs=[img, remove_bg],
            outputs=img,
        ).then(
            fn=instant_mesh.generate_mvs,
            inputs=[img, steps, seed],
            outputs=[img, out_img],
        ).then(
            fn=instant_mesh.make3d,
            inputs=[out_img],
            outputs=[out_vid, out_obj],
        )

    return demo
