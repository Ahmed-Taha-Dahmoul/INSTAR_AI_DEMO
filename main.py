from instant_mesh import MemoryOptimizedInstantMesh
from demo import create_demo


def main():
    mesh = MemoryOptimizedInstantMesh(
        config_path="configs/instant-mesh-base.yaml"
    )

    demo = create_demo(mesh)
    demo.queue(max_size=2)
    demo.launch(
        server_name="0.0.0.0",
        server_port=43548,
        share=False,
    )


if __name__ == "__main__":
    main()
