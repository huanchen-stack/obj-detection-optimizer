from optimizer import Optimizer

Optimizer(
    dep_filename="yolov4_dep.csv",
    prof_filenames=[
        "yolov4_prof.csv",
        "yolov4_prof.csv",
        "yolov4_prof.csv",
        "yolov4_prof.csv",
        "yolov4_prof.csv",
        "yolov4_prof.csv",
        "yolov4_prof.csv",
    ],
    bandwidth=2000,
    ignore_latency=False,
)