from craft_text_detector import (
    Craft,
    empty_cuda_cache,
    export_detected_regions,
    export_extra_results,
    get_prediction,
    load_craftnet_model,
    load_refinenet_model,
    read_image,
)


def load_craft():
    refine_net = load_refinenet_model(cuda=True)
    craft_net = load_craftnet_model(cuda=True)

    return craft_net, refine_net


def detect_text(image_numpy, craft_net, refine_net, i):
    prediction_result = get_prediction(
        image=image_numpy,
        craft_net=craft_net,
        refine_net=refine_net,
        text_threshold=0.7,
        link_threshold=0.4,
        low_text=0.2,
        cuda=True,
        long_size=400 - i * 20,
    )

    return prediction_result
