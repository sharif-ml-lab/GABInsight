import argparse
import logging
import os


def main(
    space,
    method,
    data,
    task,
    gpath,
    rpath,
    cpath,
    opath,
    model,
    prompt,
    neg_prompt,
    count,
):
    all_task = task == "report"
    output = []

    if space == "metric":
        if data == "image":
            if method == "quality":
                if all_task or task == "inception":
                    output.append(metric_handlers.inception_handler(gpath))
                if all_task or task == "frechet":
                    output.append(metric_handlers.frechet_handler(gpath, rpath))
            elif method == "diversity":
                if all_task or task == "perceptual":
                    output.append(metric_handlers.perceptual_handler(gpath))
                if all_task or task == "coverage":
                    output.append(metric_handlers.coverage_image_handler(gpath, rpath))
                if all_task or task == "ssim":
                    output.append(metric_handlers.ssim_image_handler(gpath))

    elif space == "genai":
        if data == "text":
            if method == "llm":
                if task == "diverse":
                    generator_handlers.prompts_llm_handler(opath, model, prompt, count)
            if method == "config":
                if task == "llm-diversity":
                    generator_handlers.llm_diversity_handler()

    elif space == "pipeline":
        if data == "text":
            if method == "full":
                pipeline_handlers.full_generation(cpath, opath)

    elif space == "experiment":
        if task == "activity-retrieval":
            experiment_handlers.activity_retrieval_experiment(gpath, opath)
        elif task == "bias":
            experiment_handlers.bias_experiment(gpath, opath)
        elif task == "text-encoder-bias":
            experiment_handlers.text_encoder_bias_experiment(opath)
        elif task == "text-image-retrieval":
            experiment_handlers.text_image_retrieval_experiment(gpath, opath)

    print("\n".join(output))


if __name__ == "__main__":
    logging.getLogger("torch").setLevel(logging.CRITICAL)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    import handlers.generators as generator_handlers
    import handlers.metrics as metric_handlers
    import handlers.pipelines as pipeline_handlers
    import handlers.experiemtns as experiment_handlers

    parser = argparse.ArgumentParser(
        description="Sharif ML-Lab Data Generation ToolKit"
    )
    parser.add_argument(
        "-s", "--space", type=str, required=True, help="Space Name (e.g. metric, genai)"
    )
    parser.add_argument(
        "-mt",
        "--method",
        type=str,
        required=False,
        help="Method Name (e.g. quality, diversity, sdm)",
    )
    parser.add_argument(
        "-d", "--data", type=str, required=False, help="Kind of Data (e.g. image, text)"
    )
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        required=False,
        help="Task Name (e.g. inception, xlarge)",
    )
    parser.add_argument(
        "-gp", "--gpath", type=str, required=False, help="Generated Data Path"
    )
    parser.add_argument(
        "-cp", "--cpath", type=str, required=False, help="Caption Data Path"
    )
    parser.add_argument(
        "-rp", "--rpath", type=str, required=False, help="Real Data Path"
    )
    parser.add_argument(
        "-op", "--opath", type=str, required=False, help="Output Data Path"
    )
    parser.add_argument("-m", "--model", type=str, required=False, help="Model Name")
    parser.add_argument("-p", "--prompt", type=str, required=False, help="Prompt")
    parser.add_argument(
        "-np", "--neg-prompt", type=str, required=False, help="Negative Prompt"
    )
    parser.add_argument(
        "-n", "--count", type=int, required=False, help="Number of Images To Generate"
    )
    args = parser.parse_args()
    main(**vars(args))
