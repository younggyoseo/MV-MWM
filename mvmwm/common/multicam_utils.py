def get_additional_cams(config):
    additional_cams = dict()
    for key, val in config.additional_cams.items():
        if val["theta"][0] != "none":
            additional_cams[key] = val
    return additional_cams


def get_eval_cams(config):
    eval_cams = dict()
    for key, val in config.eval_cams.items():
        if val["theta"][0] != "none":
            eval_cams[key] = val
    return eval_cams