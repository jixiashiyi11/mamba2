DEFAULT_NORMAL_TEMPLATES = [
    "a photo of a normal {class_name}",
    "a photo of an intact {class_name}",
    "a photo of a flawless {class_name}",
    "a photo of an undamaged {class_name}",
]

DEFAULT_ABNORMAL_TEMPLATES = [
    "a photo of a damaged {class_name}",
    "a photo of a defective {class_name}",
    "a photo of an anomalous {class_name}",
    "a photo of a broken {class_name}",
]


def pretty_class_name(cls_name):
    return str(cls_name).replace("_", " ")


def instantiate_templates(templates, cls_name):
    pretty = pretty_class_name(cls_name)
    return [template.format(class_name=pretty, cls_name=pretty) for template in templates]

