import tensorflow as tf
from tensorflow.keras import layers, models, applications

def get_backbone(name="VGG16", input_shape=(48,48,3), weights="imagenet", include_top=False):
    name_l = name.lower()
    if name_l == "vgg16":
        return applications.VGG16(weights=weights, include_top=include_top, input_shape=input_shape)
    if name_l == "resnet50":
        return applications.ResNet50(weights=weights, include_top=include_top, input_shape=input_shape)
    if name_l in ("efficientnetb0", "efficientnet"):
        return applications.EfficientNetB0(weights=weights, include_top=include_top, input_shape=input_shape)
    raise ValueError(f"Backbone {name} not supported")

def build_transfer_model(backbone_name, input_shape, num_classes, head_units=256, head_dropout=0.5, trainable_layers=0):
    base = get_backbone(backbone_name, input_shape=input_shape, weights="imagenet", include_top=False)
    # Freeze all initially
    for layer in base.layers:
        layer.trainable = False
    # Optionally unfreeze last N layers
    if trainable_layers > 0:
        for layer in base.layers[-trainable_layers:]:
            layer.trainable = True

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(head_units, activation="relu")(x)
    x = layers.Dropout(head_dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs=base.input, outputs=outputs, name=f"{backbone_name}_transfer")
    return model
