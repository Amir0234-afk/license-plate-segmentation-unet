from tensorflow import keras
from tensorflow.keras import layers

def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    return x

def down(x, filters):
    c = conv_block(x, filters)
    p = layers.MaxPooling2D()(c)
    return c, p

def up(x, skip, filters):
    x = layers.UpSampling2D()(x)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, filters)
    return x

def build_unet(img_size=(256,256), num_classes=3):
    inputs = layers.Input(shape=(*img_size, 3))
    c1,p1 = down(inputs, 64)
    c2,p2 = down(p1, 128)
    c3,p3 = down(p2, 256)
    c4,p4 = down(p3, 512)

    b = conv_block(p4, 1024)

    u1 = up(b, c4, 512)
    u2 = up(u1, c3, 256)
    u3 = up(u2, c2, 128)
    u4 = up(u3, c1, 64)

    outputs = layers.Conv2D(num_classes, 1, activation='softmax')(u4)
    model = keras.Model(inputs, outputs, name='unet')
    return model
