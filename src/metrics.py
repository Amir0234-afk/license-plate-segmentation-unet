import tensorflow as tf
def dice_coef(num_classes=3, smooth=1e-6):
    def dice(y_true, y_pred):
        axes = (0,1,2)
        intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
        denom = tf.reduce_sum(y_true + y_pred, axis=axes)
        dice = tf.reduce_mean((2.*intersection + smooth) / (denom + smooth))
        return dice
    return dice
