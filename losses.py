# import standard python packages/modules
import tensorflow as tf

def dice(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.clip_by_value(y_pred, 0.0, 1.0), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    return (2. * intersection + smooth) / (union + smooth)


# Dice for a specific class (foreground only)
def dice_coef_class(class_index, name=None, smooth=1e-6):
    """
    Returns a Dice metric for a specific class index (e.g., 1=myocardium, 2=blood pool).
    """
    return tf.keras.metrics.MeanMetricWrapper(dice, name=name or f'dice_class_{class_index}')


# Macro-average Dice over all foreground classes
def dice_coef_no_bkg(y_true, y_pred, smooth=1e-6):
    """
    Compute mean Dice Coefficient over all foreground classes (excluding background).
    Assumes multi-class one-hot encoding and softmax predictions.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Remove background (channel 0)
    y_true_fg = y_true[..., 1:]
    y_pred_fg = y_pred[..., 1:]

    # Flatten spatial dimensions, preserve class channels
    y_true_f = tf.reshape(y_true_fg, [-1, tf.shape(y_true_fg)[-1]])
    y_pred_f = tf.reshape(y_pred_fg, [-1, tf.shape(y_pred_fg)[-1]])

    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
    denominator = tf.reduce_sum(y_true_f + y_pred_f, axis=0)

    dice = (2. * intersection + smooth) / (denominator + smooth)
    mean_dice = tf.reduce_mean(dice)

    return mean_dice

def multiclass_dice_loss(y_true, y_pred, smooth=1e-6):
    """
    Multi-class soft Dice loss averaged over ALL classes (including background).
    Equivalent to focal_tversky_loss with alpha=0.5, beta=0.5, gamma=1.0.

    y_true: one-hot ground truth, shape (..., C)
    y_pred: softmax probabilities, shape (..., C)
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 0.0, 1.0)

    C = tf.shape(y_true)[-1]                         # number of classes
    y_true_f = tf.reshape(y_true, [-1, C])           # flatten batch+spatial dims
    y_pred_f = tf.reshape(y_pred, [-1, C])

    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)            # (C,)
    denominator  = tf.reduce_sum(y_true_f + y_pred_f, axis=0)            # (C,)
    dice_per_cls = (2.0 * intersection + smooth) / (denominator + smooth)

    return 1.0 - tf.reduce_mean(dice_per_cls)        # macro average over ALL classes


def focal_tversky_loss(y_true, y_pred, alpha=0.5, beta=0.5, gamma=1.0, smooth=1e-6): 
    """
    Focal Tversky loss for multi-class 3D segmentation.

    Args:
    y_true: tensor of shape [B, D, H, W, C]
    y_pred: tensor of shape [B, D, H, W, C]
    alpha: controls the penalty for false positives
    beta: controls the penalty for false negatives
    gamma: focal parameter to down-weight easy examples
    smooth: smoothing constant to avoid division by zero

    Returns:
    loss: computed Focal Tversky loss
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, smooth, 1.0 - smooth)  # Clipping to avoid log(0)

    num_classes = 3
    loss = 0.0

    for c in range(num_classes):
        y_true_c = y_true[..., c]
        y_pred_c = y_pred[..., c]
        
        true_pos = tf.reduce_sum(y_true_c * y_pred_c)
        false_neg = tf.reduce_sum(y_true_c * (1 - y_pred_c))
        false_pos = tf.reduce_sum((1 - y_true_c) * y_pred_c)
        
        tversky_index = (true_pos + smooth) / (true_pos + alpha * false_neg + beta * false_pos + smooth)
        loss_c = tf.pow((1 - tversky_index), gamma)
        loss += loss_c

    loss /= tf.cast(num_classes, tf.float32)  # Averaging over all classes
    return loss
