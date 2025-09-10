# Import standard python modules
import tensorflow as tf
import numpy as np

tf.random.set_seed(489154)

_ND_LAYERS = {
    ('Conv', 1): tf.keras.layers.Conv1D,
    ('Conv', 2): tf.keras.layers.Conv2D,
    ('Conv', 3): tf.keras.layers.Conv3D,
    ('MaxPool', 1): tf.keras.layers.MaxPool1D,
    ('MaxPool', 2): tf.keras.layers.MaxPool2D,
    ('MaxPool', 3): tf.keras.layers.MaxPool3D,
    ('UpSampling', 1): tf.keras.layers.UpSampling1D,
    ('UpSampling', 2): tf.keras.layers.UpSampling2D,
    ('UpSampling', 3): tf.keras.layers.UpSampling3D,
}

class unet3plus:
   """
   Class for building a U-Net3+ model.
   """

   def __init__(self,
                inputs,
                filters = [32,64,128,256,512],
                rank = 2,
                out_channels = 1,
                kernel_initializer=tf.keras.initializers.HeNormal(seed=0),
                bias_initializer=tf.keras.initializers.Zeros(),
                kernel_regularizer=None,
                bias_regularizer=None,
                add_dropout = False,
                padding = 'same',
                dropout_rate = 0.5,
                kernel_size = 3,
                out_kernel_size = 3,
                pool_size = 2,
                encoder_block_depth = 2,
                decoder_block_depth = 1,
                batch_norm = True,
                activation = 'relu',
                out_activation = None,
                skip_batch_norm = True,
                skip_type = 'encoder',
                CGM = False,
                deep_supervision = True):
       
       """
       Initialise the U-Net3+ model.
       Args:
           inputs: Input tensor.
           filters: List of filter sizes for each UNet level.
           rank: Number of dimensions (2D or 3D).
           out_channels: Number of output channels (for segmentation this shall be the number of distinct masks).
           kernel_initializer: Initialiser for the convolutional layers.
           bias_initializer: Initialiser for the bias terms.
           kernel_regularizer: Regulariser for the convolutional layers.
           bias_regularizer: Regulariser for the bias terms in convolutional layers.
           add_dropout: Whether to add dropout layers.
           padding: Padding type for the convolutional layers.
           dropout_rate: Dropout rate.
           kernel_size: Kernel size for the convolutional layers.
           out_kernel_size: Kernel size for the final convolutional layers of the network.
           pool_size: Pooling size for the max pooling layers. This can be a tuple specifing the max pool size for each dimension of the input, or a single integer specifying the same size for all dimensions.
           encoder_block_depth: Number of convolutional blocks in each level of the encoding arm.
           decoder_block_depth: Number of convolutional blocks in each level of the decoding arm.
           batch_norm: Whether to use batch normalization.
           activation: Activation function for the convolutional layers.
           out_activation: Activation function for the output layer. For binary segmentation this shall be 'sigmoid' or 'softmax'.
           skip_batch_norm: Whether to use batch normalization in the skip connections.
           skip_type: Type of skip connections to use in the model ('encoder', 'decoder', or 'standard_unet').
           CGM: Whether to use CGM in the model for segmentation (Classification Guided Module).
           deep_supervision: Whether to use deep supervision.
        """
        # Assign parameters
       self.inputs = inputs
       self.filters = filters
       self.levels = len(filters)
       self.rank = rank
       self.out_channels = out_channels
       self.encoder_block_depth = encoder_block_depth
       self.decoder_block_depth = decoder_block_depth
       self.kernel_size = kernel_size
       self.add_dropout = add_dropout
       self.dropout_rate = dropout_rate
       self.skip_type = skip_type  
       self.skip_batch_norm = skip_batch_norm
       self.batch_norm = batch_norm
       self.activation = activation
       self.out_activation = out_activation
       self.CGM = CGM
       self.deep_supervision = deep_supervision
       # Assign pool size based on given tuple, or if single integer is provided, assign the same value to all dimensions using the rank as a guide for the number of dimensions
       if isinstance(pool_size,tuple):
           self.pool_size = pool_size
       else:
           self.pool_size = tuple([pool_size for _ in range(rank)])
        # Assign kernel sizes based on given tuple, or if single integer is provided, assign the same value to all dimensions using the rank as a guide for the number of dimensions
       if isinstance(kernel_size,tuple):
           self.kernel_size = kernel_size
       else:
           self.kernel_size = tuple([kernel_size for _ in range(rank)])
       if isinstance(out_kernel_size,tuple):
           self.out_kernel_size = out_kernel_size
       else:
           self.out_kernel_size = tuple([out_kernel_size for _ in range(rank)])
       # Create the conv and out conv config dictionaries for the conv and out conv layers
       self.conv_config = dict(kernel_size = self.kernel_size,
                          padding = padding,
                          kernel_initializer = kernel_initializer,
                          bias_initializer = bias_initializer,
                          kernel_regularizer = kernel_regularizer,
                          bias_regularizer = bias_regularizer)
       self.out_conv_config = dict(kernel_size = out_kernel_size,
                          padding = padding,
                          kernel_initializer = kernel_initializer,
                          bias_initializer = bias_initializer,
                          kernel_regularizer = kernel_regularizer,
                          bias_regularizer = bias_regularizer)
   
   def aggregate_and_decode(self, input_list, level):
    """
    Aggregates the inputs for the decoder levels and applies convolution to get the output of the decoder level.

    Args:
        input_list: List of inputs to the decoder to be aggregated.
        level: Current decoder level.
    """
    X = ResizeAndConcatenate(name = f'D{level}_input', axis = -1)(input_list) # Takes the various inputs to a decoder level, resizes them to the 1st input size in the list and the concatenates them all.
    X = self.conv_block(X, self.filters[level-1], block_depth = self.decoder_block_depth, conv_block_purpose = 'Decoder', level=level) # Performs a decoder block convolution of the concatenated input (i.e. the concatenated list of filters)
    return X
   
   def deep_sup(self, inputs, level):
    """
    If deep supervision is used, then the network will output a prediction at each level of the decoder.
    This function upsamples the output of a decoder level, convolves it and then applies the output activation function (i.e. to get to the final output).
    If deep supervision is not used, then the network will only output a prediction at the final level of the decoder.

    Args:
        inputs: Input tensor.
        level: Current decoder level.
    """
    conv = _ND_LAYERS[('Conv', self.rank)] # gets a convolutional layer of the specified rank (2D or 3D)
    upsamp = _ND_LAYERS[('UpSampling', self.rank)] # gets an upsampling layer of the specified rank (2D or 3D)
    size = tuple(np.array(self.pool_size)** (abs(level-1))) # This specifies the amount of upsampling needed to get to the correct final output size. It is the maxpool size to the power of the current decoder level minus one.
    if self.rank == 2:
        upsamp_config = dict(size=size, interpolation='bilinear') # use bilinear interpolation for 2D upsampling
    else:
        upsamp_config = dict(size=size) # for 3D upsampling, you cannot do bilinear interpolation, so this just uses the default upsampling method.
    X = inputs  
    X = conv(self.out_channels, activation = None, **self.out_conv_config, name = f'deepsup_conv_{level}_1')(X) # Convolves the input to get the correct number of output channels
    if level != 1:
        X = upsamp(**upsamp_config, name = f'deepsup_upsamp_{level}')(X) # Upsamples the convolved input to the correct size for the final output
    X = conv(self.out_channels, activation = None, **self.out_conv_config, name = f'deepsup_conv_{level}_2')(X) # Convolves the upsampled input to get the correct number of output channels (e.g. to correct artifacts due to upsampling)
    if self.out_activation:
        X = tf.keras.layers.Activation(activation = self.out_activation, name = f'deepsup_activation_{level}')(X) # Applies the output activation function to get the final output
    return X
       
       
       
   def skip_connection(self, inputs, to_level, from_level):
    """
    This function takes an input tensor and processes it as a skip connection to the decoder level.

    Args:
        inputs: Input tensor.
        to_level: Current decoder level.
        from_level: Level of UNet the input tensor is from.    
    """
    conv = _ND_LAYERS[('Conv', self.rank)] # gets a convolutional layer of the specified rank (2D or 3D)
    level_diff = from_level - to_level  # difference between level of decoder and level of UNet the input tensor is from
    size = tuple(np.array(self.pool_size)** (abs(level_diff))) # This specifies the amount of upsampling needed to get to the correct size for decoder level. It is the maxpool size to the power of the level difference.
    maxpool = _ND_LAYERS[('MaxPool', self.rank)] # gets a maxpool layer of the specified rank (2D or 3D)
    upsamp = _ND_LAYERS[('UpSampling', self.rank)] # gets an upsampling layer of the specified rank (2D or 3D)
    if self.rank == 2:
        upsamp_config = dict(size=size, interpolation='bilinear') # use bilinear interpolation for 2D upsampling
    else:
        upsamp_config = dict(size=size) # for 3D upsampling, you cannot do bilinear interpolation, so this just uses the default upsampling method.
    
    X = inputs        
    if to_level < from_level: # If coming from a deeper level of the UNet, then we need to upsample the input tensor to the correct size for the decoder level
        X = upsamp(**upsamp_config, name = f'Skip_Upsample_{from_level}_{to_level}')(X)
    elif to_level > from_level: # If coming from a shallower level of the UNet, then we need to maxpool the input tensor to the correct size for the decoder level
        X = maxpool(pool_size = size, name = f'Skip_Maxpool_{from_level}_{to_level}')(X)
    
    if self.skip_batch_norm: # If using batch normalization in the skip connections, then apply it within the conv block
        X = self.conv_block(X, self.filters[to_level-1], block_depth = self.decoder_block_depth, conv_block_purpose ='Skip', level = f'{from_level}_{to_level}') # applies conv block to the upsampled/maxpooled input tensor (with batch normalization)
    else:
        X = conv(self.filters[to_level-1],**self.conv_config, name = f'Skip_Conv_{from_level}_{to_level}')(X)  # applies conv layer to the upsampled/maxpooled input tensor (without batch normalization)
        
    return X # note: returns the output of a single skip connection, but does not yet concatenate the output to the other skip outputs or existing decoder level filters. 
       
   def conv_block(self, inputs, filters, block_depth, conv_block_purpose, level):
       """
       This function creates a convolutional block with the specified number of stacks and filters.
         Args:
                inputs: Input tensor.
                filters: Number of filters for the convolutional layers.
                block_depth: Number of convolutional stacks in the block.
                conv_block_purpose: Type of conv block (Encoder, Decoder, Skip).
                level: Current level level.
       """
       conv = _ND_LAYERS[('Conv', self.rank)] # gets a convolutional layer of the specified rank (2D or 3D)
       X = inputs
       for i in range(block_depth): # replicate the conv block, depth number of times
           X = conv(filters, **self.conv_config, name = f'{conv_block_purpose}{level}_Conv_{i+1}')(X) # applies conv layer to the input tensor
           if self.batch_norm: # If using batch normalization, then apply it after the conv layer
               X = tf.keras.layers.BatchNormalization(axis=-1, name = f'{conv_block_purpose}{level}_BN_{i+1}')(X) 
           if self.activation: # If using an activation function, then apply it after the conv layer
            X = tf.keras.layers.Activation(activation = self.activation, name = f'{conv_block_purpose}{level}_Activation_{i+1}')(X)
       return X
   
   
   def encode(self, inputs, level, block_depth):
       """
       Creates the encoding block of the U-Net3+ model.

         Args:
                inputs: Input tensor.
                level: Current level level.
                block_depth: Number of convolutional stacks in the block.
       """
       maxpool = _ND_LAYERS[('MaxPool', self.rank)] # gets a maxpool layer of the specified rank (2D or 3D)
       level -= 1 # python indexing
       filters = self.filters[level] # get the number of filters for the current level
       X = inputs
       if level != 0: # 0 is the input level, so we do not need to maxpool it
           X = maxpool(pool_size=self.pool_size, name = f'encoding_{level}_maxpool')(X) # maxpool the input tensor to the correct size for the next level
       X = self.conv_block(X, filters, block_depth, conv_block_purpose = 'Encoder', level = level+1) # applies conv block to the maxpooled input tensor
       if level == (self.levels-1) and self.add_dropout: # Check if level is the bottom level of the UNet, and if so, apply dropout if specified
           X = tf.keras.layers.Dropout(rate = self.dropout_rate, name = f'Encoder{level+1}_dropout')(X)
       return X
       
   def outputs(self):
       """
       This is the build function for the U-Net3+ model. 

       """
       XE  = [self.inputs] # This is a list of encoder level outputs, starting with the input tensor
       for i in range(self.levels): # for each level of the UNet, we apply an encoding block to the output of the previous level
           XE.append(self.encode(XE[i], level = i+1, block_depth = self.encoder_block_depth))
       XD = [XE[-1]] # This is a list of decoder level outputs, starting with the output of the last encoder level
       if self.skip_type == 'encoder': 
           # If using encoder-type skip connections, then we apply skip connections from every encoder level to the current decoder level - except the encoder level one deeper. For this level, we use the output of the last decoder level.
           for decoder_level in range(self.levels-1,0,-1): # build the decoder levels in reverse order
               input_contributions = []
               for unet_level in range(1,self.levels+1):
                   if unet_level == decoder_level+1: # If the unet level is one deeper than the decoder level, then we get a skip connection from the output of the last decoder level
                       input_contributions.append(self.skip_connection(XD[-1], decoder_level, unet_level))
                   else: # Otherwise we get a skip connection from the output of the encoder level
                       input_contributions.append(self.skip_connection(XE[unet_level], decoder_level, unet_level))
               XD.append(self.aggregate_and_decode(input_contributions,decoder_level)) # aggregate and conv the skip connections to the current decoder level. This gives the output of the decoder level. Append this to the list of decoder level outputs.
       elif self.skip_type == 'decoder':
           # If using decoder-type skip connections, then 
           for decoder_level in range(self.levels-1,0,-1):
               skip_contributions = []
               # Append skips from encoder
               for encoder_level in range(1,decoder_level+1): # All encoders shallower or equal to the decoder level contribute a skip connection
                   skip_contributions.append(self.skip_connection(XE[encoder_level], decoder_level, encoder_level))
               # Append skips from decoder
               for i in range(len(XD)-1,-1,-1): # All decoders deeper than the current decoder level contribute a skip connection (note: XD is build iteratively in a loop from the deepest level upwards. Therefore at each stage of the loop, XD grows and deeper decoder levels contribute skip connections to the current decoder level)
                   skip_contributions.append(self.skip_connection(XD[i], decoder_level, (self.levels-i)))
               XD.append(self.aggregate_and_decode(skip_contributions,decoder_level)) # aggregate and conv the skip connections to the current decoder level. This gives the output of the decoder level. Append this to the list of decoder level outputs.
       elif self.skip_type == 'standard_unet':
           # If standard_unet type skips, then at each decoder level, we get a skip connection from the corresponding encoder level 
           for decoder_level in range(self.levels-1,0,-1): 
               skip_contributions = [XE[decoder_level],self.skip_connection(XD[-1],decoder_level,decoder_level+1)]
               XD.append(self.aggregate_and_decode(skip_contributions,decoder_level)) # aggregate and conv the skip connections to the current decoder level.
       else:
           raise ValueError(f"Invalid skip_type")
       if self.deep_supervision == True:
           XD = [self.deep_sup(xd, self.levels-i) for i,xd in enumerate(XD)] # If deep supervision is used, then we apply deep supervision to each decoder level output
           return XD
       else:
           XD[-1] = self.deep_sup(XD[-1],1) # If deep supervision is not used, then we only apply deep supervision to the final decoder level output
           return XD[-1]


class ResizeAndConcatenate(tf.keras.layers.Layer):
  """Resizes and concatenates a list of inputs.

  Similar to `tf.keras.layers.Concatenate`, but if the inputs have different
  shapes, they are resized to match the shape of the first input.

  Args:
    axis: Axis along which to concatenate.
  """
  def __init__(self, axis=-1, **kwargs):
    super().__init__(**kwargs)
    self.axis = axis

  def get_config(self):
        config = super().get_config()
        config.update({
            "axis": self.axis,
        })
        return config

  def call(self, inputs):  
    if not isinstance(inputs, (list, tuple)):
      raise ValueError(
          f"Layer {self.__class__.__name__} expects a list of inputs. "
          f"Received: {inputs}")

    rank = inputs[0].shape.rank
    if rank is None:
      raise ValueError(
          f"Layer {self.__class__.__name__} expects inputs with known rank. "
          f"Received: {inputs}")
    if self.axis >= rank or self.axis < -rank:
      raise ValueError(
          f"Layer {self.__class__.__name__} expects `axis` to be in the range "
          f"[-{rank}, {rank}) for an input of rank {rank}. "
          f"Received: {self.axis}")
    # Canonical axis (always positive).
    axis = self.axis % rank

    # Resize inputs.
    shape = inputs[0].shape[1]
    resized = [tf.image.resize_with_crop_or_pad(tensor, shape, shape)
               for tensor in inputs[1:]]

    # Set the static shape for each resized tensor.
    for i, tensor in enumerate(resized):
      static_shape = inputs[0].shape.as_list()
      static_shape[axis] = inputs[i + 1].shape.as_list()[axis]
      static_shape = tf.TensorShape(static_shape)
      resized[i] = tf.ensure_shape(tensor, static_shape)

    return tf.concat(inputs[:1] + resized, axis=self.axis)  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
