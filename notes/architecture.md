Model Architecture
=========
## Frame-wise image encoder φI
We employ the ViT-B/16 model [28] to generate a 512-dimensional representation for every RGB frame. We initialize the model's weights using the publicly available checkpoint from MineCLIP [92] and solely fine-tune the final two layers throughout the training process. The video stream input resolution is 160 × 256, deviating from CLIP's standard 224 × 224 resolution，it runs at 5 frames per second (fps). To conserve computational resources, we evenly sample the video snippet to contain 16 frames. To adjust the positional embeddings, we employ bicubic interpolation, avoiding the introduction of additional learnable parameters.

The Transformer encoder (Vaswani et al., 2017) consists of alternating layers of multiheaded selfattention (MSA, see Appendix A) and MLP blocks (Eq. 2, 3). Layernorm (LN) is applied before every block, and residual connections after every block (Wang et al., 2019; Baevski & Auli, 2019)

AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE

```
resolution: The resolution of the input images.
patch_size: The size of the patches that the images are divided into.
width: The dimension of the hidden layers in the transformer.
layers: The number of layers in the transformer.
heads: The number of attention heads in the transformer.
output_dim: The dimension of the output layer.

The __init__ method initializes several layers and parameters of the transformer, including a convolutional layer (self.conv1), a class token (self.cls_token), positional embeddings (self.pos_embed), a sequence of residual attention blocks (self.blocks), and a projection matrix (self.projection).
```
The get_layer method returns the specified layer of the transformer. If the layer is 0, it returns the first four layers. If the layer is the last one, it returns the last two layers. Otherwise, it returns the specified layer from the sequence of residual attention blocks.


The forward method defines the forward pass of the transformer. It first applies the convolutional layer to the input, reshapes and permutes the output, adds the class token and positional embeddings, applies the sequence of residual attention blocks, applies the last layer normalization, and finally applies the projection matrix if it is not None. The output of the forward method is the output of the transformer for the given input. Here are more details:

Convolutional Layer: The method starts by applying a convolutional layer (self.conv1) to the input (x). This layer is used to extract features from the input image. The convolution operation involves sliding a filter over the input and computing the dot product of the filter and the input at each position.

Reshape and Permute: The output of the convolutional layer is then reshaped and permuted. The reshaping operation changes the output into a 2D tensor, where each row corresponds to a patch of the input image. The permute operation changes the order of the dimensions of the tensor to match the expected input format of the transformer.

Class Token and Positional Embeddings: The method then adds a class token (self.cls_token) and positional embeddings (self.pos_embed) to the reshaped and permuted output. The class token is a special token that is used to aggregate information from all patches of the image. The positional embeddings are used to provide information about the relative or absolute position of the patches in the image.

Residual Attention Blocks: The method then applies a sequence of residual attention blocks (self.blocks) to the output. Each residual attention block consists of a multi-head self-attention mechanism and a feed-forward neural network. The multi-head self-attention mechanism allows the model to focus on different parts of the input at the same time. The feed-forward neural network is used to transform the output of the attention mechanism.

Layer Normalization: After the sequence of residual attention blocks, the method applies a layer normalization (self.norm). Layer normalization is a type of normalization technique that normalizes the features of each sample independently, rather than across the batch.

Projection Matrix: Finally, if the projection matrix (self.projection) is not None, the method applies it to the output. The projection matrix is used to transform the output to the desired output dimension.

scaling the values of the tensor by a certain factor. Scaling the random initialization can help with the convergence of the model during training.

## Text Encoder φG 
Regarding the text encoder, inspired by the architecture of MineCLIP [9], we utilize a GPT model with 12 layers and a width of 512, featuring 8 attention heads. Textual input undergoes tokenization using the same tokenizer as CLIP and is padded or truncated to 77 tokens. Input strings are transformed into lower-case byte pair encoding, utilizing a vocabulary size of 49,152, and are truncated to a maximum of 77 tokens. 
We initialize the model's weights with the publicly available MineCLIP checkpoint and srestrict training to fine-tuning only the last two layers.

The text encoder adopts a Transformer architecture based on Vaswani et al. (2017), with adjustments outlined in Radford et al. (2019). We employ a base model with 63 million parameters, featuring 12 layers and a width of 512, alongside 8 attention heads. Input text undergoes lower-cased byte pair encoding (BPE) with a vocabulary size of 49,152, following the approach by Sennrich et al. (2015). To enhance computational efficiency, we set a maximum sequence length limit of 76. The text sequence is enclosed within [SOS] and [EOS] tokens, and the activations of the highest layer of the transformer at the [EOS] token are treated as the text's feature representation. These representations undergo layer normalization before being linearly projected into the multi-modal embedding space. 

```
embed_dim: The dimension of the embedding layer.
context_length: The length of the context for the transformer model.
vocab_size: The size of the vocabulary.
layers: The number of layers in the transformer model.
width: The dimension of the hidden layers.
heads: The number of attention heads.
is_discrete_text: A boolean flag indicating whether the input is discrete text or not.
```

The initialize_parameters method initializes the weights of the model using the normal distribution. The standard deviation of the distribution is different for different parts of the model.

The build_attention_mask method creates a causal attention mask for the transformer model. This mask ensures that during the self-attention computation, each token only attends to earlier positions in the sequence.

Input: The method takes a batch of text sequences as input. Each sequence is a list of token indices. The shape of the input is (batch_size, sequence_length).

Token Embedding: The input sequences are passed through the token embedding layer. This layer transforms each token index into a dense vector. The output of this layer has the shape (batch_size, sequence_length, embed_dim).

Positional Embedding: Positional embeddings are added to the token embeddings. These embeddings encode the position of each token in the sequence. They allow the model to consider the order of the tokens.

Transformer Blocks: The combined embeddings are then passed through a series of transformer blocks. Each block consists of a multi-head self-attention mechanism and a position-wise feedforward network, with residual connections and layer normalization around each. The self-attention mechanism allows each token to consider all other tokens when producing its output. The feedforward network transforms the output of the self-attention mechanism.

Layer Normalization: After passing through all transformer blocks, the output is passed through a layer normalization. This is a technique that stabilizes the learning process and reduces the training time.

Projection: Finally, the output of the layer normalization is passed through a linear layer (the projection). This layer reduces the dimension of the output to the size of the vocabulary, producing a score for each possible output token.

Output: The output of the model is a batch of sequences of token scores. Each score indicates the likelihood of the corresponding token being the next token in the sequence. The shape of the output is (batch_size, sequence_length, vocab_size).

Depending on the is_discrete_text flag, the forward method handles the output differently. If the flag is True, it takes the features from the token in the sequence that has the highest number. If the flag is False, it takes the features from the last token in the sequence.

## Temporal aggregator φa 
With a sequence of RGB features extracted frame by frame, a temporal aggregator network condenses the sequence into a single video embedding. Following this aggregation, we integrate two additional layers of residual CLIP Adapter [38]. The residual weights are initialized to closely resemble an identity function at the onset of training.

## Residual Attention Block
The __init__ method initializes the class with several components:
```
d_model: This is the dimension of the input and output of the block. It is also the dimension of the internal layers in the multi-head attention mechanism and the MLP.

n_head: This is the number of attention heads in the multi-head attention mechanism. The input dimension d_model should be divisible by n_head.

attn_mask: This is an optional attention mask that can be used to prevent certain elements from attending to others. If provided, it should be a binary tensor of shape (L, L) where L is the sequence length.
```
The class contains the following components:

self.attn: This is a multi-head attention mechanism. It takes in three inputs (query, key, value) and returns the attended values.

self.ln_1 and self.ln_2: These are layer normalization modules. They normalize the input across the feature dimension (dimension 1).

self.mlp: This is a multi-layer perceptron, which is a fully connected feed-forward network. It consists of two linear layers with a GELU activation function in between.

The attention method applies the multi-head attention mechanism to the input tensor x. If an attention mask is provided, it is moved to the same device and data type as x.

The forward method is where the residual connections take place. First, the input x is normalized and passed through the attention mechanism. The output is then added back to the original x (forming a residual connection). The result is again normalized and passed through the MLP, and the output of the MLP is added back to form a second residual connection.

This design allows for direct paths for the information and gradients to flow through the network, which can help alleviate the vanishing gradient problem in deep networks. The attention mechanism allows the model to focus on different parts of the input sequence when producing each element of the output sequence, which can be beneficial for tasks that require understanding the relationships between different parts of the input.