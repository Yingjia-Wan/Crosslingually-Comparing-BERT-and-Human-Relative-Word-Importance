
import numpy as np
import scipy.special

# This code has been adapted from bertviz: https://github.com/jessevig/bertviz/

def get_attention_1st_layer_for_sentence(model, tokenizer, sentence):

    inputs = tokenizer.encode_plus(sentence, return_tensors='tf', add_special_tokens=True)
    input_ids = inputs['input_ids']
    attention_1st_layer = model(input_ids)[-1]
    input_id_list = input_ids[0].numpy().tolist() # Batch index 0
    tokens = tokenizer.convert_ids_to_tokens(input_id_list)
    
    return tokens, attention_1st_layer

# For the attention baseline, we fixed several experimental choices (see below) which might affect the results.
def calculate_relative_attention_1st_layer( tokens, attention_1st_layer):
    # We use the first layer for comparison
    layer = 0

    # We use the first element of the batch because batch size is 1
    attention_1st_layer = attention_1st_layer[layer][0]
    
    # 1. We take the mean over the 12 attention heads (like Abnar & Zuidema 2020)
    # I also tried the sum once, but the result was even worse
    mean_attention_1st_layer = np.mean(attention_1st_layer, axis=0)

    # We drop CLS and SEP tokens
    mean_attention_1st_layer = mean_attention_1st_layer[1:-1]

    # Optional: make plot to examine
    #    ax = sns.heatmap(mean_attention[1:-1, 1:-1], linewidth=0.5, xticklabels=tokens[1:-1], yticklabels=tokens[1:-1])
    #    plt.show()

    # 2. For each word, we sum over the attention to the other words to determine relative importance
    sum_attention_1st_layer = np.sum(mean_attention_1st_layer, axis=0)

    # Taking the softmax does not make a difference for calculating correlation
    # It can be useful to scale the salience signal to the same range as the human attention
    relative_attention_1st_layer = scipy.special.softmax(sum_attention_1st_layer)

    return tokens, relative_attention_1st_layer

def extract_attention_1st_layer(model, tokenizer, sentence):

    tokens, attention_1st_layer = get_attention_1st_layer_for_sentence(model, tokenizer, sentence)
    tokens, relative_attention_1st_layer = calculate_relative_attention_1st_layer(tokens, attention_1st_layer)

    return tokens, relative_attention_1st_layer
