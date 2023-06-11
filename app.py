import io, keyword, torch, random, pickle
from tokenize import tokenize, untokenize
from transformer import *
from params import *
from torchtext.data.utils import get_tokenizer
from torchtext.legacy import data
from flask import Flask, request, jsonify
from flask_cors import cross_origin


# Init our Flask App
app = Flask(__name__)
# Add configs here
# -----------------


# Utility Functions
def tokenize_python_code(python_code_str, augment_prob=0.3):
    ignore_words = ['range', 'float', 'zip' 'char', 'list', 'dict', 'tuple', 'set', 
                    'enumerate', 'print', 'ord', 'int', 'len', 'sum', 'min', 'max']
    ignore_words.extend(keyword.kwlist)
    var_counter = 1
    python_tokens = list(tokenize(io.BytesIO(python_code_str.encode('utf-8')).readline))
    augmented_vars = {}
    result = []
    for i in range(0, len(python_tokens)):
        if python_tokens[i].type == 1 and python_tokens[i].string not in ignore_words:
            if i>0 and python_tokens[i-1].string in ['def', '.', 'import', 'raise', 'except', 'class']: 
                ignore_words.append(python_tokens[i].string)
                result.append((python_tokens[i].type, python_tokens[i].string))
            elif python_tokens[i].string in augmented_vars:  
                result.append((python_tokens[i].type, augmented_vars[python_tokens[i].string]))
            elif random.uniform(0, 1) > 1 - augment_prob: 
                augmented_vars[python_tokens[i].string] = 'var_' + str(var_counter)
                var_counter += 1
                result.append((python_tokens[i].type, augmented_vars[python_tokens[i].string]))
            else:
                ignore_words.append(python_tokens[i].string)
                result.append((python_tokens[i].type, python_tokens[i].string))
        else:
            result.append((python_tokens[i].type, python_tokens[i].string))
    return result


def load_vocabulary():
    source = data.Field(tokenize = get_tokenizer('spacy', language='en_core_web_sm'), init_token='<sos>', eos_token='<eos>', lower=True)
    target = data.Field(tokenize = tokenize_python_code, init_token='<sos>', eos_token='<eos>', lower=False)
    src_file = open("vocab/src_vocab.pkl", "rb")
    trg_file = open("vocab/trg_vocab.pkl", "rb")
    source.vocab = pickle.load(src_file)
    target.vocab = pickle.load(trg_file)
    src_file.close()
    trg_file.close()
    return source, target

def translate_sentence(query, src_vocab, trg_vocab, model, device, max_len = 1000):
    model.eval()   
    if isinstance(query, str):
        nlp = get_tokenizer('spacy', language='en_core_web_sm')
        tokens = [token.text.lower() for token in nlp(query)]
    else:
        tokens = [token.lower() for token in query]
    tokens = [src_vocab.init_token] + tokens + [src_vocab.eos_token]
    src_indexes = [src_vocab.vocab.stoi[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    src_mask = model.make_src_mask(src_tensor)
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)
    trg_indexes = [trg_vocab.vocab.stoi[trg_vocab.init_token]]
    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        trg_mask = model.make_trg_mask(trg_tensor) 
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        pred_token = output.argmax(2)[:,-1].item()
        trg_indexes.append(pred_token)
        if pred_token == trg_vocab.vocab.stoi[trg_vocab.eos_token]:
            break 
    trg_tokens = [trg_vocab.vocab.itos[i] for i in trg_indexes]
    return trg_tokens[1:], attention


def load_model(device, source, target):
    INPUT_DIM = len(source.vocab)
    print("Source Vocab: ")
    print(len(source.vocab))
    OUTPUT_DIM = len(target.vocab)
    print("Target Vocab: ")
    print(len(target.vocab))
    encoder = Encoder(INPUT_DIM, HID_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, device)
    decoder = Decoder(OUTPUT_DIM, HID_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, device)
    SRC_PAD_IDX = source.vocab.stoi[source.pad_token]
    print(SRC_PAD_IDX)
    TRG_PAD_IDX = target.vocab.stoi[target.pad_token]
    print(TRG_PAD_IDX)
    model = Transformer(encoder, decoder, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
    model.load_state_dict(torch.load('model/model_02.pt', map_location=device))
    return model

# Select Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Loading Our Vocabulary
source, target = load_vocabulary()
# Loading our model
model = load_model(device, source, target)

# available models
available_models = [
    "CodeVerbTLM-0.1B"
]

# inference types
inference_types = [
    "Comment2Python",
    # "Algo2Python",
]


# Our API Routes
@app.route('/')
@cross_origin()
def home():
    msg = {
        "API Name": "CodeVerb TLM API",
        "API Version": "v0.1",
        "API Status": "Running",
        "Available Models": available_models
    }
    return jsonify(msg), 200, {'Content-Type': 'application/json; charset=utf-8'}

@app.route('/api/predict', methods=['POST'])
@cross_origin()
def predict():
    if request.method == 'POST':
        data = request.json
        query = data['query']
        model_name = data['model']
        inference_type = data['inference_type']
        if inference_type not in inference_types:
            msg = {
                "error": "Inference type not available! Available inference types: {}".format(inference_types)
            }
            return jsonify(msg), 400, {'Content-Type': 'application/json; charset=utf-8'}
        if model_name not in available_models:
            msg = {
                "error": "Model not available! Available models: {}".format(available_models)
            }
            return jsonify(msg), 400, {'Content-Type': 'application/json; charset=utf-8'}
        
        translation, attention = translate_sentence(query.split(" "), source, target, model, device)
        predicted_code = untokenize(translation[:-1]).replace('utf-8', '')
        msg = {
            "query": query,
            "result": predicted_code
        }
        return jsonify(msg), 200, {'Content-Type': 'application/json; charset=utf-8'}


if __name__ == '__main__':
    app.run(port=5050,debug=True)