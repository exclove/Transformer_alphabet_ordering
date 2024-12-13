import torch
import torch.nn as nn
from global_name_space import get_config
from data_preparation import build_vocab_alphabet


# Get arguments from globalnamespace.py
parser = get_config()
args = parser.parse_args()

# Device setting
if torch.cuda.is_available():
    device = torch.device("cuda") 
    print("CUDA is available")

else :
    print("You can only use CPU")
    device = torch.device("cpu")



class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embed size needs to be div by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0] # batch size of query. Query.shape = (N, T, D)  
            # N is batch size. 한 번에 처리할 샘플(문장) 개수. 
            # T is sequence lengh. 각 샘플(문장)의 토큰 개수. 
            # D is Embedding Dimension. 각 토큰의 임베딩 크기
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        # Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
            # 원래 values의 형태는 (N, T, D)이며, 이를 (N, T, H, D_h)로 변환합니다.
	        # •	 H = \text{self.heads} : Attention Head 개수.
	        # •	 D_h = \text{self.head_dim} : 각 Head의 차원 크기.
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd -> nhqk", queries, keys)
        # queries shape : (N, query_len, heads, heads_dim)
        # keys shape : (N, key_len, heads, heads_dim)
        # energy shape : (N, heads, query_len, key_len)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim = 3)
        out = torch.einsum("nhql, nlhd -> nqhd", attention, values).reshape(
            N, query_len, self.heads*self.head_dim)
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # out: (N,query_len, heads, heads_dim) then flatten last two dimensions
        
        out = self.fc_out(out)
        return out
        

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out
    
class Encoder(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion
                )
            for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask)
        
        return out

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out
    
class Decoder(nn.Module):
    def __init__(self,
                 trg_vocab_size,
                 embed_size,
                 num_layers,
                 heads,
                 forward_expansion,
                 dropout,
                 device,
                 max_length,
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
             for _ in range(num_layers)]

        )

        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)  # enc_out = value of 125th line, key of 125th line

        out = self.fc_out(x)
        return out



class Transformer(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            trg_vocab_size,
            src_pad_idx,
            trg_pad_idx,
            embed_size= args.embed_size,
            num_layers=args.num_layers,
            forward_expansion=args.forward_expansion,
            heads=args.heads,
            dropout=args.dropout,
            device= device,
            max_length=args.max_length
    ):

        super(Transformer, self).__init__()

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length
        )

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)
    
    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        ) # Duplicate N number of Lower triangle matrix
        return trg_mask.to(self.device)
    
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_scr = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_scr, src_mask, trg_mask)
        return out


# Vocabulary 생성 (train.py와 동일)
vocab = build_vocab_alphabet()

# 모델 정의 (train.py와 동일한 하이퍼파라미터 사용)
model = Transformer(
    src_vocab_size=len(vocab),
    trg_vocab_size=len(vocab),
    src_pad_idx=vocab["<pad>"],
    trg_pad_idx=vocab["<pad>"],
    embed_size=args.embed_size,
    num_layers=args.num_layers,
    forward_expansion=args.forward_expansion,
    heads=args.heads,
    dropout=args.dropout,
    max_length=args.max_length,
    device=device
).to(device)

# 저장된 모델 가중치 불러오기
model_path = "model_epoch_100.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print(f"Model loaded successfully from {model_path}")


# 테스트 데이터 준비
test_input = "c a b"  # 평가할 입력 시퀀스
src_seq = [vocab["<sos>"]] + [vocab[token] for token in test_input.split()] + [vocab["<eos>"]]
src_seq += [vocab["<pad>"]] * (args.max_length - len(src_seq))  # 패딩 추가
src_seq = torch.tensor(src_seq).unsqueeze(0).to(device)

# 디코딩 초기화
test_seq = [vocab["<sos>"]]
test_seq_tensor = torch.tensor(test_seq).unsqueeze(0).to(device)

# 모델 예측 (Greedy Decoding)
predicted_tokens = []
with torch.no_grad():
    for _ in range(args.max_length):
        # 디코더 입력 갱신
        output = model(src_seq, test_seq_tensor)
        next_token = output[:, -1, :].argmax(-1).item()  # 가장 마지막 토큰의 예측
        if next_token == vocab["<eos>"]:  # <eos> 토큰이면 종료
            break
        predicted_tokens.append(next_token)
        test_seq_tensor = torch.cat([test_seq_tensor, torch.tensor([[next_token]]).to(device)], dim=1)

# 토큰 -> 문자열 변환
predicted_tokens = [list(vocab.keys())[list(vocab.values()).index(idx)] for idx in predicted_tokens]

# 출력 결과
print("Input Sequence:", test_input)
print("Predicted Output:", " ".join(predicted_tokens))