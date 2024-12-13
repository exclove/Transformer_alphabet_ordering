import random
import torch


def generate_alphabet_data(num_samples=1000):
    data = []
    for _ in range(num_samples):
        letters = list("abcdefghijklmnopqrstuvwxyz")
        random.shuffle(letters)  # 랜덤하게 섞기
        sequence_length = random.randint(3, 10)  # 시퀀스 길이 설정
        input_seq = letters[:sequence_length]
        output_seq = sorted(input_seq)  # 정렬된 시퀀스가 정답
        data.append((" ".join(input_seq), " ".join(output_seq)))
    return data

generated_data = generate_alphabet_data()

def build_vocab_alphabet():
    vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2}
    idx = 3
    for letter in "abcdefghijklmnopqrstuvwxyz":
        vocab[letter] = idx
        idx += 1
    return vocab

letter_num = build_vocab_alphabet()

def preprocess_alphabet_data(data, vocab, max_len=28):
    inputs, targets = [], []
    for input_seq, output_seq in data:
        input_tokens = input_seq.split()
        target_tokens = output_seq.split()
        
        input_indices = [vocab["<sos>"]] + [vocab[token] for token in input_tokens] + [vocab["<eos>"]]
        target_indices = [vocab["<sos>"]] + [vocab[token] for token in target_tokens] + [vocab["<eos>"]]

        input_indices += [vocab["<pad>"]] * (max_len - len(input_indices))
        target_indices += [vocab["<pad>"]] * (max_len - len(target_indices))

        inputs.append(input_indices)
        targets.append(target_indices)
    
    return torch.tensor(inputs), torch.tensor(targets)

if __name__ == "__main__":
    data = generate_alphabet_data()
    vocab = build_vocab_alphabet()
    print(len(vocab))

    inputs, targets = preprocess_alphabet_data(data, vocab)
    # print("Inputs shape:", inputs.shape)
    # print("Targets shape:", targets.shape)