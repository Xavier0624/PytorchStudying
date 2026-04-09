import torch
import torch.nn as nn
import jieba
import torch.optim as optim
import time
from torch.utils.data import DataLoader

def build_vocab():
    unique_words, all_words = [], []
    for line in open('./data/jaychou_lyrics.txt', 'r', encoding='utf-8'):
        words = jieba.lcut(line)
        all_words.append(words)
        for word in words:
            if word not in unique_words:
                unique_words.append(word)
    word_count = len(unique_words)
    # print(word_count)
    word_to_index = {word:i for i, word in enumerate(unique_words)}
    print(word_to_index)

    corpus_idx = []
    for words in all_words:
        tmp = []
        for word in words:
            tmp.append(word_to_index[word])
        tmp.append(word_to_index[' '])
        corpus_idx.extend(tmp)

    return unique_words, word_to_index, corpus_idx, word_count
class LyricsDataset(torch.utils.data.Dataset):
    def __init__(self, corpus_idx, num_chars):
        self.corpus_idx = corpus_idx
        self.num_chars = num_chars
        self.word_count = len(corpus_idx)
        self.number = self.word_count // self.num_chars

    def __len__(self):
        return self.number
    def __getitem__(self, idx):
        start = min(max(idx, 0), self.word_count - self.num_chars - 1)
        end = start + self.num_chars
        x = self.corpus_idx[start:end]
        y = self.corpus_idx[start + 1:end + 1]
        return torch.tensor(x), torch.tensor(y)

class TextGenerator(nn.Module):
    def __init__(self, unique_word_count):
        super().__init__()
        # 词嵌入层 词数量 unique_word_count，词向量维度 128
        self.embedding = nn.Embedding(unique_word_count, 128)
        # RNN层 输入维度 128，隐藏层维度 256，层数 1
        self.rnn = nn.RNN(128, 256, 1)
        # 输出层 输入维度 256，输出维度 unique_word_count
        self.out = nn.Linear(256, unique_word_count)

    def forward(self, inputs, hidden):
        # 初始化词嵌入层
        embedded = self.embedding(inputs)
        # RNN层
        output, hidden = self.rnn(embedded.transpose(0, 1), hidden)
        output = self.out(output.reshape(shape=(-1, output.shape[-1])))
        return output, hidden

    def init_hidden(self, batch_size):
        # 初始化隐藏状态
        return torch.zeros((1, batch_size, 256))
def train():
    unique_words, word_to_index, corpus_idx, unique_word_count = build_vocab()
    dataset = LyricsDataset(corpus_idx, num_chars=32)

    model = TextGenerator(unique_word_count)
    lyrics_dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    for epoch in range(epochs):
        total_loss = 0.0
        start_time = time.time()
        iter_num = 0
        for x, y in lyrics_dataloader:
            hidden = model.init_hidden(5)
            output, hidden = model(x, hidden)
            y = torch.transpose(y, 0, 1).reshape(shape=(-1, ))
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            iter_num += 1
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / iter_num:.4f}, Time: {time.time() - start_time:.2f} seconds")

    torch.save(model.state_dict(), './model/text_generator.pth')
def evaluate(start_word, sentence_length):
    unique_words, word_to_index, corpus_idx, unique_word_count = build_vocab()
    model = TextGenerator(unique_word_count)
    model.load_state_dict(torch.load('./model/text_generator.pth'))
    hidden = model.init_hidden(1)
    word_idx = word_to_index[start_word]
    generate_sentence = [word_idx]
    for i in range(sentence_length):
        output, hidden = model(torch.tensor([[word_idx]]), hidden)
        word_idx = torch.argmax(output)
        generate_sentence.append(word_idx.item())
    for idx in generate_sentence:
        print(unique_words[idx], end='')
if __name__ == '__main__':
    # 构建词表
    # unique_words, word_to_index, corpus_idx, word_count = build_vocab()
    # 构建数据集
    # dataset = LyricsDataset(corpus_idx, num_chars=5)
    # print(len(dataset))
    # model = TextGenerator(len(unique_words))
    # for name, parameter in model.named_parameters():
    #     print(name, parameter.shape)
    # train()

    evaluate('分手', 50)