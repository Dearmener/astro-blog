---
title: 'GPT完全指南（三）：分词与词表构建'
description: '深入解析GPT的分词技术，包括BPE、WordPiece、SentencePiece算法原理与实现，以及如何从零构建高效词表'
pubDate: '2024-03-03'
heroImage: ''
category: '技术'
tags: ['GPT', 'Tokenization', 'BPE', 'NLP', '分词']
series: 'GPT完全指南'
seriesOrder: 3
---

分词（Tokenization）是GPT等大语言模型处理文本的第一步，也是最关键的预处理环节。一个好的分词策略直接影响模型的性能、效率和多语言能力。本文将深入探讨GPT系列使用的分词技术，从理论原理到代码实现，帮助你全面理解这一基础但重要的技术。

## 为什么需要分词？

### 从字符到Token

神经网络无法直接处理原始文本，需要将文本转换为数值表示。最简单的方法有两种极端：

**字符级别（Character-level）**：
- 词表大小极小（英文约100个字符）
- 序列长度极长，计算成本高
- 难以捕捉语义信息

**词级别（Word-level）**：
- 词表极大（英文常用词超过100万）
- 无法处理未登录词（OOV问题）
- 不同语言词表差异大

```python
# 两种极端的分词方式
text = "Hello, world!"

# 字符级别
char_tokens = list(text)
# ['H', 'e', 'l', 'l', 'o', ',', ' ', 'w', 'o', 'r', 'l', 'd', '!']

# 词级别
word_tokens = text.split()
# ['Hello,', 'world!']  # 标点问题
```

### 子词分词的诞生

子词分词（Subword Tokenization）是一个折中方案：
- **常见词**保持完整（如 "the", "is"）
- **罕见词**拆分为子词（如 "tokenization" → "token" + "ization"）
- 词表大小可控（通常3万-10万）
- 理论上可以表示任意文本

## BPE算法详解

### 算法原理

Byte Pair Encoding（BPE）最初是一种数据压缩算法，后被应用于NLP分词。核心思想是**迭代合并最频繁出现的字符对**。

**训练过程**：
1. 初始化词表为所有字符
2. 统计所有相邻字符对的频率
3. 合并频率最高的字符对，加入词表
4. 重复步骤2-3，直到达到目标词表大小

**示例**：

```
初始语料: "low lower lowest"

初始词表: ['l', 'o', 'w', 'e', 'r', 's', 't', ' ']

第1轮: 'lo' 出现3次，最频繁 → 合并
词表: ['l', 'o', 'w', 'e', 'r', 's', 't', ' ', 'lo']

第2轮: 'low' 出现3次 → 合并
词表: ['l', 'o', 'w', 'e', 'r', 's', 't', ' ', 'lo', 'low']

第3轮: 'er' 出现1次, 'est' ... → 继续合并
...
```

### Python实现BPE训练

```python
from collections import defaultdict
import re

class BPETokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.merges = {}  # 合并规则
        self.vocab = {}   # 词表
        
    def get_stats(self, vocab):
        """统计所有相邻字符对的频率"""
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs
    
    def merge_vocab(self, pair, vocab):
        """合并词表中的字符对"""
        new_vocab = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        
        for word, freq in vocab.items():
            new_word = word.replace(bigram, replacement)
            new_vocab[new_word] = freq
        return new_vocab
    
    def train(self, corpus):
        """训练BPE模型"""
        # 预处理：将文本转换为字符序列，词尾加</w>标记
        word_freqs = defaultdict(int)
        for text in corpus:
            words = text.strip().split()
            for word in words:
                # 字符之间加空格，词尾加特殊标记
                word_chars = ' '.join(list(word)) + ' </w>'
                word_freqs[word_chars] += 1
        
        vocab = dict(word_freqs)
        
        # 初始词表：所有字符
        self.vocab = set()
        for word in vocab:
            for char in word.split():
                self.vocab.add(char)
        
        # 迭代合并
        num_merges = self.vocab_size - len(self.vocab)
        for i in range(num_merges):
            pairs = self.get_stats(vocab)
            if not pairs:
                break
                
            # 找最频繁的pair
            best_pair = max(pairs, key=pairs.get)
            
            # 合并
            vocab = self.merge_vocab(best_pair, vocab)
            
            # 记录合并规则
            self.merges[best_pair] = ''.join(best_pair)
            self.vocab.add(''.join(best_pair))
            
            if (i + 1) % 100 == 0:
                print(f"Merge {i+1}: {best_pair} -> {''.join(best_pair)}")
        
        print(f"Final vocab size: {len(self.vocab)}")
        return self.vocab, self.merges
    
    def tokenize(self, text):
        """使用训练好的BPE模型分词"""
        tokens = []
        for word in text.strip().split():
            word = ' '.join(list(word)) + ' </w>'
            
            # 按照训练时的合并顺序应用规则
            for pair, merged in self.merges.items():
                bigram = ' '.join(pair)
                word = word.replace(bigram, merged)
            
            tokens.extend(word.split())
        
        return tokens

# 使用示例
corpus = [
    "low lower lowest",
    "new newer newest",
    "the quick brown fox jumps over the lazy dog"
]

tokenizer = BPETokenizer(vocab_size=100)
vocab, merges = tokenizer.train(corpus)

# 测试分词
test_text = "lower newest"
tokens = tokenizer.tokenize(test_text)
print(f"Input: {test_text}")
print(f"Tokens: {tokens}")
```

## GPT使用的Byte-level BPE

### 为什么是Byte-level？

GPT-2引入了Byte-level BPE，与传统BPE的区别：

| 特性 | 传统BPE | Byte-level BPE |
|------|---------|----------------|
| 基础单元 | Unicode字符 | UTF-8字节（256个） |
| 初始词表 | 所有出现的字符 | 固定256个字节 |
| 未知字符 | 可能出现UNK | 永远不会有UNK |
| 多语言支持 | 需要大词表 | 天然支持 |

### 实现原理

```python
import regex as re

class ByteLevelBPE:
    """GPT-2风格的Byte-level BPE"""
    
    def __init__(self):
        # 字节到Unicode字符的映射
        # 将256个字节映射到可打印的Unicode字符
        self.byte_encoder = self._bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        
        # GPT-2的分词正则（预分词）
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
    
    def _bytes_to_unicode(self):
        """
        创建字节到Unicode的映射
        将0-255的字节映射到可打印字符
        """
        # 可打印的ASCII字符
        bs = list(range(ord("!"), ord("~") + 1))
        bs += list(range(ord("¡"), ord("¬") + 1))
        bs += list(range(ord("®"), ord("ÿ") + 1))
        
        cs = bs[:]
        n = 0
        # 不可打印的字节映射到256之后的Unicode
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        
        cs = [chr(n) for n in cs]
        return dict(zip(bs, cs))
    
    def encode_bytes(self, text):
        """将文本转换为字节表示的字符串"""
        return ''.join(self.byte_encoder[b] for b in text.encode('utf-8'))
    
    def decode_bytes(self, tokens):
        """将字节表示转回文本"""
        text = ''.join(tokens)
        return bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors='replace')
    
    def pre_tokenize(self, text):
        """预分词：使用正则将文本切分为初步的token"""
        return re.findall(self.pat, text)

# 演示字节编码
bpe = ByteLevelBPE()

text = "Hello, 世界! 🌍"
encoded = bpe.encode_bytes(text)
print(f"Original: {text}")
print(f"Byte-encoded: {encoded}")
print(f"Pre-tokens: {bpe.pre_tokenize(text)}")
```

### GPT-2/GPT-3的完整Tokenizer

```python
import json
import regex as re
from functools import lru_cache

class GPT2Tokenizer:
    """GPT-2 Tokenizer的简化实现"""
    
    def __init__(self, encoder_path, bpe_merges_path):
        # 加载词表
        with open(encoder_path, 'r') as f:
            self.encoder = json.load(f)
        self.decoder = {v: k for k, v in self.encoder.items()}
        
        # 加载BPE合并规则
        with open(bpe_merges_path, 'r') as f:
            bpe_data = f.read().split('\n')[1:-1]
        self.bpe_ranks = {tuple(merge.split()): i for i, merge in enumerate(bpe_data)}
        
        # 字节编码
        self.byte_encoder = self._bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        
        # 预分词正则
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
        
        self.cache = {}
    
    def _bytes_to_unicode(self):
        bs = list(range(ord("!"), ord("~")+1))
        bs += list(range(ord("¡"), ord("¬")+1))
        bs += list(range(ord("®"), ord("ÿ")+1))
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        return dict(zip(bs, [chr(n) for n in cs]))
    
    def get_pairs(self, word):
        """获取word中所有相邻的字符对"""
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs
    
    def bpe(self, token):
        """对单个token应用BPE"""
        if token in self.cache:
            return self.cache[token]
        
        word = tuple(token)
        pairs = self.get_pairs(word)
        
        if not pairs:
            return token
        
        while True:
            # 找rank最小（最早合并）的pair
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            
            if bigram not in self.bpe_ranks:
                break
            
            first, second = bigram
            new_word = []
            i = 0
            
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                
                if word[i] == first and i < len(word) - 1 and word[i+1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = self.get_pairs(word)
        
        word = ' '.join(word)
        self.cache[token] = word
        return word
    
    def encode(self, text):
        """编码文本为token ids"""
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            # 转换为字节表示
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            # 应用BPE
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens
    
    def decode(self, tokens):
        """解码token ids为文本"""
        text = ''.join(self.decoder[token] for token in tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors='replace')
        return text
```

## 使用tiktoken库

OpenAI开源了高性能的tokenizer库tiktoken：

```python
import tiktoken

# GPT-4使用cl100k_base编码
enc = tiktoken.get_encoding("cl100k_base")

# 也可以直接获取模型的编码
enc = tiktoken.encoding_for_model("gpt-4")

# 编码
text = "Hello, world! 这是一个测试。"
tokens = enc.encode(text)
print(f"Text: {text}")
print(f"Tokens: {tokens}")
print(f"Token count: {len(tokens)}")

# 解码
decoded = enc.decode(tokens)
print(f"Decoded: {decoded}")

# 查看每个token对应的文本
for token in tokens:
    print(f"  {token} -> {repr(enc.decode([token]))}")
```

输出示例：
```
Text: Hello, world! 这是一个测试。
Tokens: [9906, 11, 1917, 0, 220, 41565, 21043, 16937, 17161, 1811]
Token count: 10
Decoded: Hello, world! 这是一个测试。
  9906 -> 'Hello'
  11 -> ','
  1917 -> ' world'
  0 -> '!'
  220 -> ' '
  41565 -> '这是'
  21043 -> '一个'
  16937 -> '测试'
  17161 -> '。'
```

### 不同GPT版本的Tokenizer对比

```python
import tiktoken

encodings = {
    "gpt2": tiktoken.get_encoding("gpt2"),           # GPT-2
    "p50k_base": tiktoken.get_encoding("p50k_base"), # text-davinci-003
    "cl100k_base": tiktoken.get_encoding("cl100k_base"), # GPT-3.5/GPT-4
    "o200k_base": tiktoken.get_encoding("o200k_base"),   # GPT-4o
}

test_texts = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "这是一段中文文本。",
    "🎉 Emoji test! 🚀",
    "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
]

for text in test_texts:
    print(f"\n{'='*60}")
    print(f"Text: {text}")
    for name, enc in encodings.items():
        tokens = enc.encode(text)
        print(f"  {name:15} : {len(tokens):3} tokens")
```

## 词表构建最佳实践

### 1. 训练数据准备

```python
def prepare_training_data(files, sample_size=1000000):
    """
    准备BPE训练数据
    - 采样以控制训练时间
    - 确保语言/领域平衡
    """
    import random
    
    all_text = []
    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            all_text.extend(f.readlines())
    
    # 采样
    if len(all_text) > sample_size:
        all_text = random.sample(all_text, sample_size)
    
    return all_text

def analyze_corpus(texts):
    """分析语料库统计信息"""
    from collections import Counter
    
    char_counter = Counter()
    word_counter = Counter()
    total_chars = 0
    total_words = 0
    
    for text in texts:
        chars = list(text)
        words = text.split()
        
        char_counter.update(chars)
        word_counter.update(words)
        total_chars += len(chars)
        total_words += len(words)
    
    print(f"Total characters: {total_chars:,}")
    print(f"Total words: {total_words:,}")
    print(f"Unique characters: {len(char_counter):,}")
    print(f"Unique words: {len(word_counter):,}")
    print(f"\nTop 20 characters:")
    for char, count in char_counter.most_common(20):
        print(f"  {repr(char):10} : {count:,}")
    print(f"\nTop 20 words:")
    for word, count in word_counter.most_common(20):
        print(f"  {word:20} : {count:,}")
```

### 2. 选择合适的词表大小

| 词表大小 | 优点 | 缺点 | 适用场景 |
|---------|------|------|---------|
| 小（8K-16K） | 模型参数少，内存小 | 序列长，表达力弱 | 小模型，移动端 |
| 中（32K-50K） | 平衡 | - | 通用场景 |
| 大（100K+） | 表达力强，序列短 | 参数多，稀疏词 | 大模型，多语言 |

```python
def evaluate_vocab_size(tokenizer, test_texts):
    """评估不同词表大小的效果"""
    total_tokens = 0
    total_chars = 0
    unknown_count = 0
    
    for text in test_texts:
        tokens = tokenizer.encode(text)
        total_tokens += len(tokens)
        total_chars += len(text)
    
    compression_ratio = total_chars / total_tokens
    print(f"Compression ratio: {compression_ratio:.2f} chars/token")
    print(f"Average tokens per text: {total_tokens / len(test_texts):.1f}")
    
    return compression_ratio
```

### 3. 特殊Token处理

```python
class SpecialTokens:
    """GPT模型的特殊token"""
    
    # 通用特殊token
    PAD = "<|pad|>"
    UNK = "<|unk|>"
    BOS = "<|startoftext|>"
    EOS = "<|endoftext|>"
    
    # ChatGPT对话格式
    IM_START = "<|im_start|>"
    IM_END = "<|im_end|>"
    
    # GPT-4的特殊token
    SYSTEM = "<|system|>"
    USER = "<|user|>"
    ASSISTANT = "<|assistant|>"

def add_special_tokens(base_vocab, special_tokens):
    """向词表添加特殊token"""
    vocab = base_vocab.copy()
    max_id = max(vocab.values())
    
    for token in special_tokens:
        if token not in vocab:
            max_id += 1
            vocab[token] = max_id
    
    return vocab
```

## SentencePiece：统一的分词框架

Google的SentencePiece是一个语言无关的分词工具：

```python
import sentencepiece as spm

# 训练SentencePiece模型
spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='my_tokenizer',
    vocab_size=32000,
    model_type='bpe',  # 或 'unigram'
    character_coverage=0.9995,  # 字符覆盖率
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    user_defined_symbols=['<mask>', '<sep>'],  # 自定义特殊token
)

# 加载模型
sp = spm.SentencePieceProcessor()
sp.load('my_tokenizer.model')

# 编码
text = "Hello, world!"
pieces = sp.encode_as_pieces(text)  # ['▁Hello', ',', '▁world', '!']
ids = sp.encode_as_ids(text)        # [123, 45, 678, 90]

# 解码
decoded = sp.decode_pieces(pieces)
decoded = sp.decode_ids(ids)
```

### BPE vs Unigram

SentencePiece支持两种算法：

| 特性 | BPE | Unigram |
|------|-----|---------|
| 训练方式 | 自底向上合并 | 自顶向下删减 |
| 确定性 | 确定性分词 | 概率性，可采样多种分词 |
| 使用模型 | GPT系列 | T5, ALBERT, XLNet |
| 优点 | 简单直观 | 分词方式更灵活 |

```python
# Unigram训练
spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='unigram_tokenizer',
    vocab_size=32000,
    model_type='unigram',
)

# Unigram可以采样不同的分词方式
sp = spm.SentencePieceProcessor()
sp.load('unigram_tokenizer.model')

# 采样多种分词（用于数据增强）
for _ in range(3):
    pieces = sp.encode_as_pieces("tokenization", enable_sampling=True, alpha=0.1)
    print(pieces)
# 可能输出:
# ['▁token', 'ization']
# ['▁to', 'ken', 'iz', 'ation']
# ['▁token', 'iz', 'ation']
```

## 从零实现完整的Tokenizer

```python
import json
import os
from collections import defaultdict
import regex as re

class CustomGPTTokenizer:
    """
    自定义GPT风格的Tokenizer
    支持训练和推理
    """
    
    def __init__(self, vocab_size=50257):
        self.vocab_size = vocab_size
        self.encoder = {}
        self.decoder = {}
        self.bpe_ranks = {}
        
        # 字节编码映射
        self.byte_encoder = self._bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        
        # GPT-2风格的预分词正则
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
        
        self.cache = {}
        
        # 特殊token
        self.special_tokens = {
            '<|endoftext|>': 50256,
        }
    
    def _bytes_to_unicode(self):
        bs = list(range(ord("!"), ord("~")+1))
        bs += list(range(ord("¡"), ord("¬")+1)) 
        bs += list(range(ord("®"), ord("ÿ")+1))
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        return dict(zip(bs, [chr(n) for n in cs]))
    
    def _get_pairs(self, word):
        pairs = set()
        prev = word[0]
        for char in word[1:]:
            pairs.add((prev, char))
            prev = char
        return pairs
    
    def train(self, texts, num_merges=None):
        """训练BPE模型"""
        if num_merges is None:
            num_merges = self.vocab_size - 256 - len(self.special_tokens)
        
        # 统计词频（预分词后）
        word_freqs = defaultdict(int)
        for text in texts:
            for token in re.findall(self.pat, text):
                # 转换为字节表示
                token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
                word_freqs[' '.join(token)] += 1
        
        # 迭代合并
        merges = []
        vocab = set(self.byte_encoder.values())
        
        for i in range(num_merges):
            # 统计pair频率
            pairs = defaultdict(int)
            for word, freq in word_freqs.items():
                symbols = word.split()
                for j in range(len(symbols) - 1):
                    pairs[symbols[j], symbols[j+1]] += freq
            
            if not pairs:
                break
            
            # 找最频繁的pair
            best = max(pairs, key=pairs.get)
            
            # 合并
            new_word_freqs = {}
            bigram = ' '.join(best)
            replacement = ''.join(best)
            
            for word, freq in word_freqs.items():
                new_word = word.replace(bigram, replacement)
                new_word_freqs[new_word] = freq
            
            word_freqs = new_word_freqs
            merges.append(best)
            vocab.add(replacement)
            
            if (i + 1) % 1000 == 0:
                print(f"Merge {i+1}/{num_merges}: {best}")
        
        # 构建词表
        self.bpe_ranks = {merge: i for i, merge in enumerate(merges)}
        
        # encoder: token -> id
        self.encoder = {chr(i): i for i in range(256)}  # 基础字节
        for i, v in enumerate(sorted(vocab - set(chr(i) for i in range(256)))):
            self.encoder[v] = 256 + i
        
        # 添加特殊token
        for token, idx in self.special_tokens.items():
            self.encoder[token] = idx
        
        self.decoder = {v: k for k, v in self.encoder.items()}
        
        print(f"Vocabulary size: {len(self.encoder)}")
        return self
    
    def _bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        
        word = tuple(token)
        pairs = self._get_pairs(word)
        
        if not pairs:
            return token
        
        while True:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                
                if i < len(word) - 1 and word[i] == first and word[i+1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = self._get_pairs(word)
        
        result = ' '.join(word)
        self.cache[token] = result
        return result
    
    def encode(self, text):
        """编码文本"""
        tokens = []
        for match in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in match.encode('utf-8'))
            tokens.extend(self.encoder[t] for t in self._bpe(token).split(' '))
        return tokens
    
    def decode(self, token_ids):
        """解码token ids"""
        text = ''.join(self.decoder.get(t, '') for t in token_ids)
        return bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors='replace')
    
    def save(self, path):
        """保存tokenizer"""
        os.makedirs(path, exist_ok=True)
        
        with open(os.path.join(path, 'encoder.json'), 'w') as f:
            json.dump(self.encoder, f)
        
        with open(os.path.join(path, 'merges.txt'), 'w') as f:
            for pair in self.bpe_ranks:
                f.write(f"{pair[0]} {pair[1]}\n")
    
    def load(self, path):
        """加载tokenizer"""
        with open(os.path.join(path, 'encoder.json'), 'r') as f:
            self.encoder = json.load(f)
        self.decoder = {int(v): k for k, v in self.encoder.items()}
        
        with open(os.path.join(path, 'merges.txt'), 'r') as f:
            merges = [tuple(line.strip().split()) for line in f]
        self.bpe_ranks = {merge: i for i, merge in enumerate(merges)}
        
        return self


# 使用示例
if __name__ == "__main__":
    # 准备训练数据
    corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning models have revolutionized many fields.",
        "GPT is a transformer-based language model.",
    ] * 100  # 重复以增加数据量
    
    # 训练tokenizer
    tokenizer = CustomGPTTokenizer(vocab_size=500)
    tokenizer.train(corpus)
    
    # 测试
    test = "The machine learning model is powerful."
    encoded = tokenizer.encode(test)
    decoded = tokenizer.decode(encoded)
    
    print(f"Original: {test}")
    print(f"Encoded:  {encoded}")
    print(f"Decoded:  {decoded}")
```

## 总结

分词是GPT模型处理文本的基础环节，本文详细介绍了：

1. **BPE算法原理**：从数据压缩到NLP分词的演变
2. **Byte-level BPE**：GPT-2/3/4使用的分词方案，天然支持任意语言
3. **tiktoken库**：OpenAI官方高性能tokenizer
4. **词表构建**：训练数据准备、词表大小选择、特殊token处理
5. **SentencePiece**：Google的统一分词框架
6. **完整实现**：从零实现GPT风格的tokenizer

### 关键要点

- 子词分词平衡了词表大小和表达能力
- Byte-level BPE确保永远不会出现未知字符
- 词表大小影响模型效率和序列长度
- 特殊token对于对话、指令遵循至关重要

下一篇文章，我们将深入探讨GPT的预训练技术，包括语言建模目标、大规模数据处理和分布式训练策略。
