import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
import numpy as np

text_neg = []
text_pos = []


def loaddata(path):
    numwords = []
    file = open(path, encoding="utf-8")
    texts_projects = []
    label_projects = []
    for line in file.readlines():
        line = line.strip("\n")
        a = line[13:]
        texts_projects.append(a)

        counter = len(a.split(" "))
        numwords.append(counter)
        if line[4:12] == "negative":
            label_projects.append(0)
            text_neg.append(a)
        else:
            label_projects.append(1)
            text_pos.append(a)
    return texts_projects, label_projects, numwords


texts, labels, numwords = loaddata("data/Ant_processing.csv")

font = r"C:\Windows\Fonts\SIMLI.TTF"
word_wordcloud = WordCloud(font_path=font).generate(str(text_neg))
plt.figure(figsize=(8, 8))
plt.title("negative word", fontsize=10)
plt.imshow(word_wordcloud)
plt.show()
word_wordcloud = WordCloud(font_path=font).generate(str(text_pos))
plt.figure(figsize=(8, 8))
plt.title("positive word", fontsize=10)
plt.imshow(word_wordcloud)
plt.show()

myfont = fm.FontProperties(fname="C:\Windows\Fonts\SIMLI.TTF", size=13)
plt.hist(numwords, 150)
plt.xlabel("Length of comments", fontproperties=myfont)
plt.ylabel("Number of comments", fontproperties=myfont)
plt.axis([0, 100, 0, 70000])
plt.show()

max_len = 128
top_words = 10000

tokenizer = Tokenizer(num_words=top_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
dict = tokenizer.word_index
vocab = len(dict) + 1   #改
data = pad_sequences(sequences, maxlen=max_len)
labels = np.asarray(labels)

x_tr, x_va, y_tr, y_va = train_test_split(data, labels, train_size=0.9, random_state=3)
