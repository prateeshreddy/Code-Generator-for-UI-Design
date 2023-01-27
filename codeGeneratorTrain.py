# Imports
import os
import sys
import shutil
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, \
                         RepeatVector, LSTM, concatenate, \
                         Conv2D, MaxPooling2D, Flatten, Bidirectional
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import *
from keras.models import model_from_json
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

# Constants
CONTEXT_LENGTH = 48
IMAGE_SIZE = 256
BATCH_SIZE = 64
EPOCHS = 200
STEPS_PER_EPOCH = 72000
START_TOKEN = "<START>"
END_TOKEN = "<END>"
PLACEHOLDER = " "
SEPARATOR = '->'

# Paths
input_path = "../datasets/web/training_set"
output_path = "../datasets/web/training_features"


class Utils:
    @staticmethod
    def sparsify(label_vector, output_size):
        sparse_vector = []

        for label in label_vector:
            sparse_label = np.zeros(output_size)
            sparse_label[label] = 1
            sparse_vector.append(sparse_label)
        return np.array(sparse_vector)

    @staticmethod
    def get_preprocessed_img(img_path, image_size):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (image_size, image_size))
        img = img.astype('float32')
        img /= 255
        return img

    @staticmethod
    def show(image):
        cv2.namedWindow("view", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("view", image)
        cv2.waitKey(0)
        cv2.destroyWindow("view")

for f in os.listdir(input_path):
    if f.find(".png") != -1:
        img = Utils.get_preprocessed_img("{}/{}".format(input_path, f), IMAGE_SIZE)
        file_name = f[:f.find(".png")]

        np.savez_compressed("{}/{}".format(output_path, file_name), features=img)
        retrieve = np.load("{}/{}.npz".format(output_path, file_name))["features"]

        assert np.array_equal(img, retrieve)

        shutil.copyfile("{}/{}.gui".format(input_path, file_name), "{}/{}.gui".format(output_path, file_name))

print("Numpy arrays created.")

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

class Vocabulary:
    def __init__(self):
        self.binary_vocabulary = {}
        self.vocabulary = {}
        self.token_lookup = {}
        self.size = 0

        self.append(START_TOKEN)
        self.append(END_TOKEN)
        self.append(PLACEHOLDER)

    def append(self, token):
        if token not in self.vocabulary:
            self.vocabulary[token] = self.size
            self.token_lookup[self.size] = token
            self.size += 1

    def create_binary_representation(self):
        if sys.version_info >= (3,):
            items = self.vocabulary.items()
        else:
            items = self.vocabulary.iteritems()
        for key, value in items:
            binary = np.zeros(self.size)
            binary[value] = 1
            self.binary_vocabulary[key] = binary

    def get_serialized_binary_representation(self):
        if len(self.binary_vocabulary) == 0:
            self.create_binary_representation()

        string = ""
        if sys.version_info >= (3,):
            items = self.binary_vocabulary.items()
        else:
            items = self.binary_vocabulary.iteritems()
        for key, value in items:
            array_as_string = np.array2string(value, separator=',', max_line_width=self.size * self.size)
            string += "{}{}{}\n".format(key, SEPARATOR, array_as_string[1:len(array_as_string) - 1])
        return string

    def save(self, path):
        output_file_name = "{}/words.vocab".format(path)
        output_file = open(output_file_name, 'w')
        output_file.write(self.get_serialized_binary_representation())
        output_file.close()

    def retrieve(self, path):
        input_file = open("{}/words.vocab".format(path), 'r')
        buffer = ""
        for line in input_file:
            try:
                separator_position = len(buffer) + line.index(SEPARATOR)
                buffer += line
                key = buffer[:separator_position]
                value = buffer[separator_position + len(SEPARATOR):]
                value = np.fromstring(value, sep=',')

                self.binary_vocabulary[key] = value
                self.vocabulary[key] = np.where(value == 1)[0][0]
                self.token_lookup[np.where(value == 1)[0][0]] = key

                buffer = ""
            except ValueError:
                buffer += line
        input_file.close()
        self.size = len(self.vocabulary)

class Dataset:
    def __init__(self):
        self.input_shape = None
        self.output_size = None

        self.ids = []
        self.input_images = []
        self.partial_sequences = []
        self.next_words = []

        self.voc = Vocabulary()
        self.size = 0

    @staticmethod
    def load_paths_only(path):
        print("Parsing data...")
        gui_paths = []
        img_paths = []
        for f in os.listdir(path):
            if f.find(".gui") != -1:
                path_gui = "{}/{}".format(path, f)
                gui_paths.append(path_gui)
                file_name = f[:f.find(".gui")]

                if os.path.isfile("{}/{}.png".format(path, file_name)):
                    path_img = "{}/{}.png".format(path, file_name)
                    img_paths.append(path_img)
                elif os.path.isfile("{}/{}.npz".format(path, file_name)):
                    path_img = "{}/{}.npz".format(path, file_name)
                    img_paths.append(path_img)

        assert len(gui_paths) == len(img_paths)
        return gui_paths, img_paths

    def load(self, path, generate_binary_sequences=False):
        print("Loading data...")
        for f in os.listdir(path):
            if f.find(".gui") != -1:
                gui = open("{}/{}".format(path, f), 'r')
                file_name = f[:f.find(".gui")]

                if os.path.isfile("{}/{}.png".format(path, file_name)):
                    img = Utils.get_preprocessed_img("{}/{}.png".format(path, file_name), IMAGE_SIZE)
                    self.append(file_name, gui, img)
                elif os.path.isfile("{}/{}.npz".format(path, file_name)):
                    img = np.load("{}/{}.npz".format(path, file_name))["features"]
                    self.append(file_name, gui, img)

        print("Generating sparse vectors...")
        self.voc.create_binary_representation()
        self.next_words = self.sparsify_labels(self.next_words, self.voc)
        if generate_binary_sequences:
            self.partial_sequences = self.binarize(self.partial_sequences, self.voc)
        else:
            self.partial_sequences = self.indexify(self.partial_sequences, self.voc)

        self.size = len(self.ids)
        print ("Size == ", self.size)
        assert self.size == len(self.input_images) == len(self.partial_sequences) == len(self.next_words)
        assert self.voc.size == len(self.voc.vocabulary)

        print("Dataset size: {}".format(self.size))
        print("Vocabulary size: {}".format(self.voc.size))

        self.input_shape = self.input_images[0].shape
        self.output_size = self.voc.size

        print("Input shape: {}".format(self.input_shape))
        print("Output size: {}".format(self.output_size))

    def convert_arrays(self):
        print("Convert arrays...")
        self.input_images = np.array(self.input_images)
        self.partial_sequences = np.array(self.partial_sequences)
        self.next_words = np.array(self.next_words)

    def append(self, sample_id, gui, img, to_show=False):
        if to_show:
            pic = img * 255
            pic = np.array(pic, dtype=np.uint8)
            Utils.show(pic)

        token_sequence = [START_TOKEN]
        for line in gui:
            line = line.replace(",", " ,").replace("\n", " \n")
            tokens = line.split(" ")
            for token in tokens:
                self.voc.append(token)
                token_sequence.append(token)
        token_sequence.append(END_TOKEN)

        suffix = [PLACEHOLDER] * CONTEXT_LENGTH

        a = np.concatenate([suffix, token_sequence])
        for j in range(0, len(a) - CONTEXT_LENGTH):
            context = a[j:j + CONTEXT_LENGTH]
            label = a[j + CONTEXT_LENGTH]

            self.ids.append(sample_id)
            self.input_images.append(img)
            self.partial_sequences.append(context)
            self.next_words.append(label)

    @staticmethod
    def indexify(partial_sequences, voc):
        temp = []
        for sequence in partial_sequences:
            sparse_vectors_sequence = []
            for token in sequence:
                sparse_vectors_sequence.append(voc.vocabulary[token])
            temp.append(np.array(sparse_vectors_sequence))

        return temp

    @staticmethod
    def binarize(partial_sequences, voc):
        temp = []
        for sequence in partial_sequences:
            sparse_vectors_sequence = []
            for token in sequence:
                sparse_vectors_sequence.append(voc.binary_vocabulary[token])
            temp.append(np.array(sparse_vectors_sequence))

        return temp

    @staticmethod
    def sparsify_labels(next_words, voc):
        temp = []
        for label in next_words:
            temp.append(voc.binary_vocabulary[label])

        return temp

    def save_metadata(self, path):
        np.save("{}/meta_dataset".format(path), np.array([self.input_shape, self.output_size, self.size], dtype=object))

class Generator:
    @staticmethod
    def data_generator(voc, gui_paths, img_paths, batch_size, generate_binary_sequences=False, verbose=False, loop_only_one=False):
        assert len(gui_paths) == len(img_paths)
        voc.create_binary_representation()

        while 1:
            batch_input_images = []
            batch_partial_sequences = []
            batch_next_words = []
            sample_in_batch_counter = 0

            for i in range(0, len(gui_paths)):
                if img_paths[i].find(".png") != -1:
                    img = Utils.get_preprocessed_img(img_paths[i], IMAGE_SIZE)
                else:
                    img = np.load(img_paths[i])["features"]
                gui = open(gui_paths[i], 'r')

                token_sequence = [START_TOKEN]
                for line in gui:
                    line = line.replace(",", " ,").replace("\n", " \n")
                    tokens = line.split(" ")
                    for token in tokens:
                        voc.append(token)
                        token_sequence.append(token)
                token_sequence.append(END_TOKEN)

                suffix = [PLACEHOLDER] * CONTEXT_LENGTH

                a = np.concatenate([suffix, token_sequence])
                for j in range(0, len(a) - CONTEXT_LENGTH):
                    context = a[j:j + CONTEXT_LENGTH]
                    label = a[j + CONTEXT_LENGTH]

                    batch_input_images.append(img)
                    batch_partial_sequences.append(context)
                    batch_next_words.append(label)
                    sample_in_batch_counter += 1

                    if sample_in_batch_counter == batch_size or (loop_only_one and i == len(gui_paths) - 1):
                        if verbose:
                            print("Generating sparse vectors...")
                        batch_next_words = Dataset.sparsify_labels(batch_next_words, voc)
                        if generate_binary_sequences:
                            batch_partial_sequences = Dataset.binarize(batch_partial_sequences, voc)
                        else:
                            batch_partial_sequences = Dataset.indexify(batch_partial_sequences, voc)

                        if verbose:
                            print("Convert arrays...")
                        batch_input_images = np.array(batch_input_images)
                        batch_partial_sequences = np.array(batch_partial_sequences)
                        batch_next_words = np.array(batch_next_words)

                        if verbose:
                            print("Yield batch")
                        yield ([batch_input_images, batch_partial_sequences], batch_next_words)

                        batch_input_images = []
                        batch_partial_sequences = []
                        batch_next_words = []
                        sample_in_batch_counter = 0

class AModel:
    def __init__(self, input_shape, output_size, output_path):
        self.model = None
        self.input_shape = input_shape
        self.output_size = output_size
        self.output_path = output_path
        self.name = ""

    def save(self):
        model_json = self.model.to_json()
        with open("{}/{}.json".format(self.output_path, self.name), "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights("{}/{}.h5".format(self.output_path, self.name))

    def load(self, name=""):
        output_name = self.name if name == "" else name
        with open("{}/{}.json".format(self.output_path, output_name), "r") as json_file:
            loaded_model_json = json_file.read()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights("{}/{}.h5".format(self.output_path, output_name))

class pix2code(AModel):
    def __init__(self, input_shape, output_size, output_path):
        AModel.__init__(self, input_shape, output_size, output_path)
        self.name = "pix2code"

        image_model = Sequential()
        image_model.add(Conv2D(32, (3, 3), padding='valid', activation='relu', input_shape=input_shape))
        image_model.add(Conv2D(32, (3, 3), padding='valid', activation='relu'))
        image_model.add(MaxPooling2D(pool_size=(2, 2)))
        image_model.add(Dropout(0.25))

        image_model.add(Conv2D(64, (3, 3), padding='valid', activation='relu'))
        image_model.add(Conv2D(64, (3, 3), padding='valid', activation='relu'))
        image_model.add(MaxPooling2D(pool_size=(2, 2)))
        image_model.add(Dropout(0.25))

        image_model.add(Conv2D(128, (3, 3), padding='valid', activation='relu'))
        image_model.add(Conv2D(128, (3, 3), padding='valid', activation='relu'))
        image_model.add(MaxPooling2D(pool_size=(2, 2)))
        image_model.add(Dropout(0.25))

        image_model.add(Flatten())
        image_model.add(Dense(1024, activation='relu'))
        image_model.add(Dropout(0.3))
        image_model.add(Dense(1024, activation='relu'))
        image_model.add(Dropout(0.3))

        image_model.add(RepeatVector(CONTEXT_LENGTH))

        visual_input = Input(shape=input_shape)
        encoded_image = image_model(visual_input)

        language_model = Sequential()
        language_model.add(LSTM(128, return_sequences=True, input_shape=(CONTEXT_LENGTH, output_size)))
        language_model.add(LSTM(128, return_sequences=True))

        textual_input = Input(shape=(CONTEXT_LENGTH, output_size))
        encoded_text = language_model(textual_input)

        decoder = concatenate([encoded_image, encoded_text])

        decoder = Bidirectional(LSTM(512, return_sequences=True))(decoder)
        decoder = Bidirectional(LSTM(512, return_sequences=False))(decoder)
        decoder = Dense(output_size, activation='softmax')(decoder)

        self.model = Model(inputs=[visual_input, textual_input], outputs=decoder)

        optimizer = RMSprop(learning_rate=0.0001, clipvalue=1.0)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    def fit(self, images, partial_captions, next_words):
        self.model.fit([images, partial_captions], next_words, shuffle=False, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
        self.save()

    def fit_generator(self, generator, steps_per_epoch):
        self.model.fit_generator(generator, steps_per_epoch=steps_per_epoch, epochs=EPOCHS, verbose=1)
        self.save()

    def predict(self, image, partial_caption):
        return self.model.predict([image, partial_caption], verbose=0)[0]

    def predict_batch(self, images, partial_captions):
        return self.model.predict([images, partial_captions], verbose=1)


# Paths
input_path = "../datasets/web/training_features"
output_path = "../bin"
use_generator = 1

def run(input_path, output_path, is_memory_intensive=False, pretrained_model=None):
    np.random.seed(1234)
    dataset = Dataset()
    dataset.load(input_path, generate_binary_sequences=True)
    dataset.save_metadata(output_path)
    dataset.voc.save(output_path)
    if not is_memory_intensive:
        dataset.convert_arrays()
        input_shape = dataset.input_shape
        output_size = dataset.output_size
    else:
        gui_paths, img_paths = Dataset.load_paths_only(input_path)
        input_shape = dataset.input_shape
        output_size = dataset.output_size
        steps_per_epoch = dataset.size / BATCH_SIZE
        voc = Vocabulary()
        voc.retrieve(output_path)
        generator = Generator.data_generator(voc, gui_paths, img_paths, batch_size=BATCH_SIZE, generate_binary_sequences=True)
    model = pix2code(input_shape, output_size, output_path)

    if pretrained_model is not None:
        model.model.load_weights(pretrained_model)

    if not is_memory_intensive:
        model.fit(dataset.input_images, dataset.partial_sequences, dataset.next_words)
    else:
        model.fit_generator(generator, steps_per_epoch=steps_per_epoch)

run(input_path, output_path, is_memory_intensive=use_generator)

print("Training Complete")
print("Starting validation")

from os.path import basename
class Node:
    def __init__(self, key, value, data=None):
        self.key = key
        self.value = value
        self.data = data
        self.parent = None
        self.root = None
        self.children = []
        self.level = 0

    def add_children(self, children, beam_width):
        for child in children:
            child.level = self.level + 1
            child.value = child.value * self.value

        nodes = sorted(children, key=lambda node: node.value, reverse=True)
        nodes = nodes[:beam_width]

        for node in nodes:
            self.children.append(node)
            node.parent = self

        if self.parent is None:
            self.root = self
        else:
            self.root = self.parent.root
        child.root = self.root

    def remove_child(self, child):
        self.children.remove(child)

    def max_child(self):
        if len(self.children) == 0:
            return self

        max_childs = []
        for child in self.children:
            max_childs.append(child.max_child())

        nodes = sorted(max_childs, key=lambda child: child.value, reverse=True)
        return nodes[0]

    def show(self, depth=0):
        print(" " * depth, self.key, self.value, self.level)
        for child in self.children:
            child.show(depth + 2)

class BeamSearch:
    def __init__(self, beam_width=1):
        self.beam_width = beam_width

        self.root = None
        self.clear()

    def search(self):
        result = self.root.max_child()

        self.clear()
        return self.retrieve_path(result)

    def add_nodes(self, parent, children):
        parent.add_children(children, self.beam_width)

    def is_valid(self):
        leaves = self.get_leaves()
        level = leaves[0].level
        counter = 0
        for leaf in leaves:
            if leaf.level == level:
                counter += 1
            else:
                break

        if counter == len(leaves):
            return True

        return False

    def get_leaves(self):
        leaves = []
        self.search_leaves(self.root, leaves)
        return leaves

    def search_leaves(self, node, leaves):
        for child in node.children:
            if len(child.children) == 0:
                leaves.append(child)
            else:
                self.search_leaves(child, leaves)

    def prune_leaves(self):
        leaves = self.get_leaves()

        nodes = sorted(leaves, key=lambda leaf: leaf.value, reverse=True)
        nodes = nodes[self.beam_width:]

        for node in nodes:
            node.parent.remove_child(node)

        while not self.is_valid():
            leaves = self.get_leaves()
            max_level = 0
            for leaf in leaves:
                if leaf.level > max_level:
                    max_level = leaf.level

            for leaf in leaves:
                if leaf.level < max_level:
                    leaf.parent.remove_child(leaf)

    def clear(self):
        self.root = None
        self.root = Node("root", 1.0, None)

    def retrieve_path(self, end):
        path = [end.key]
        data = [end.data]
        while end.parent is not None:
            end = end.parent
            path.append(end.key)
            data.append(end.data)

        result_path = []
        result_data = []
        for i in range(len(path) - 2, -1, -1):
            result_path.append(path[i])
            result_data.append(data[i])
        return result_path, result_data

class Sampler:
    def __init__(self, voc_path, input_shape, output_size, context_length):
        self.voc = Vocabulary()
        self.voc.retrieve(voc_path)

        self.input_shape = input_shape
        self.output_size = output_size

        print("Vocabulary size: {}".format(self.voc.size))
        print("Input shape: {}".format(self.input_shape))
        print("Output size: {}".format(self.output_size))

        self.context_length = context_length

    def predict_greedy(self, model, input_img, require_sparse_label=True, sequence_length=150, verbose=False):
        current_context = [self.voc.vocabulary[PLACEHOLDER]] * (self.context_length - 1)
        current_context.append(self.voc.vocabulary[START_TOKEN])
        if require_sparse_label:
            current_context = Utils.sparsify(current_context, self.output_size)

        predictions = START_TOKEN
        out_probas = []

        for i in range(0, sequence_length):
            if verbose:
                print("predicting {}/{}...".format(i, sequence_length))

            probas = model.predict(input_img, np.array([current_context]))
            prediction = np.argmax(probas)
            out_probas.append(probas)

            new_context = []
            for j in range(1, self.context_length):
                new_context.append(current_context[j])

            if require_sparse_label:
                sparse_label = np.zeros(self.output_size)
                sparse_label[prediction] = 1
                new_context.append(sparse_label)
            else:
                new_context.append(prediction)

            current_context = new_context

            predictions += self.voc.token_lookup[prediction]

            if self.voc.token_lookup[prediction] == END_TOKEN:
                break

        return predictions, out_probas

    def recursive_beam_search(self, model, input_img, current_context, beam, current_node, sequence_length):
        probas = model.predict(input_img, np.array([current_context]))

        predictions = []
        for i in range(0, len(probas)):
            predictions.append((i, probas[i], probas))

        nodes = []
        for i in range(0, len(predictions)):
            prediction = predictions[i][0]
            score = predictions[i][1]
            output_probas = predictions[i][2]
            nodes.append(Node(prediction, score, output_probas))

        beam.add_nodes(current_node, nodes)

        if beam.is_valid():
            beam.prune_leaves()
            if sequence_length == 1 or self.voc.token_lookup[beam.root.max_child().key] == END_TOKEN:
                return

            for node in beam.get_leaves():
                prediction = node.key

                new_context = []
                for j in range(1, self.context_length):
                    new_context.append(current_context[j])
                sparse_label = np.zeros(self.output_size)
                sparse_label[prediction] = 1
                new_context.append(sparse_label)

                self.recursive_beam_search(model, input_img, new_context, beam, node, sequence_length - 1)

    def predict_beam_search(self, model, input_img, beam_width=3, require_sparse_label=True, sequence_length=150):
        predictions = START_TOKEN
        out_probas = []

        current_context = [self.voc.vocabulary[PLACEHOLDER]] * (self.context_length - 1)
        current_context.append(self.voc.vocabulary[START_TOKEN])
        if require_sparse_label:
            current_context = Utils.sparsify(current_context, self.output_size)

        beam = BeamSearch(beam_width=beam_width)

        self.recursive_beam_search(model, input_img, current_context, beam, beam.root, sequence_length)

        predicted_sequence, probas_sequence = beam.search()

        for k in range(0, len(predicted_sequence)):
            prediction = predicted_sequence[k]
            probas = probas_sequence[k]
            out_probas.append(probas)

            predictions += self.voc.token_lookup[prediction]

        return predictions, out_probas

# Paths
trained_weights_path = "../bin"
trained_model_name = "pix2code"
input_path = "../datasets/web/eval_set/0CE73E18-575A-4A70-9E40-F000B250344F.png"
output_path = "../code"
# search_method = "greedy"
search_method = 3 # for beam search with beamlength of 3

class Vocabulary1(object):
    def __init__ (self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        
    def add_word (self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
    
    def __call__ (self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]
    
    def __len__ (self):
        return len(self.word2idx)

def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def build_vocab (vocab_file_path):
    vocab = Vocabulary1()
    words_raw = load_doc(vocab_file_path)
    words = set(words_raw.split(' '))
    for i, word in enumerate(words):
        vocab.add_word(word)
    vocab.add_word(' ')
    vocab.add_word('<unk>')
    print('Created vocabulary of ' + str(len(vocab)) + ' items from ' + vocab_file_path)
    return vocab

def transform_idx_to_words (input):
    vocab_path = './bootstrap.vocab'
    vocab = build_vocab(vocab_path)
    vocab_size = len(vocab)
    
    sampled_caption = []
    result.replace(START_TOKEN, "\n"+START_TOKEN+"\n")
    result.replace(END_TOKEN, "\n"+END_TOKEN+"\n")
    
    print (vocab.idx2word)
    input_list = input.split("\n")
    for idx in input_list:
        word = vocab.idx2word[idx]
        sampled_caption.append(word)
        if word == '<END>':
            break
    output = ' '.join(sampled_caption[1:-1])
    output = output.replace(' ,', ',')
    return output.split(' ')

meta_dataset = np.load("{}/meta_dataset.npy".format(trained_weights_path), allow_pickle = True)
input_shape = meta_dataset[0]
output_size = meta_dataset[1]

model = pix2code(input_shape, output_size, trained_weights_path)
model.load(trained_model_name)

sampler = Sampler(trained_weights_path, input_shape, output_size, CONTEXT_LENGTH)

file_name = basename(input_path)[:basename(input_path).find(".")]
gui_file = input_path[:-3]+'gui'
evaluation_img = Utils.get_preprocessed_img(input_path, IMAGE_SIZE)
evaluation_caption = open("{}".format(gui_file), 'r')

if search_method == "greedy":
    result, _ = sampler.predict_greedy(model, np.array([evaluation_img]))
    print("Result greedy: {}".format(result))
else:
    beam_width = int(search_method)
    print("Search with beam width: {}".format(beam_width))
    result, _ = sampler.predict_beam_search(model, np.array([evaluation_img]), beam_width=beam_width)
    print("Result beam: {}".format(result))

res = result.replace(START_TOKEN, "").replace(END_TOKEN, "").replace(",", " ,").replace("\n", " \n")
predicted = res.split(" ")

actual = []
for line in evaluation_caption:
    l = line.replace(",", " ,").replace("\n", " \n")
    tokens = l.split(" ")
    actual.extend(tokens)

bleu = sentence_bleu([actual], actual)
bleu = corpus_bleu([actual], [predicted], smoothing_function=SmoothingFunction().method4)
print ("*BLEU Score is: *", bleu)

with open("{}/{}.gui".format(output_path, file_name), 'w') as out_f:
    out_f.write(result.replace(START_TOKEN, "").replace(END_TOKEN, ""))

print("Validation Ended")
print("Starting compiling")

# Compiler
input_file = "../code/0CE73E18-575A-4A70-9E40-F000B250344F.gui"

from os.path import basename
import string
import random


class Utils:
    @staticmethod
    def get_random_text(length_text=10, space_number=1, with_upper_case=True):
        results = []
        while len(results) < length_text:
            char = random.choice(string.ascii_letters[:26])
            results.append(char)
        if with_upper_case:
            results[0] = results[0].upper()

        current_spaces = []
        while len(current_spaces) < space_number:
            space_pos = random.randint(2, length_text - 3)
            if space_pos in current_spaces:
                break
            results[space_pos] = " "
            if with_upper_case:
                results[space_pos + 1] = results[space_pos - 1].upper()

            current_spaces.append(space_pos)

        return ''.join(results)

    @staticmethod
    def get_ios_id(length=10):
        results = []

        while len(results) < length:
            char = random.choice(string.digits + string.ascii_letters)
            results.append(char)

        results[3] = "-"
        results[6] = "-"

        return ''.join(results)

    @staticmethod
    def get_android_id(length=10):
        results = []

        while len(results) < length:
            char = random.choice(string.ascii_letters)
            results.append(char)

        return ''.join(results)

import json

class Node:
    def __init__(self, key, parent_node, content_holder):
        self.key = key
        self.parent = parent_node
        self.children = []
        self.content_holder = content_holder

    def add_child(self, child):
        self.children.append(child)

    def show(self):
        print(self.key)
        for child in self.children:
            child.show()

    def render(self, mapping, rendering_function=None):
        content = ""
        for child in self.children:
            content += child.render(mapping, rendering_function)

        value = mapping[self.key]
        if rendering_function is not None:
            value = rendering_function(self.key, value)

        if len(self.children) != 0:
            value = value.replace(self.content_holder, content)

        return value



class Compiler:
    def __init__(self, dsl_mapping_file_path):
        with open(dsl_mapping_file_path) as data_file:
            self.dsl_mapping = json.load(data_file)

        self.opening_tag = self.dsl_mapping["opening-tag"]
        self.closing_tag = self.dsl_mapping["closing-tag"]
        self.content_holder = self.opening_tag + self.closing_tag

        self.root = Node("body", None, self.content_holder)

    def compile(self, input_file_path, output_file_path, rendering_function=None):
        dsl_file = open(input_file_path)
        current_parent = self.root

        for token in dsl_file:
            token = token.replace(" ", "").replace("\n", "")
            print (token)
            if token.find(self.opening_tag) != -1:
                token = token.replace(self.opening_tag, "")

                element = Node(token, current_parent, self.content_holder)
                print ("*****", current_parent)
                current_parent.add_child(element)
                current_parent = element
            elif token.find(self.closing_tag) != -1:
                current_parent = current_parent.parent
            else:
                tokens = token.split(",")
                for t in tokens:
                    element = Node(t, current_parent, self.content_holder)
                    current_parent.add_child(element)

        output_html = self.root.render(self.dsl_mapping, rendering_function=rendering_function)
        with open(output_file_path, 'w') as output_file:
            output_file.write(output_html)


FILL_WITH_RANDOM_TEXT = True
TEXT_PLACE_HOLDER = "[]"

dsl_path = "web-dsl-mapping.json"
compiler = Compiler(dsl_path)


def render_content_with_text(key, value):
    if FILL_WITH_RANDOM_TEXT:
        if key.find("btn") != -1:
            value = value.replace(TEXT_PLACE_HOLDER, Utils.get_random_text())
        elif key.find("title") != -1:
            value = value.replace(TEXT_PLACE_HOLDER, Utils.get_random_text(length_text=5, space_number=0))
        elif key.find("text") != -1:
            value = value.replace(TEXT_PLACE_HOLDER,
                                  Utils.get_random_text(length_text=56, space_number=7, with_upper_case=False))
    return value

file_uid = basename(input_file)[:basename(input_file).find(".")]
path = input_file[:input_file.find(file_uid)]

input_file_path = "{}{}.gui".format(path, file_uid)
output_file_path = "{}{}.html".format(path, file_uid)
print (input_file_path, output_file_path)

compiler.compile(input_file_path, output_file_path, rendering_function=render_content_with_text)
