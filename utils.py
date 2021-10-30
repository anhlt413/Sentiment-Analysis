from nltk import word_tokenize

def pad_sents(sents, pad_token):
    """ Pad list of sentences according to the longest sentence in the batch.
    @param sents (list[list[str]]): list of sentences, where each sentence
                                    is represented as a list of words
    @param pad_token (str): padding token
    @returns sents_padded (list[list[str]]): list of sentences where sentences shorter
        than the max length sentence are padded out with the pad_token, such that
        each sentences in the batch now has equal length.
    """

    sents_padded = []
    max_length = 0
    for sent in sents:
        if len(sent) > max_length:
            max_length = len(sent)
    for sent in sents:
        sent_i = sent + [pad_token]*(max_length - len(sent))
        sents_padded.append(sent_i)

    return sents_padded



def read_corpus(file_path):
    """ Read file, where each sentence is dilineated by a `\n`.
    @param file_path (str): path to file containing corpus
    """
    data = []
    for line in open(file_path):
        sent = word_tokenize(line.strip().lower())
        data.append(sent)

    return data