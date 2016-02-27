import nltk
from nltk.tokenize import sent_tokenize

def orStepParser(sentence):
    words = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(words)

    chunkGram = r"""Chunk: {<POS|PRP\$><.*>*<NN.?>+<.*>*<CC><.*>*<NN.?>+}"""
    chunkParser = nltk.RegexpParser(chunkGram)
    chunked = chunkParser.parse(tagged)

    nounChunkGram = r"""Chunk: {<NN.?>+}"""
    nounChunkParser = nltk.RegexpParser(nounChunkGram)

    possessiveChunkGram = r"""Chunk: {<POS|PRP\$>}"""
    possessiveParser = nltk.RegexpParser(possessiveChunkGram)

    gerundActionGram = r"""Chunk: {<VBG><.*>*}(?:<POS|PRP\$><.*>*<CC>)"""
    gerundActionParser = nltk.RegexpParser(gerundActionGram)
    gerundActionChunked = gerundActionParser.parse(tagged)

    returnList = []

    main_entities = []
    def extractChunk(t):
            entities = []
            if hasattr(t, 'label') and t.label() == 'Chunk':
                entities.append(' '.join(c[0] for c in t.leaves()))
            else:
                for child in t:
                    if not isinstance(child, str):
                        entities.extend(extractChunk(child))

            return entities
    main_entities.extend(extractChunk(chunked))

    gerund_action_entities = []
    gerund_action_entities.extend(extractChunk(gerundActionChunked))
    
    noun_entities = []

    main_entity = main_entities[0]

    words = nltk.word_tokenize(main_entity)
    tagged_words = nltk.pos_tag(words)
    possessive_entities = set()
    for tagged_word in tagged_words:
        if tagged_word[1] in ('POS', 'PRP$'):
            possessive_entities.add(tagged_word[0])

    for possessive in possessive_entities:
        main_entity = main_entity.replace(possessive, 'your')
    new_statement = 'Are you {} {}'.format(gerund_action_entities[0], main_entity)
    chunked = nounChunkParser.parse(tagged_words)
    noun_entities.extend(extractChunk(chunked))

    nounKeyList = []
    for noun_entity in noun_entities:
        nounTuple = (noun_entity.replace(" ", "_"), noun_entity)
        nounKeyList.append(nounTuple)
    nounKeyTuple = tuple(nounKeyList)
        
    returnList.append(new_statement)
    returnList.append(nounKeyTuple)

    return returnList
