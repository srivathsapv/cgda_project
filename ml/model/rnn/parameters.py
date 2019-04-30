class Parameters:
    def __init__(self, classifier_type):

        self.vocab_size = 4
        self.embedding_dim = 50
        self.hidden_dim = 50
        self.batch_size=1
        if classifier_type=='phylum':
            self.epochs=50
            self.label_size=3
        elif classifier_type=='class':
            self.epochs=70
            self.label_size=5
        else:
            self.epochs=90
            self.label_size=19
