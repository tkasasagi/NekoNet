class CharacterSet:

    def __init__(self):       
        import pandas as pd       
        charlist = pd.read_csv('./resource/charlist400.csv')       
        charlist = list(charlist['Unicode'])        
        char2ind = {x: i for i, x in enumerate(charlist)}    
        ind2char = {}

        for char in char2ind:
            ind2char[char2ind[char]] = char

        self.ind2char = ind2char        
        self.char2ind = char2ind
        self.num_characters = len(char2ind)
