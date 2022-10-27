class TObject():
    
    def __init__(self,oid,label,pos):

        self.ID = oid
        self.label = label
        self.cur_position = pos
        self.starting_position = pos
        self.disappeard_duration = 0
