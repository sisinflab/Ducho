import pandas as pd


class TextualFileManager:
    def __init__(self, column):
        """
        It manages the input textual file and its contents. Note that it is used also to build the names of the outputs
        files of the textual extraction
        """
        self._internal_list = None
        self._type_of_extraction = None
        self._file_path = None
        self._id_column = None
        self._last_column = None
        self._column = column

    def set_type_of_extraction(self, type_of_extraction):
        """
        Set type  of extraction which is the source of the input: from user interaction or from items. Here it is needed
        to across correctly the file and for build the name of the output file at the end of the extraction
        Args:
            type_of_extraction: 'interactions' or 'items'
        """
        self._type_of_extraction = type_of_extraction

    def set_file_path(self, file_path):
        """
        It sets the absolute path of the textual input file that later will be open.

        Args:
            file_path: absolute path as a string
        """
        self._file_path = file_path

    def build_path_from_id(self, id_):
        """
        It builds the name of the output file of a single sentence processed. This will later be used to build the
        complete path of the single output file

        Args:
            id_: the row id as an int, here it used only to build the name
        Returns:
            the output name file as a string. It is not the complete path, nor the complete name of the file (it
            misses the extension)
        """
        if self._type_of_extraction == 'interactions':
            return f'{id_[0]}__{id_[1]}'
        elif self._type_of_extraction == 'items':
            return str(id_)

    def initiate_element_list_and_get_len(self):
        """
        Reads the file, instantiate the internal list of what it contains and returns the len of sentences to elaborate

        Returns:
            len of object to elaborate
        """
        df = pd.read_csv(self._file_path, sep='\t')
        num_columns = len(df.columns)
        self._id_column = df.columns[0] if num_columns == 2 else (df.columns[0], df.columns[1])
        self._last_column = df.columns[-1]
        self._internal_list = df
        ids_list = df[df.columns[0]].tolist() if num_columns == 2 else \
            list(zip(df[df.columns[0]].tolist(), df[df.columns[1]].tolist()))
        return len(self._internal_list), ids_list

    def get_item_from_id(self, idx):
        """
        It gives the sentence to elaborate for a specific row of the file. If the origin of elaboration is from
        interactions, it searches the sentence in the 'comment' column, otherwise if the origin is from item description
        it searches the sentence in the 'description' column

        Args:
            idx: the row from which retrieve the sentence.
        Returns:
            the sentence as a string, preprocessing is needed
        """
        if self._type_of_extraction == 'items':
            row = self._internal_list[self._internal_list[self._id_column] == idx].to_dict('records')[0]
        else:
            row = self._internal_list[(self._internal_list[self._id_column[0]] == idx[0]) &
                                      (self._internal_list[self._id_column[1]] == idx[1])].to_dict('records')[0]

        if self._type_of_extraction == 'interactions':
            return row[self._column if self._column else self._last_column]
        elif self._type_of_extraction == 'items':
            return row[self._column if self._column else self._last_column]