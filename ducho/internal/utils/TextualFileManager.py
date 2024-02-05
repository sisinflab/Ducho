import pandas as pd


class TextualFileManager:
    def __init__(self, columns):
        """
        It manages the input textual file and its contents. Note that it is used also to build the names of the outputs
        files of the textual extraction
        Args:
            columns: tuple with columns' names
        """
        self._internal_list = None
        self._type_of_extraction = None
        self._file_path = None
        self._id_column = None
        self._text_column = None
        self._columns = columns

    def set_type_of_extraction(self, type_of_extraction):
        """
        Set type  of extraction which is the source of the input: from user interaction or from items. Here it is needed
        to across correctly the file and for build the name of the output file at the end of the extraction
        Args:
            type_of_extraction (str): The type of extraction, either 'interactions' or 'items'.

        Returns:
            None
        """
        self._type_of_extraction = type_of_extraction

    def set_file_path(self, file_path):
        """
        Set the absolute path of the textual input file that will be opened later.

        Args:
            file_path (str): The absolute path of the input file as a string.

        Returns:
            None
        """
        self._file_path = file_path

    def build_path_from_id(self, id_):
        """
        Build the name of the output file for a single processed sentence.

        This method constructs the name of the output file for a single processed sentence.
        Later, this name will be used to build the complete path of the single output file.

        Args:
            id_ (Any): The row ID. This is used only to build the name.

        Returns:
            str: The output file name. This is not the complete path or the complete name of the file
            (it misses the extension).
        """
        if self._type_of_extraction == 'interactions':
            return f'{id_[0]}__{id_[1]}'
        elif self._type_of_extraction == 'items':
            return str(id_)

    def initiate_element_list_and_get_len(self):
        """
        Read the file, instantiate the internal list of its contents, and return the number of sentences to be elaborated.

        Returns:
            Tuple[int, List[Any]]: A tuple containing the length of the object to be elaborated and a list of IDs.
        """
        df = pd.read_csv(self._file_path, sep='\t')
        self._id_column = self._columns[0]
        self._text_column = self._columns[1]
        self._internal_list = df
        ids_list = df[self._id_column].tolist() if type(self._id_column) != list else \
            list(zip(df[self._id_column[0]].tolist(), df[self._id_column[1]].tolist()))
        return len(self._internal_list), ids_list

    def get_item_from_id(self, idx):
        """
        Retrieve the sentence to be elaborated for a specific row of the file.

        If the origin of elaboration is from interactions, it searches the sentence in the 'comment' column.
        Otherwise, if the origin is from item description, it searches the sentence in the 'description' column.

        Args:
            idx (Any): The row from which to retrieve the sentence.

        Returns:
            str: The sentence to be elaborated. Preprocessing may be needed.
        """
        if self._type_of_extraction == 'items':
            return self._internal_list[self._internal_list[self._id_column] == idx].to_dict('records')[0][self._text_column]
        else:
            return self._internal_list[(self._internal_list[self._id_column[0]] == idx[0]) &
                                       (self._internal_list[self._id_column[1]] == idx[1])].to_dict('records')[0][self._text_column]
