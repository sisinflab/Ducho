import pandas
import csv


class TextualFileManager:
    def __init__(self, ):
        """
        It manages the input textual file and its contents. Note that it is used also to build the names of the outputs
        files of the textual extraction
        """
        self._internal_list = None
        self._type_of_extraction = None
        self._file_path = None
        return

    def set_type_of_extraction(self, type_of_extraction):
        """
        Set type  of extraction which is the source of the input: from user interaction or from items. Here it is needed
        to across correctly the file and for build the name of the output file at the end of the extraction

        :param type_of_extraction: 'interactions' or 'items'
        """
        self._type_of_extraction = type_of_extraction

    def set_file_path(self, file_path):
        """
        It sets the absolute path of the textual input file that later will be open.
        :param file_path: absolute path as a string
        """
        self._file_path = file_path

    def build_path_from_id(self, id_):
        """
        It builds the name of the output file of a single sentence processed. This will later be used to build the
        complete path of the single output file
        :param id_: the row id as an int, here it used only to build the name
        :return: the output name file as a string. It is not the complete path, nor the complete name of the file (it
        misses the extension)
        """
        if self._type_of_extraction == 'interactions':
            user = self._file_path[id_]['user']
            return user+'_'+str(id_)
        elif self._type_of_extraction == 'items':
            return str(id_)

    def initiate_element_list_and_get_len(self):
        """
        Reads the file, instantiate the internal list of what it contains and returns the len of sentences to elaborate
        :return: len of object to elaborate
        """
        internal_list = []
        # element_list = []
        with open(self._file_path, newline='') as csvfile:
            file_dict = csv.DictReader(csvfile, delimiter='\t')
            for row in file_dict:
                internal_list.append(row)
                # if self._type_of_extraction == 'interactions':
                #     element_list.append(row['comment'])
                # elif self._type_of_extraction == 'items':
                #     element_list.append(row['description'])
        self._internal_list = internal_list
        return len(internal_list)

    def get_item_from_id(self, idx):
        """
        It gives the sentence to elaborate for a specific row of the file. If the origin of elaboration is from
        interactions, it searches the sentence in the 'comment' column, otherwise if the origin is from item description
        it searches the sentence in the 'description' column
        :param idx: the row from which retrieve the sentence.
        :return:  the sentence as a string, preprocessing is needed
        """
        row = self._internal_list[idx]
        if self._type_of_extraction == 'interactions':
            return row['comment']
        elif self._type_of_extraction == 'items':
            return row['description']

