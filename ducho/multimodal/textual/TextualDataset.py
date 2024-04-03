import os
import re
import numpy
from ducho.internal.father_classes.DatasetFather import DatasetFather
from ducho.internal.utils.TextualFileManager import TextualFileManager


# the following function is not called right now. but it will be needed in the future
def complex_spit_of_list_of_string(sample, splitter):
    sample_list = []
    for el in sample:
        temp = el.split(splitter)
        for sentence in temp[:-1]:
            sentence = sentence + splitter
            sample_list.append(sentence)
        # now append the last that was excluded in the for each
        sample_list.append(temp[-1])
    return sample_list


class TextualDataset(DatasetFather):
    """
    This class represents the Textual Dataset used for the data loading process.
    """
    def __init__(self, input_directory_path, output_directory_path, columns=None):
        """
        It manages the Textual Dataset, which consists of a folder containing input data and another folder for output data.
        It handles the preprocessing of input data and manages the output data.

        Args:
            input_directory_path: Folder of the input data to elaborate as String.
            output_directory_path: Folder of where put Output as String, it will be created if it does not exist.
            columns: Columns for the textual modality.

        Returns:
            None
        """

        super().__init__(input_directory_path, output_directory_path, model_name=None)
        self._text_to_be_cleaned = False
        self._textual_file_manager = TextualFileManager(columns=columns)
        self._ids = None
        # if num_sample is 1, it means it have to be the num of sample in the single file
        # in this case the textual file manager have to behave accordingly
        if self._num_samples == 1:
            self._prepare_environment_for_single_file_extractions()

    def _prepare_environment_for_single_file_extractions(self):
        """
        It prepares the env to utilize only one file the runner cycles through the num samples. if there is only one file
        the num samples is the number of row of the file. Right now this is the only choice, but in the future maybe a
        user will need to give different files, so this func is accommodated to build this kind of login in the future.

        Returns:
            None
        """
        if self._filenames[0] == '':
            file_path = self._input_directory_path
        else:
            file_path = os.path.join(self._input_directory_path, self._filenames[0])
        self._textual_file_manager.set_file_path(file_path)
        self._num_samples, self._ids = self._textual_file_manager.initiate_element_list_and_get_len()

    def __getitem__(self, index):
        """
        It retrieves a sample preprocessed given its id. Only in the Textual case the id refers to the row of the file.

        Args:
            index: It is the index in the filenames list from which extract the name of te file to elaborate.

        Returns:
            a String which contains the data of the file. It may be processed and cleaned
        """
        return self._pre_processing(self._textual_file_manager.get_item_from_id(self._ids[index])), self._ids[index]

    def _pre_processing(self, sample):
        """
        It cleans the String.

        Args:
            sample: String to clean.

        Returns:
             Cleaned String
        """
        # the following code is inspired by:
        # https://github.com/JarenceSJ/ReviewGraph/blob/main/nlp_util.py#L123
        if self._text_to_be_cleaned:
            sample = re.sub(r"[^A-Za-z0-9',.!;?()]", " ", sample)

            sample = re.sub(r"\.", " . ", sample)
            sample = re.sub(r"!+", " ! ", sample)
            sample = re.sub(r",", " , ", sample)
            sample = re.sub(r";", " ; ", sample)
            sample = re.sub(r"\\", " \\ ", sample)
            sample = re.sub(r"!", " ! ", sample)
            sample = re.sub(r"\(", " ( ", sample)
            sample = re.sub(r"\)", " ) ", sample)
            sample = re.sub(r"\?", " ? ", sample)

            sample = re.sub(r"\s{2,}", " ", sample)
            sample = re.sub(r"(\.|\s){7,}", " ... ", sample)
            sample = re.sub(r"(?<= )(\w \. )+(\w \.)", lambda x: x.group().replace(" ", ""), sample)
            # sample = re.sub(r"(\.|\s){4,}", " ... ", sample)

            sample = re.sub(r"\'s", " \'s", sample)
            sample = re.sub(r"\'ve", " \'ve", sample)
            sample = re.sub(r"n\'t", " n\'t", sample)
            sample = re.sub(r"\'re", " \'re", sample)
            sample = re.sub(r"\'d", " \'d", sample)
            sample = re.sub(r"\'m", " \'m", sample)
            sample = re.sub(r"\'ll", " \'ll", sample)

            # sample = re.sub(r"[^A-Za-z0-9']", " ", sample)
            sample = re.sub(
                r"(?!(('(?=s\b))|('(?=ve\b))|('(?=re\b))|('(?=d\b))|('(?=ll\b))|('(?=m\b))|((?<=n\b)'(?=t\b))))'",
                " ", sample)

            # Glove style
            # sample = re.sub(' [0-9]{5,} ', ' ##### ', sample)
            # sample = re.sub(' [0-9]{4} ', ' #### ', sample)
            # sample = re.sub(' [0-9]{3} ', ' ### ', sample)
            # sample = re.sub(' [0-9]{2} ', ' ## ', sample)
            sample = re.sub(' 0 ', ' zero ', sample)
            sample = re.sub(' 1 ', ' one ', sample)
            sample = re.sub(' 2 ', ' two ', sample)
            sample = re.sub(' 3 ', ' three ', sample)
            sample = re.sub(' 4 ', ' four ', sample)
            sample = re.sub(' 5 ', ' five ', sample)
            sample = re.sub(' 6 ', ' six ', sample)
            sample = re.sub(' 7 ', ' seven ', sample)
            sample = re.sub(' 8 ', ' eight ', sample)
            sample = re.sub(' 9 ', ' nine ', sample)

            sample = re.sub(r"\s{2,}", " ", sample)
            sample.strip().lower()

        return sample

    def set_preprocessing_flag(self, preprocessing_flag):
        assert type(preprocessing_flag) == bool
        self._text_to_be_cleaned = preprocessing_flag

    def set_type_of_extraction(self, type_of_extraction):
        """
        It set the origin of the data, from item or users interactions, it is needed later to read correctly the tsv.

        Args:
             type_of_extraction: 'items' or 'interactions'.

        Returns:
            None
        """
        self._textual_file_manager.set_type_of_extraction(type_of_extraction)

    def create_output_file(self, input_batch, extracted_data, model_layer, fusion=None):
        """
        Overwrites the method of the Father class because all the Strings come from the same file, and it only changes
        the row.

        Args:
            index: It indicates the row of the String.
            extracted_data: The output to put in the file.
            model_layer: The layer used, it is a String, it will be shown on the final name.
            fusion: Fusion is not used for TextualDataset's objects.

        Returns:
            None
        """
        filenames = input_batch[1]
        
        # generate output path
        framework = self._backend_libraries_list[0]
        output_path = os.path.join(self._output_directory_path, framework)
        output_path = os.path.join(output_path, self._model_name)
        output_path = os.path.join(output_path, str(model_layer))
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # checking whether batch size is > 1.
            
        if len(extracted_data) > 1:
            output_file_name = [f + '.npy' for f in filenames]
            for f, e in zip(output_file_name, extracted_data):
                path = os.path.join(output_path, f)
                e = numpy.expand_dims(e, axis=0)
                numpy.save(path, e)
        else:
            output_file_name = filenames[0] + '.npy'
            path = os.path.join(output_path, output_file_name)
            numpy.save(path, extracted_data)
