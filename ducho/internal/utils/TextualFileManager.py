import os
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

    def initiate_element_list_and_get_len(self):
        """
        Read the file, instantiate the internal list of its contents, and return the number of sentences to be elaborated.

        Returns:
            Tuple[int, List[Any]]: A tuple containing the length of the object to be elaborated and a list of IDs.
        """
        self._id_column = self._columns[0]
        self._text_column = self._columns[1]
        df = self._load_textual_dataframe()
        self._internal_list = df
        ids_list = df[self._id_column].tolist() if type(self._id_column) != list else \
            list(zip(df[self._id_column[0]].tolist(), df[self._id_column[1]].tolist()))
        return len(self._internal_list), ids_list

    def _build_dtype_map(self):
        """
        Build dtype hints for textual identifiers and text columns.

        Returns:
            dict: A dictionary mapping required columns to `str` dtype.
        """
        dtype_map = {self._text_column: str}
        if isinstance(self._id_column, list):
            for col in self._id_column:
                dtype_map[col] = str
        else:
            dtype_map[self._id_column] = str
        return dtype_map

    def _detect_input_format(self):
        """
        Detect the textual input format from file extension.

        Returns:
            str: One among {'tsv', 'hdf', 'parquet'}.
        """
        extension = os.path.splitext(self._file_path)[1].lower()

        if extension == '.tsv':
            return 'tsv'
        if extension in ['.h5', '.hdf5', '.hdf']:
            return 'hdf'
        if extension == '.parquet':
            return 'parquet'

        raise ValueError(
            f"Unsupported textual input format '{extension}'. Supported formats: .tsv, .h5/.hdf5/.hdf, .parquet"
        )

    def _read_hdf_with_fallback(self):
        """
        Read an HDF5 file. If no key is explicitly provided in the file call, try the first available key.

        Returns:
            pandas.DataFrame: Data read from the HDF5 file.
        """
        try:
            return pd.read_hdf(self._file_path)
        except (ValueError, KeyError):
            with pd.HDFStore(self._file_path, mode='r') as store:
                keys = store.keys()
            if not keys:
                raise ValueError(f"No dataset key found in HDF5 file: {self._file_path}")
            return pd.read_hdf(self._file_path, key=keys[0])

    def _ensure_required_columns_are_string(self, dataframe):
        """
        Validate required columns and cast them to string.

        Args:
            dataframe (pandas.DataFrame): Input dataframe.

        Returns:
            pandas.DataFrame: Dataframe with casted required columns.
        """
        required_columns = [self._text_column]
        if isinstance(self._id_column, list):
            required_columns.extend(self._id_column)
        else:
            required_columns.append(self._id_column)

        missing_columns = [col for col in required_columns if col not in dataframe.columns]
        if missing_columns:
            raise KeyError(f"Missing required textual columns in input file: {missing_columns}")

        for col in required_columns:
            dataframe[col] = dataframe[col].astype(str)

        return dataframe

    def _load_textual_dataframe(self):
        """
        Load textual data according to the input file format.

        Returns:
            pandas.DataFrame: Loaded textual dataframe.
        """
        input_format = self._detect_input_format()
        dtype_map = self._build_dtype_map()

        if input_format == 'tsv':
            return pd.read_csv(self._file_path, sep='\t', dtype=dtype_map)
        if input_format == 'hdf':
            dataframe = self._read_hdf_with_fallback()
            return self._ensure_required_columns_are_string(dataframe)
        if input_format == 'parquet':
            dataframe = pd.read_parquet(self._file_path)
            return self._ensure_required_columns_are_string(dataframe)

        raise ValueError(f"Unsupported textual input format: {input_format}")

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
