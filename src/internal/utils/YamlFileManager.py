import os
import yaml


class YamlFileManager:
    def __init__(self, yaml_file_path):
        """
        It manages a general Yaml file, it is created with the scope to only read this kind of file

        :param yaml_file_path: the path of the file as a String. It is preferred to be the absolute path.

        """
        self._correct_yaml_file_path(yaml_file_path)

    def _correct_yaml_file_path(self, old_path):
        """
        It corrects the path to the absolute one.
        if old_path links to a directory the method search a 'yaml' file in the directory. Otherwise, if it points to a
        file, all is fine. Else the method try to correct the path in a working one, if it fails raise an error
        Args:
            old_path: the path given from the user. It is a String

        Returns:
            it returns nothing but set the _yaml_file_path that points directly to the yaml file. It is a String

        """
        # the path can be:
        # - a path only to the directory
        # - a complete path to a yml/yaml, in this case must be verified that the extension is correct
        if os.path.isdir(old_path):
            # search through the directory a file with the correct extension
            dir_list = os.listdir(old_path)
            for file in dir_list:
                # the extensions can be both .yml or .yaml
                if file[-4:] == '.yml' or file[-5:] == '.yaml':
                    self._yaml_file_path = os.path.join(old_path, file)
                    return
        elif os.path.exists(old_path):
            # the path points directly to the file, all is fine
            self._yaml_file_path = old_path
        else:
            # in this case an error has occurred, thanks to the 2 possible extension
            # maybe the user wrote .yml but the correct extension is .yaml or the opposite
            if os.path.exists(old_path[-3:] + 'yaml'):
                self._yaml_file_path = old_path[-3:] + 'yaml'
            elif os.path.exists(old_path[-4:] + 'yml'):
                self._yaml_file_path = old_path[-4:] + 'yml'
            else:
                # it is impossible to find the config file
                raise FileNotFoundError('the path given is wrong: ' + old_path)

    def get_raw_dict(self):
        """
            it simply loads the data contained in the file
            Returns:
                the String that is contained in the yaml file
        """
        # there is no need here to raise an exception if the file is not found
        # since the os raises it autonomously
        with open(self._yaml_file_path, 'r') as file:
            return yaml.safe_load(file)

