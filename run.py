from ducho.runner.Runner import MultimodalFeatureExtractor
import sys


def main(config, additional_argv):
    extractor_obj = MultimodalFeatureExtractor(config_file_path=config.split("=")[1], argv=additional_argv)
    extractor_obj.execute_extractions()


if __name__ == '__main__':
    main(config=sys.argv[1], additional_argv=sys.argv[1:])
