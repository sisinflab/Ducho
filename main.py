from src.runner.Runner import MultimodalFeatureExtractor
import sys


def main(argv):
    extractor_obj = MultimodalFeatureExtractor(argv=argv)
    extractor_obj.execute_extractions()


if __name__ == '__main__':
    main(sys.argv[1:])
