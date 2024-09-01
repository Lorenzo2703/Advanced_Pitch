import argparse
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

from basic_pitch.dataset import commandline
from basic_pitch.dataset.guitarset import main as guitarset_main
from basic_pitch.dataset.slakh import main as slakh_main
from basic_pitch.dataset.hwd import main as hwd_main
from basic_pitch.dataset.dagstuhl import main as dagstuhl_main

DATASET_DICT = {
    'guitarset': guitarset_main,
    'dagstuhl': dagstuhl_main,
    'slakh': slakh_main,
    'hwd': hwd_main,
}


def main():
    dataset_parser = argparse.ArgumentParser()
    dataset_parser.add_argument("--dataset", choices=list(DATASET_DICT.keys()), help="The dataset to download / process.")
    args, remaining_args = dataset_parser.parse_known_args()
    dataset = args.dataset
    logger.info(f"Downloading and processing {dataset}")

    cl_parser = argparse.ArgumentParser()
    commandline.add_default(cl_parser, dataset)
    commandline.add_split(cl_parser)
    known_args, pipeline_args = cl_parser.parse_known_args(remaining_args)
    for arg in vars(known_args):
        logger.info(f"known_args:: {arg} = {getattr(known_args, arg)}")
    logger.info(f"pipeline_args = {pipeline_args}")

    DATASET_DICT[dataset](known_args, pipeline_args)


if __name__ == '__main__':
    main()
