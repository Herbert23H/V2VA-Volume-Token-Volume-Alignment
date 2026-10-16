import argparse
import logging
import pprint

import yaml

from evals.scaffold import draw_main as eval_main


parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str,
    help='name of config file to load',
    default='configs.yaml')
parser.add_argument(
    '--device', type=str, default='cuda:0',
    help='single device for drawing, e.g. cuda:0 / cpu')


def main():
    args = parser.parse_args()

    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logger.info(f'called-params {args.fname}')

    with open(args.fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)

    params.setdefault('draw', {})
    params['draw']['device'] = args.device

    logger.info('loaded params...')
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(params)

    eval_main(params['eval_name'], args_eval=params)


if __name__ == '__main__':
    main()
