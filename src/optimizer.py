import argparse
import logging
import os
from mlboardclient.api import client
from mlboardclient.api.v2 import optimizator
import json

logging.basicConfig(
    format='%(asctime)s %(levelname)-10s %(name)-25s [-] %(message)s',
    level='INFO'
)
SUCCEEDED = 'Succeeded'
FAILED = 'Failed'
LOG = logging.getLogger('INFO')


def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.debug:
        logging.root.setLevel('DEBUG')

    m = client.Client()
    app = m.apps.get()
    task = app.task('train')
    task.resource('worker')['command'] = 'python mnist.py --training_iteration=1000 --version 0'
    spec = (optimizator.ParamSpecBuilder().resource('worker')
            .param('fully_neurons')
            .int()
            .bounds(1,3)
            .param('drop_out')
            .bounds(0.2,0.7)
            .build())
    LOG.info('Run with param spec = %s', spec)
    result = task.optimize(
        'test_accuracy',
        spec,
        init_steps=args.init_steps,
        iterations=args.iterations,
        method=args.method,
        max_parallel=args.parallel,
        direction='maximize'
    )
    best = result['best']
    LOG.info('Found best build %s:%s: %.2f', best.name,best.build,best.exec_info['test_accuracy'])
    LOG.info('Exporting model to catalog mnist/%s',best.build)
    export = app.task('export')
    export.resource('run')['command'] = 'python mnist.py'
    export.resource('run')['args']= {
        'mode': 'export',
        'catalog_name': 'mnist',
        'build': best.build,
        'model_version': 1
    }
    export.start()
    export.wait()



def get_parser():
    parser = argparse.ArgumentParser(
        description='Calculate steps'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging',
    )
    parser.add_argument(
        '--init_steps',
        type=int,
        default=5,
        help='Number of init steps'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='skopt',
        help='Optimization method'
    )
    parser.add_argument(
        '--parallel',
        type=int,
        default=1,
        help='How many task will be executed in paralle'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=5,
        help='Number of iterations'
    )
    return parser

if __name__ == '__main__':
    main()
