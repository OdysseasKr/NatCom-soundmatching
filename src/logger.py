import os
import datetime
import json
import numpy as np


class Logger:
    def __init__(self, path, description=""):
        super().__init__()
        self.date = str(datetime.datetime.now()).replace(" ", "-")
        self.path = os.path.join(path, self.date + '.json')
        if not os.path.exists(path):
            os.mkdir(path)
        self.log = {'description': description, 'targets': []}
        self._curr_target = None

    def set_header(self, header_parameters):
        for k in header_parameters:
            self.log[k] = header_parameters[k]

    def set_target(self, target):
        self._curr_target = len(self.log['targets'])
        self.log['targets'].append({'target': target, 'runs': []})

    def add_run(self, best, best_fit, n_gens, early_stopping, runtime):
        if self._curr_target is None:
            raise AssertionError('No target so far. ' +
                                 'Can\'t add a new run for an unknown target')

        run_obj = {
            'best': best,
            'best_fit': float(best_fit),
            'n_gens': n_gens,
            'early_stopping': early_stopping,
            'runtime': runtime
        }

        self.log['targets'][self._curr_target]['runs'].append(run_obj)

    def calculate_metrics(self):
        metrics = {}
        metrics['mean_fitness'] = self._mean_fitness()
        # TODO: Add 2 more metrics
        self.log['metrics'] = metrics

    def _mean_fitness(self):
        fitness = 0
        for t in self.log['targets']:
            target_fitness = [r['best_fit'] for r in t['runs']]
            fitness += np.mean(target_fitness) / len(self.log['targets'])
        return fitness

    def close(self):
        self.calculate_metrics()
        with open(self.path, 'w') as outfile:
            json.dump(self.log, outfile)
