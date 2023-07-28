import numpy as np
import json
import logging
import socket
import ast
from copy import deepcopy
import os
from pathlib import Path
from stat import S_IREAD, S_IROTH, S_IRGRP
import sys

import utilities as ut

if __name__ == '__main__':
    logger = logging.getLogger()
    logger.handlers = [logging.StreamHandler(sys.stdout)]
else:
    logger = logging.getLogger(__name__)
logger.level = logging.INFO

HOSTNAME = socket.gethostname()


# separators to create the run name from the run arguments
arg_sep = '--' # separator between arguments
value_sep = '__' # separator between an argument and its value



####################################
### OPERATIONS WITH RUN METADATA ###
####################################

def make_run_name(run_id:int, **kwargs) -> str:
    folder = f'{run_id}{arg_sep}'
    for k in sorted(kwargs):
        if k == 'load_from': # skip putting load_from in the name as it messes it
            continue
        folder += f'{k}{value_sep}{kwargs[k]}{arg_sep}'
    folder = folder[:-len(arg_sep)] # remove the last arg_sep
    folder = ut.make_safe(folder)
    return folder

def parse_run_name(run_name:str, evaluate=False) -> dict:
    '''
    Parses a string into a dictionary

    Parameters
    ----------
    run_name: str
        run name formatted as *<param_name>_<param_value>__*
    evaluate : bool, optional
        whether to try to evaluate the string expressions (True), or leave them as strings (False).
        If unable to evaluate an expression, it is left as is
    
    Returns
    -------
    dict
    
    Examples
    --------
    >>> parse_run_name('a__5--b__7')
    {'a': '5', 'b': '7'}
    >>> parse_run_name('test_arg__bla--b__7')
    {'test_arg': 'bla', 'b': '7'}
    >>> parse_run_name('test_arg__bla--b__7', evaluate=True)
    {'test_arg': 'bla', 'b': 7}
    '''
    d = {}
    args = run_name.split(arg_sep)
    for arg in args:
        if value_sep not in arg:
            continue
        key, value = arg.rsplit(value_sep,1)
        if evaluate:
            try:
                value = ast.literal_eval(value)
            except:
                pass
        d[key] = value
    return d

def remove_args_at_default(run_args: dict, config_dict_flat: dict) -> dict:
    '''
    Removes from a dictionary of parameters the values that are at their default one.

    Parameters
    ----------
    run_args : dict
        dictionary where each item is a dictionary of the arguments of the run
    config_dict_flat : dict
        flattened config dictionary with the default values

    Returns
    -------
    dict
        epurated run_args
    '''
    _run_args = deepcopy(run_args)
    for k,args in _run_args.items():
        new_args = {}
        for arg,value in args:
            if value != config_dict_flat[arg]:
                new_args[arg] = value
        _run_args[k] = new_args
    return _run_args


class Trainer(object):
    def __init__(self) -> None:
        pass

    def schedule(self, **kwargs):
        '''
        Here kwargs can be iterables. This function schedules several runs and calls on each of them `self._run`
        You can also set telegram kwargs with this function.

        Special arguments:
            first_from_scratch : bool, optional
                Whether the first run should be created from scratch or from transfer learning, by default False (i.e. by default transfer learning)
        '''
        first_from_scratch = kwargs.pop('first_from_scratch', False)  # this argument is removed from the kwargs because it affects only the first run
        
        # detect variables over which to iterate
        iterate_over = [] # list of names of arguments that are lists and so need to be iterated over
        non_iterative_kwargs = {} # dictionary of provided arguments that have a single value
        for k,v in kwargs.items():
            if k not in self.config_dict_flat:
                raise KeyError(f'Invalid argument {k}')
            if k in self.telegram_kwargs: # deal with telegram arguments separately
                self.telegram_kwargs[k] = v
                continue
            iterate = False
            if isinstance(v, list): # the argument is a list: possible need to iterate over the argument
                if isinstance(self.config_dict_flat[k], list): # the default value is a list as well, so maybe we don't need to iterate over v
                    if isinstance(v[0], list): # v is a list of lists: we need to iterate over it
                        iterate = True
                else:
                    iterate = True
            if iterate:
                iterate_over.append(k)
            elif v != self.config_dict_flat[k]: # skip parameters already at their default value
                non_iterative_kwargs[k] = v


        # rearrange the order of the arguments over which we need to iterate such that the runs are performed in the most efficient way
        # namely we want arguments for loading data to tick like hours, arguments for preparing X,Y to tick like minutes and arguments for k_fold_cross_val like seconds
        new_iterate_over = []
        # arguments for loading fields
        to_add = []
        for k in iterate_over:
            if k in self.default_run_kwargs['load_data_kwargs']:
                to_add.append(k)
        new_iterate_over += to_add
        for k in to_add:
            iterate_over.remove(k)
        # arguments for preparing XY
        to_add = []
        for k in iterate_over:
            if k in self.default_run_kwargs['prepare_XY_kwargs']:
                to_add.append(k)
        new_iterate_over += to_add
        for k in to_add:
            iterate_over.remove(k)
        # remaining arguments
        new_iterate_over += iterate_over
        
        iterate_over = new_iterate_over

        # retrieve values of the arguments
        iteration_values = [kwargs[k] for k in iterate_over]
        # expand the iterations into a list performing the meshgrid
        iteration_values = ut.zipped_meshgrid(*iteration_values)
        # ensure json serializability by converting to string and back
        iteration_values = ast.literal_eval(str(iteration_values))

        # add the non iterative kwargs
        self.scheduled_kwargs = [{**non_iterative_kwargs, **{k: l[i] for i,k in enumerate(iterate_over) if l[i] != self.config_dict_flat[k]}} for l in iteration_values]

        ## this block of code does exactly the same of the previous line but possibly in a clearer way
        # self.scheduled_kwargs = []
        # for l in iteration_values:
        #     self.scheduled_kwargs.append(non_iterative_kwargs) # add non iterative kwargs
        #     # add the iterative kwargs one by one checking if they are at their default value
        #     for i,k in enumerate(iterate_over):
        #         v = l[i]
        #         if v != self.config_dict_flat[k]: # skip parameters at their default value
        #             self.scheduled_kwargs[-1][k] = v

        if len(self.scheduled_kwargs) == 0: # this is fix to avoid empty scheduled_kwargs if it happens there are no iterative kwargs
            self.scheduled_kwargs = [non_iterative_kwargs]
            if len(non_iterative_kwargs) == 0:
                logger.info('Scheduling 1 run at default values')
            else:
                logger.info(f'Scheduling 1 run at values {non_iterative_kwargs}')
        else:
            logger.info(f'Scheduled the following {len(self.scheduled_kwargs)} runs:')
            for i,kw in enumerate(self.scheduled_kwargs):
                logger.info(f'{i}: {kw}')

        if first_from_scratch: 
            self.scheduled_kwargs[0]['load_from'] = None # disable transfer learning for the first run
            logger.warning('Forcing the first run to be loaded from scratch')
    
    def run_multiple(self):
        '''
        Performs all the scheduled runs
        '''
        # add telegram logger
        th = self.telegram(**self.telegram_kwargs)
        nruns = len(self.scheduled_kwargs)
        logger.log(45, f"Starting {nruns} run{'' if nruns == 1 else 's'}")
        with ut.TelegramLogger(logger, **self.telegram_kwargs):
            for i,kwargs in enumerate(self.scheduled_kwargs):
                logger.log(48, f'{HOSTNAME}: Run {i+1}/{nruns}')
                self._run(**kwargs)
            logger.log(49, f'{HOSTNAME}: \n\n\n\n\n\nALL RUNS COMPLETED\n\n')