# '''
# Created on 2022-01-13

# @author: Alessandro Lovo
# '''
'''
Set of general purpose functions
'''

# import libraries
import os
import numpy as np
import sys
from functools import wraps
from pathlib import Path
import time
from datetime import datetime
import json
import logging
import importlib
from scipy import interpolate

if __name__ == '__main__':
    logger = logging.getLogger()
    logger.handlers = [logging.StreamHandler(sys.stdout)]
else:
    logger = logging.getLogger(__name__)
logger.level = logging.INFO

MAX_FILENAME_LENGTH = 128

######## time formatting ##########
def now():
    '''
    Returns the current time as string formatted as year-month-day hour:minute:second
    '''
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def pretty_time(t):
    '''
    Takes a time in seconds and returns it in a string with the format <hours> h <minutes> min <seconds> s

    Examples
    --------
    >>> pretty_time(124)
    '2 min 4.0 s'
    >>> pretty_time(3601.4)
    '1 h 1.4 s'
    '''
    h = t//3600
    t = t - h*3600
    m = t//60
    s = t - m*60
    pt = ''
    if h > 0:
        pt += f'{h:.0f} h '
    if m > 0:
        pt += f'{m:.0f} min '
    pt += f'{s:.1f} s'
    return pt

default_formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%m/%d/%Y %H:%M:%S')

###### function decorators for logging ###

## indenting ####
indentation_sep = '\t' # spacing amount at each indentation

def indent_write(write):
    '''
    decorator for a function that writes to a stream, e.g. sys.stdout or a file. Indents the message.

    Examples
    --------
    >>> def test():
    ...     print('before')
    ...     old_write = sys.stdout.write
    ...     sys.stdout.write = indent_write(sys.stdout.write)
    ...     print('Hello!')
    ...     sys.stdout.write = old_write
    ...     print('after')

    Will give output
    before
        Hello!
    after
    '''
    @wraps(write)
    def wrapper(message):
        message = (indentation_sep+f'\n{indentation_sep}'.join(message[:-1].split('\n')) + message[-1])
        return write(message)
    return wrapper

def indent(*streams):
    '''
    Returns a decorator that indents the output produced by the decorated function on the streams provided

    Examples
    --------
    >>> @indent(sys.stdout)
    ... def show(a=0):
    ...     print(f'{a = }')
    >>> def test(a=0):
    ...     print('before')
    ...     show(a)
    ...     print('after')
    
    When running `test(24)` you will get
    before
        a = 24
    after

    Indentation can be chained

    >>> @indent(sys.stdout)
    ... def test_innner(a=0):
    ...     print('before inner')
    ...     show(a)
    ...     print('after inner')
    >>> def test_outer(a=0):
    ...     print('before outer')
    ...     test_inner(a)
    ...     print('after outer')

    test_outer(24) will give
    before outer
        before inner
            a = 24
        after inner
    after outer

    You can also indent a handler `h` of the logging module by creating a decorator @indent(h.stream)
    '''
    def wrapper_outer(func):
        @wraps(func)
        def wrapper_inner(*args, **kwargs):
            # save old write and emit functions
            old_write = [stream.write if hasattr(stream, 'write') else None for stream in streams]
            # indent write and emit functions
            for i,stream in enumerate(streams):
                if old_write[i] is not None:
                    stream.write = indent_write(stream.write)
            try:
                r = func(*args, **kwargs)
            finally:
                # restore original functions
                for i,stream in enumerate(streams):
                    if old_write[i] is not None:
                        stream.write = old_write[i]
            return r
        return wrapper_inner
    return wrapper_outer

def indent_logger(logger=None):
    '''
    Indents all handlers of a given logger when the decorated function is running

    Parameters
    ----------
    logger : logging.loggers.Logger, optional
        logger, if None the root logger is used. The default is None
    '''
    if logger is None:
        logger = logging.getLogger()
    if isinstance(logger, str):
        logger = logging.getLogger(logger)
    def wrapper_outer(func):
        @wraps(func)
        def wrapper_inner(*args, **kwargs):
            streams = []
            # get the handlers of the logger and its parents
            c = logger
            while c:
                # # avoid indenting the same stream more than once
                # # in case both a logger and one of its parent log to the same stream, which would be silly anyways
                # _streams = [h.stream for h in c.handlers if hasattr(h, 'stream')]
                # for s in _streams:
                #     if s not in streams:
                #         streams.append(s)

                # assuming the loggers are not silly and so no stream is repeated
                streams = [h.stream for h in c.handlers if hasattr(h, 'stream')]
                if not c.propagate:
                    c = None    #break out
                else:
                    c = c.parent
            
            # save old write functions
            old_write = [stream.write if hasattr(stream, 'write') else None for stream in streams]
            # indent write functions
            for i,stream in enumerate(streams):
                if old_write[i] is not None:
                    stream.write = indent_write(stream.write)
            try:
                r = func(*args, **kwargs)
            finally:
                # restore original functions
                for i,stream in enumerate(streams):
                    if old_write[i] is not None:
                        stream.write = old_write[i]
            return r
        return wrapper_inner
    return wrapper_outer

def indent_stdout(func):
    '''
    Indents the stdout output produced by a function
    '''
    return indent(sys.stdout)(func)

## execution time    

def execution_time(func):
    '''
    Prints the execution time of a function

    Examples
    --------
    >>> logger.handlers = [logging.StreamHandler(sys.stdout)]
    >>> @execution_time
    ... def test(a):
    ...     time.sleep(1)
    ...     logger.info(a)
    >>> test('Hi')
    test:
    Hi
    test: completed in 1.0 s
    '''
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.info(f'{func.__name__}:')
        r = func(*args, **kwargs)
        logger.info(f'{func.__name__}: completed in {pretty_time(time.time() - start_time)}')
        return r
    return wrapper

#### TELEGRAM LOGGER ####

def new_telegram_handler(chat_ID=None, token=None, level=logging.WARNING, formatter=default_formatter, **kwargs):
    '''
    Creates a telegram handler object

    Parameters
    ----------
    chat_ID : int or str or None, optional
        chat ID of the telegram user or group to whom send the logs. If None it is the last used. If str it is a path to a file where it is stored.
        To find your chat ID go to telegram and search for 'userinfobot' and type '/start'. The bot will provide you with your chat ID.
        You can do the same with a telegram group, and, in this case, you will need to invite 'ENSMLbot' to the group.
        The default is None.
    token: str
        token for the telegram bot or path to a text file where the first line is the token
    level : int or logging.(NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL), optional
        The default is logging.WARNING.
    formatter : logging.Formatter, str or None, optional
        The formatter used to log the messages. The default is default_formatter.
        If string it can be for example '%(levelname)s: %(message)s'
    **kwargs :
        additional arguments for telegram_handler.handlers.TelegramHandler

    Returns
    -------
    th: telegram_handler.handlers.TelegramHandler
        handler that logs to telegram
    '''
    import telegram_handler # NOTE: to install this package run pip install python-telegram-handler
    try:
        if token.startswith('~'):
            token = f"{os.environ['HOME']}{token[1:]}"
        with open(token, 'r') as token_file:
            token = token_file.readline().rstrip('\n')
    except FileNotFoundError:
        pass
    if isinstance(chat_ID, str) or isinstance(chat_ID, Path):
        with open(chat_ID, 'r') as chat_ID_file:
            chat_ID = int(chat_ID_file.readline().rstrip('\n'))
    th = telegram_handler.handlers.TelegramHandler(token=token, chat_id=chat_ID, **kwargs)
    if isinstance(formatter, str):
        if formatter == 'default':
            formatter = default_formatter
        else:
            formatter = logging.Formatter(formatter)
    if formatter is not None:
        th.setFormatter(formatter)
    th.setLevel(level)
    return th


########## ARGUMENT PARSING ####################

def run_smart(func, default_kwargs, **kwargs): # this is not as powerful as it looks like
    '''
    Runs a function in a vectorized manner:

    Parameters
    ----------
    func : function with signature func(**kwargs) -> None
    default_kwargs : dict
        default values for the keyword arguments of func
    **kwargs : 
        non default values of the keyword arguments. If a list is provided, the function is run iterating over the list

    Examples
    --------
    >>> def add(x, y=0):
    ...     print(x + y)
    >>> run_smart(add, {'x': 0, 'y': 0}, x=1)
    1
    >>> run_smart(add, {'x': 0, 'y': 0}, x=1, y=[1,2,3]) # iterates over y
    2
    3
    4
    >>> run_smart(add, {'x': 0, 'y': 0}, x=[0, 10], y=[1,2]) # iterates over x and y
    1
    2
    11
    12
    >>> run_smart(add, {'x': [0], 'y': [0]}, x=[1,2], y=[1]) # correctly interprets lists when not supposed to iterate over them
    [1, 2, 1]
    >>> run_smart(add, {'x': [0], 'y': [0]}, x=[1,2], y=[[1], [0]]) # to iterate over list arguments, nest the lists
    [1, 2, 1]
    [1, 2, 0]
    '''
    evaluate = True
    for k,v in kwargs.items():
        if k not in default_kwargs:
            raise KeyError(f'Unknown argument {k}')
        iterate = False
        if isinstance(v, list): # possible need to iterate over the argument
            if isinstance(default_kwargs[k], list):
                if isinstance(v[0], list):
                    iterate = True
            else:
                iterate = True
        if iterate:
            evaluate = False
            for _v in v:
                kwargs[k] = _v
                run_smart(func, default_kwargs, **kwargs)
            break
    if evaluate:
        f_kwargs = default_kwargs
        for k,v in kwargs.items():
            f_kwargs[k] = v
        func(**f_kwargs)

#### JSON IO #########

def json2dict(filename):
    '''
    Reads a json file `filename` as a dictionary

    Returns
    -------
    d : dict
    '''
    with open(filename, 'r') as j:
        d = json.load(j)
    return d

def dict2json(d, filename):
    '''
    Saves a dictionary `d` to a json file `filename`
    '''
    with open(filename, 'w') as j:
        json.dump(d, j, indent=4)
        
def dict2str(d, indent=4, **kwargs):
    '''
    A nice way of printing a nested dictionary
    '''
    return json.dumps(d, indent=indent, **kwargs)

#### MANAGE NESTED DICTIONARIES #####

def collapse_dict(d_nested, d_flat=None):
    '''
    Flattens a nested dictionary `d_nested` into a flat one `d_flat`.

    Parameters
    ----------
    d_nested : dict, can contain dictionaries and other types.
        If a key is present more times the associated values must be the same, otherwise an error will be raised
    d_flat : dict, optional
        flat dictionary into which to store the items of `d_nested`. If None an empty one is created.
        WARNING: If provided the variable passed as d_flat is modified inplace.
        The default is None
    
    Returns
    -------
    d_flat: dict

    Raises
    ------
    ValueError
        if a key appears more than once with different values

    Examples
    --------
    >>> collapse_dict({'a': 10, 'b': {'a': 10, 'c': 4}})
    {'a': 10, 'c': 4}
    >>> collapse_dict({'a': 10, 'b': {'a': 10, 'c': 4}}, d_flat={'a': 10, 'z': 7})
    {'a': 10, 'z': 7, 'c': 4}
    '''
    if d_flat is None:
        d_flat = {}

    for k,v in d_nested.items():
        if isinstance(v, dict):
            d_flat = collapse_dict(v,d_flat)
        else:
            if k in d_flat and v != d_flat[k]:
                raise ValueError(f'Multiple definitions for argument {k}')
            d_flat[k] = v
    return d_flat

def extract_nested(d_nested, key):
    '''
    Method to access items in a nested dictionary

    Parameters
    ----------
    d_nested : dict
        nested dictionary
    key : str
    
    Returns
    -------
    v : Any
        The value corresponding to `key` at the highest hierarchical level

    Raises
    ------
    KeyError
        if `key` is not a key of `d_nested` or the dictionaries inside it at every nested level

    Examples
    --------
    >>> d = {'a': 10, 'b': {'z': 1, 'w': {'q': 20}}}
    >>> extract_nested(d, 'a')
    10
    >>> extract_nested(d, 'b')
    {'z': 1, 'w': {'q': 20}}
    >>> extract_nested(d, 'q')
    20
    '''
    try: 
        return d_nested[key]
    except KeyError:
        for v in d_nested.values():
            if isinstance(v, dict):
                try:
                    return extract_nested(v, key)
                except KeyError:
                    continue
        raise KeyError(f'{key} is not a valid key')

def keys_exists(d_nested, key):
    '''
    Checks if `key` (str) appears in `d_nested` (nested dict) at some level of indentation.
    This is basically like extract_nested() but does not raise the KeyError as the output
    '''
    try: 
        _element = extract_nested(d_nested, key)
        return True
    except KeyError:
        return False


def set_values_recursive(d_nested, d_flat, inplace=False):
    '''
    Given a nested dictionary `d_nested` replaces its values at any level of indentation according to the ones in `d_flat`.
    keys in `d_flat` that do not appear in `d_nested` are ignored.
    If `inplace`, `d_nested` is modified and returned, otherwise a copy is returned (i.e. the variable `d_nested` keeps its original value)

    Examples
    --------
    >>> d = {'a': 10, 'b': {'a': 10, 'c': 8}}
    >>> set_values_recursive(d, {'a': 'hello', 'z': 42}, inplace=True)
    {'a': 'hello', 'b': {'a': 'hello', 'c': 8}}
    >>> d
    {'a': 'hello', 'b': {'a': 'hello', 'c': 8}}
    >>> d = {'a': 10, 'b': {'a': 10, 'c': 8}}
    >>> set_values_recursive(d, {'a': 'hello', 'z': 42}, inplace=False)
    {'a': 'hello', 'b': {'a': 'hello', 'c': 8}}
    >>> d
    {'a': 10, 'b': {'a': 10, 'c': 8}}
    '''
    if len(d_flat) == 0:
        return d_nested
    
    if inplace:
        d_n = d_nested
    else:
        d_n = d_nested.copy()

    for k,v in d_n.items():
        if isinstance(v, dict):
            d_n[k] = set_values_recursive(v, d_flat, inplace=inplace)
        elif k in d_flat:
            d_n[k] = d_flat[k]
    return d_n

def compare_nested(d1, d2):
    '''
    Compares two nested dictionary.
    An item is considered 'added' if it appears in the new version (`d1`) and not in the old one (`d2`) and so on

    Parameters
    ----------
    d1 : dict
        new version
    d2 : dict
        old version

    Returns
    -------
    dict
        dictionary containing the nested keys of the arguments that differ between the two dictionaries, either by being added, removed or changed

    Examples
    --------
    >>> d1 = {'a': 10, 'b': {'e': 10, 'c': 8}, 'z': 'NO'}
    >>> d2 = {'a': 11, 'b': {'e': 11, 'c': 8, 'w': 'hi'}}
    >>> compare_nested(d1, d2)
    {'a': {'old': 11, 'new': 10}, 'b': {'e': {'old': 11, 'new': 10}, 'w': {'removed': 'hi'}}, 'z': {'added': 'NO'}}
    '''
    diff = {}

    for k,v in d1.items():
        if k not in d2:
            diff[k] = {'added': v}
        elif v != d2[k]:
            if isinstance(v, dict) and isinstance(d2[k], dict):
                diff[k] = compare_nested(v, d2[k])
            else:
                diff[k] = {'old': d2[k], 'new': v}

    for k,v in d2.items():
        if k not in d1:
            diff[k] = {'removed': v}

    return diff


### run arguments from folder name ###

def get_run_arguments(run_folder):
    '''
    Retrieves the values of the parameters of a run

    Parameters
    ----------
    run_folder : str
        folder where the run is located, with subfolders containing the folds

    Returns
    -------
    dict
        nested dictionary with the arguments of the run
    '''
    run_folder = run_folder.rstrip('/')
    spl = run_folder.rsplit('/',1)
    if len(spl) == 2:
        root_folder, run_name = spl
    else:
        root_folder = './'
        run_name = spl[-1]
    run_id = run_name.split('--',1)[0]
    runs = json2dict(f'{root_folder}/runs.json')
    try:
        run_id = int(run_id)
        run = runs[str(run_id)]
    except (ValueError, KeyError):
        logger.error(f'{run_name} is not a successful run')
        raise

    config_dict = json2dict(f'{root_folder}/config.json')

    run_config_dict = set_values_recursive(config_dict, run['args'])

    return run_config_dict

#### PERMUTATIONS ####

def invert_permutation(permutation):
    '''
    Inverts a permutation.

    Parameters
    ----------
    permutation : 1D array-like
        permutation of an array of the kind `np.arange(n)` with `n` integer

    Returns
    -------
    np.ndarray
        inverted permutation

    Examples
    --------
    >>> a = np.array([3,4,2,5])
    >>> p = np.random.permutation(np.arange(4))
    >>> a_permuted = a[p]
    >>> p_inverse = invert_permutation(p)
    >>> all(a == a_permuted[p_inverse])
    True
    '''
    return np.argsort(permutation)

def compose_permutations(permutations):
    '''
    Composes a series of permutations

    Parameters
    ----------
    permutations : list
        list of 1D array-like that must be a permutation of an array of the kind `np.arange(n)` with `n` integer and the same for every permutation
    
    Examples
    --------
    >>> a = np.array([3,4,2,5])
    >>> p1 = np.random.permutation(np.arange(4))
    >>> p2 = np.random.permutation(np.arange(4))
    >>> p_composed = compose_permutations([p1,p2])
    >>> a_permuted1 = a[p1]
    >>> a_permuted2 = a_permuted1[p2]
    >>> a_permuted_c = a[p_composed]
    >>> all(a_permuted_c == a_permuted2)
    True
    '''
    l = len(permutations[0])
    for p in permutations[1:]:
        if len(p) != l:
            raise ValueError('All permutations must have the same length')
    ps = permutations[::-1]
    p = ps[0]
    for _p in ps[1:]:
        p = _p[p]
    return p

def zipped_meshgrid(*xi):
    '''
    Creates a zipped meshgrid of a series of iterables. If some of the inputs are multidimensional, they are broadcasted only along the first axis.
    See examples

    Returns
    -------
    list[tuple]
        zipped meshgrid: contains `np.prod([len(x) for x in xi])` tuples of `len(xi)`

    Examples
    --------
    >>> zipped_meshgrid([1,2], [10,11,12])
    [(1, 10), (1, 11), (1, 12), (2, 10), (2, 11), (2, 12)]
    
    >>> zipped_meshgrid([1,2], ['a','b'])
    [(1, 'a'), (1, 'b'), (2, 'a'), (2, 'b')]

    >>> zipped_meshgrid([[1,2], [3,4]], ['a','b'])
    [([1, 2], 'a'), ([1, 2], 'b'), ([3, 4], 'a'), ([3, 4], 'b')]

    >>> zipped_meshgrid(np.array([[1,2], [3,4]]), ['a','b'])
    [(array([1, 2]), 'a'), (array([1, 2]), 'b'), (array([3, 4]), 'a'), (array([3, 4]), 'b')]
    '''
    l = [range(len(x)) for x in xi]

    mesh_i = list(zip(*[m.flatten() for m in np.meshgrid(*l, indexing='ij')]))

    output = [tuple([xi[i][j] for i,j in enumerate(m_i)]) for m_i in mesh_i]

    return output

class Buffer():
    '''
    A simple class for storing a string.
    '''
    def __init__(self):
        self.msg = ''
    def append(self, x):
        x = str(x)
        self.msg += x

def make_safe(path):
    '''
    Replaces square brackets with round ones and removes spaces and ' characters

    Parameters
    ----------
    path : str
        path to be modified

    Returns
    -------
    str
        modified path

    Examples
    --------
    >>> make_safe("tau 5")
    'tau5'
    >>> make_safe("label_field__'t2m'--tau__[0, 1, 2]")
    'label_field__t2m--tau__(0,1,2)'
    '''
    path = path.replace(' ', '')
    path = path.replace('[', '(')
    path = path.replace(']', ')')
    path = path.replace("'", '')

    path_to = None
    if '/' in path:
        path_to, path = path.rsplit('/', 1)
    
    if len(path) > MAX_FILENAME_LENGTH:
        clipped_path = path[:MAX_FILENAME_LENGTH - 3] + '...'
        logger.warning(f'Too long filename\n\t{path}\nClipping to\n\t{clipped_path}')
        path = clipped_path
    if path_to is not None:
        path = f'{path_to}/{path}'
    
    return path


### stuff useful for computing metrics ####

def entropy(p, q=None, epsilon=1e-15):
    '''
    Returns `-p*log(max(q, epsilon)) - (1-p)*log(max(1-q, epsilon))`

    If q is None, q = p
    '''
    epsilon = np.float64(epsilon)
    if 1 - epsilon == 1:
        raise ValueError('Too small epsilon')

    if q is None:
        q = p
    return -p*np.log(np.maximum(q, epsilon)) - (1-p)*np.log(np.maximum(1-q, epsilon))

def unbias_probabilities(Y_pred_prob, u=1):
    '''
    Removes the bias in probabilities due to undersampling

                            u P[Y=0]_biased
    P[Y=0]_unbiased = ----------------------------
                       1 - (1 - u) P[Y=0]_biased
    
    Parameters
    ----------
    Y_pred_prob : np.ndarray of shape (n, 2)
        Array of probabilities computed on the undersampled dataset. Assumes Y_pred_prob[:,0] == 1 - Y_pred_prob[:,1] and Y_pred_prob[:,0] are the probabilities of being in the majority class, the une that has been undersampled
    u : float >= 1, optional
        undersampling factor, by default 1
    epsilon : float, optional
        probailities will be clipped to be in [epsilon, 1-epsilon], by default 1e-15

    Returns
    -------
    np.ndarray of shape (n, 2)
        Array of unbiased probabilities

    Raises
    ------
    ValueError
        If u < 1
    '''

    if u == 1:
        return Y_pred_prob
    elif u < 1:
        raise ValueError('u must be >= 1')
    Y_unb = np.zeros_like(Y_pred_prob, dtype=np.float64)
    Y_unb[:,0] = u*Y_pred_prob[:,0]/(1 - (1 - u)*Y_pred_prob[:,0])
    Y_unb[:,1] = 1 - Y_unb[:,0]

    return Y_unb

### IMPORT MODULE FROM A FILE ###

def module_from_file(module_name, file_path): 
    '''
    The code that imports the file which originated the training with all the instructions
    
    Parameters
    ----------
    module_name : str
        The name we give to the imported module
    file_path : str
        The path to the python file containing the module

    Returns
    -------
    module

    Examples
    --------
    foo = module_from_file("foo", 'models/Funs.py')
    
    Potential Problems
    --------
        When used with files generated by Learn2_new.py we get an error:
            this_module = sys.modules[__name__]
            KeyError: ???foo???
    '''
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


### OTHER GENERAL PURPOSE STUFF ###

class DelayedInitWrapper(object):
    '''
    Wrapper for handling delayed initialization of a class:
        The class is initialized the first time one tries to acces one of its attributes

    Parameters
    ----------
    constructor : callable
        Constructor of the class
    *args, **kwargs: arguments to pass to the constructor

    Examples:
    ---------
    >>> class Test():
    ...     def __init__(self, a, b=1):
    ...         print('Initializing Test')
    ...         self.a = a
    ...         self.b = b

    >>> a = 0
    >>> t = DelayedInitWrapper(Test, a, b=42) # the Test object is not created yet

    >>> print(t.a) # the first time you try to access an attribute of the class, it is created
    Initializing Test
    0

    >>> print(t.a) # now it behaves like a normal Test object
    0

    '''
    
    wrargs = ['obj', '_constructor', '_init_args', '_init_kwargs']
    
    def __init__(self, constructor, *args, **kwargs):
        self.obj = None
        self._constructor = constructor
        self._init_args = args
        self._init_kwargs = kwargs
        
    
    def __getattribute__(self, name):
        if name in object.__getattribute__(self,'__dict__'):
        # if name in object.__getattribute__(self,'__dir__')():
            return object.__getattribute__(self,name)
        if self.obj is None:
            # print(f'Called {name}')
            self.obj = self._constructor(*self._init_args, **self._init_kwargs)
        return self.obj.__getattribute__(name)
    
    # def __getattr__(self, name):
    #     # if name in ['obj', '_constructor', '_init_args', '_init_kwargs']:
    #     #     return self.__dict__[name]
    #     if self.obj is None:
    #         print(f'Called {name}')
    #         self.obj = self._constructor(*self._init_args, **self._init_kwargs)
    #     return self.obj.__getattribute__(name)
    
    
    def __setattr__(self, name, value):
        if name in DelayedInitWrapper.wrargs:
            object.__setattr__(self, name, value)
            return
        if self.obj is None:
            print(f'Called {name}')
            self.obj = self._constructor(*self._init_args, **self._init_kwargs)
        self.obj.__setattr__(name, value)


def adaptive_interpolation(func, x_range, max_xstep=0.1, max_ystep_rel=0.1, verbose=False, **kwargs):
    '''
    Interpolates a 1d function using an adaptive grid for computing it,
    ensuring a better precision than with a fixed grid of points.

    Parameters
    ----------
    func : callable with signature y = func(x)
        Function to be interpolated
    x_range : tuple
        range in which to interpolate the function
    max_xstep : float, optional
        Maximum x-separation between consecutive points. The default is 0.1.
    max_ystep_rel : float, optional
        Maximum relative change of y between consecutive points, i.e. |1 - y[i-1]/y[i]|. The default is 0.1.
    verbose : bool, optional
        The default is False.
        
    **kwargs : 
        additional arguments to pass to scipy.interpolate.interp1d

    Returns
    -------
    xs : np.ndarray
        x points (sorted)
    ys : np.ndarray
        y points corresponding to xs
    yfunc : function with signature y = yfunc(x)
        interpolating function

    '''
    if 'kind' not in kwargs:
        kwargs['kind'] = 'cubic'
    
    xs = list(np.linspace(x_range[0], x_range[1], int((x_range[1] - x_range[0])/max_xstep)))
    
    ys = [func(x) for x in xs]
    
    i = len(ys) - 1
    while i > 0:
        if verbose:
            print(f'\r{i = } out of {len(ys)}         ', end='')
        if np.abs((ys[i] - ys[i - 1])/ys[i]) > max_ystep_rel and np.abs(xs[i] - xs[i - 1]) > 1e-8:
            x = (xs[i] + xs[i - 1])/2
            xs.insert(i, x)
            # print(f'{x = }')
            ys.insert(i, func(x))
            i += 1
        else:
            i -= 1
            
    yfunc = interpolate.interp1d(xs, ys, **kwargs)
    return np.array(xs), np.array(ys), yfunc


def average_with_significance(x:np.ndarray, axis=0, significance_level=0):
    '''
    Computes the average and the significance of an array along a specific axis.

    significance is defined as mean/std

    Parameters
    ----------
    x : np.ndarray
        
    axis : int, optional
        axis over which to perform the operation, by default 0
    significance_level : float, optional
        level of significance that discriminates between significant and non significant data, by default 0

    Returns
    -------
    x_mean : np.ndarray
        mean
    significance : np.ndarray
        significance
    significance_mask : np.ndarray
        `|significance| > significance_level`
    '''
    logger.info(f'Performing mean over {x.shape[axis]} samples.')
    x_mean = x.mean(axis=axis)
    x_std = x.std(axis=axis)
    significance_mask = np.abs(x_mean) > significance_level*x_std
    significance = x_mean/x_std

    return x_mean, significance, significance_mask

def significative_data(data, t_values=None, t=None, both=False, default_value=0):
    '''
    Filters `data` depending whether `t_values` exceeds a threshold `t`

    Parameters
    ----------
    data : np.ndarray
        data
    t_values : np.ndarray, optional
        significance values, by default None
    t : float (>=0), optional
        significance threshold, by default None
    both : bool, optional
        whether to return also the non significant data, by default False
    default_value : float, optional
        value to assign to non significant data, by default 0

    Returns
    -------
    Out_taken : np.ndarray
        data where the non significant values are set to `default_value`
    Out_not_taken : np.ndarray, returned only if `both == True`
        data where the significant values are set to `default_value` (complementary to `Out_taken`)
    N_points_taken : int
        number of significant datapoints
    '''
    if data is None:
        if both:
            return None, 0, 0
        else:
            return None, 0
    
    data = np.array(data)
    
    if t_values is None or t is None:
        logger.warn('Assuming all data are significant')
        if both:
            return data, np.ones_like(data)*default_value, np.product(data.shape)
        else:
            return data, np.product(data.shape)
        
    t_values = np.array(t_values)
    if data.shape != t_values.shape:
        raise ValueError('Shape mismatch')

    Out_taken = data.copy()

    mask = t_values >= t
    N_points_taken = np.sum(mask)
    Out_taken[np.logical_not(mask)] = default_value
    
    if both:
        Out_not_taken = data.copy()
        Out_not_taken[mask] = default_value
        return Out_taken, Out_not_taken, N_points_taken
    else:
        return Out_taken, N_points_taken